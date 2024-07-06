import torch.nn as nn
import torch
from .token_performer import Token_performer
from .Transformer import saliency_token_inference, token_TransformerEncoder
import matplotlib.pyplot as plt


class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, temb, text_emb):
        B, _, _ = fea.shape
        fea = self.mlp(self.norm(fea)) # B,1024,384

        # fea = torch.cat((saliency_tokens, fea), dim=1) # B,1025,384
        fea = torch.cat((fea, temb), dim=1) # B,1026,384
        fea = torch.cat((fea, text_emb), dim=1) # B,1027,384

        fea = self.encoderlayer(fea) # B,1027,384
        # saliency_tokens = fea[:, 0, :].unsqueeze(1) # B,1,384
        # fea = fea[:,1:-1,:]

        saliency_fea = self.saliency_token_pre(fea) # B,1024,384

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea)) # B,1024,64

        return saliency_fea, fea


class decoder_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=[32, 1024], ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size[0] // ratio,  img_size[1] // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse

        if self.fuse:
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                    nn.Linear(dim, token_dim),
                    nn.GELU(),
                    nn.Linear(token_dim, token_dim),
                )
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim

    def forward(self, dec_fea, enc_fea=None,co_tokens=None):

        # from 384 to 64
        if self.fuse:
            dec_fea = self.mlp(self.norm(dec_fea))
        # if enc_fea.shape[2] == 384:
        #     enc_fea = self.mlp(self.norm(enc_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)

        return dec_fea

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h.squeeze()


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

dwt_module = DWT()
idwt_module = IWT()

class Decoder(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=[32, 1024]):

        super(Decoder, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder4 = decoder_module(dim=embed_dim, token_dim=token_dim,  img_size=img_size, ratio=1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=False)

        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_2 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 2)
        self.pre_1_8 = nn.Linear(token_dim, 2)
        self.pre_1_4 = nn.Linear(token_dim, 2)
        self.pre_1_2 = nn.Linear(token_dim, 2)
        self.pre_1_1 = nn.Linear(token_dim, 2)

        # dwt
        self.conv = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.avg = nn.AdaptiveAvgPool2d(output_size=(16, 512))
        self.sig = nn.Sigmoid()

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, saliency_h_1_16, h_1_16, h_1_8, h_1_4, h_1_2, temb, text_emb):

        B, _, _, = h_1_16.size()
        temb = temb.unsqueeze(dim=1) # B,1,384
        text_emb = text_emb.unsqueeze(dim=1) # B,1,384
        saliency_h_1_16 = self.mlp(self.norm(saliency_h_1_16)) # B,128,64
        # saliency_fea_1_16 B,256,64
        mask_1_16 = self.pre_1_16(saliency_h_1_16) # B,128,2
        mask_1_16 = mask_1_16.transpose(1, 2).reshape(B, 2, self.img_size[0] // 16, self.img_size[1] // 16) # B,2,2,64

        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        fea_1_8 = self.decoder1(h_1_16[:, 0:-2, :], h_1_8) # B,512,64
        saliency_fea_1_8, token_fea_1_8 = self.token_pre_1_8(fea_1_8, temb, text_emb)
        mask_1_8 = self.pre_1_8(saliency_fea_1_8)
        mask_1_8 = mask_1_8.transpose(1, 2).reshape(B, 2, self.img_size[0] // 8, self.img_size[1] // 8) # 1,2,8,128

        # 1/8 -> 1/4
        fea_1_4 = self.decoder2(token_fea_1_8[:, 0:-2, :], h_1_4) # B,2048,64
        saliency_fea_1_4, token_fea_1_4 = self.token_pre_1_4(fea_1_4, temb, text_emb)
        mask_1_4 = self.pre_1_4(saliency_fea_1_4)
        mask_1_4 = mask_1_4.transpose(1, 2).reshape(B, 2, self.img_size[0] // 4, self.img_size[1] // 4) # B,2,8,256

        # 1/4 -> 1/2
        fea_1_2 = self.decoder3(token_fea_1_4[:, 0:-2, :], h_1_2) # B,4096,64
        saliency_fea_1_2, token_fea_1_2 = self.token_pre_1_2(fea_1_2, temb, text_emb)
        mask_1_2 = self.pre_1_2(saliency_fea_1_2)
        mask_1_2 = mask_1_2.transpose(1, 2).reshape(B, 2, self.img_size[0] // 2, self.img_size[1] // 2) # B,2,32,512

        # 1/2 -> 1
        # saliency_fea_1_1 = self.decoder3(token_fea_1_4[:, 1:-1, :], temb)
        saliency_fea_1_1 = self.decoder4(saliency_fea_1_2)
        mask_1_1 = self.pre_1_1(saliency_fea_1_1)
        mask_1_1 = mask_1_1.transpose(1, 2).reshape(B, 2, self.img_size[0] // 1, self.img_size[1] // 1) # B,2,32,1024
        # out = mask_1_1[:, 1:2, :, :].cpu().detach().numpy()
        # out = out.squeeze()
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(out, cmap='jet')
        # plt.savefig('/project/r2dm-main/debugfig/mask_1_1.png', dpi=300, bbox_inches='tight', pad_inches=0)
        dwt_prepare = mask_1_1.clone()
        dwt = self.conv(dwt_prepare)
        dwt = self.sig(self.avg(dwt))

        x_LL, x_HL, x_LH, x_HH = dwt_module(dwt_prepare[:,0].unsqueeze(dim=1))
        idwt = torch.cat((x_LL*dwt[:,0].unsqueeze(dim=1), x_HL*dwt[:,0].unsqueeze(dim=1), x_LH*dwt[:,0].unsqueeze(dim=1), x_HH*dwt[:,0].unsqueeze(dim=1)), 1)
        mask_1_1[:,0] = idwt_module(idwt)


        return mask_1_1, [mask_1_2, mask_1_4, mask_1_8, mask_1_16]
        # return mask_1_1

