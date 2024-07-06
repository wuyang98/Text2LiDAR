import torch
from torch import nn
from .transformer_block import MutualSelfBlock
from .transformer_block import Block as TrBlock
from timm.models.layers import trunc_normal_

class TransformerEncoder_co(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(TransformerEncoder_co, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 TrBlock(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.rgb_norm_co = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb_fea_co):

        for block in self.blocks:
            rgb_fea_co = block(rgb_fea_co)

        rgb_fea_co = self.rgb_norm_co(rgb_fea_co)
        return rgb_fea_co

class TransformerEncoder(nn.Module):
    def __init__(self, numbers, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, numbers)]  # stochastic numbers decay rule
        self.blocks = nn.ModuleList([
                 TrBlock(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(numbers)])
        self.norm = norm_layer(embed_dim)
        
    def forward(self, h):

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        return h


class token_TransformerEncoder(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(token_TransformerEncoder, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
                 TrBlock(
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                        for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, fea):

        for block in self.blocks:
            fea = block(fea)

        fea = self.norm(fea)

        return fea


class Transformer(nn.Module):
    def __init__(self, embed_dim=384, numbers=14, num_heads=6, mlp_ratio=3.):
        super(Transformer, self).__init__()

        self.encoderlayer_co = TransformerEncoder(embed_dim=embed_dim, numbers=numbers, num_heads=num_heads,mlp_ratio=mlp_ratio)

    def forward(self, h_1_16):

        h_1_16 = self.encoderlayer_co(h_1_16)

        return h_1_16


class saliency_token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        B, N, C = fea.shape
        x = self.norm(fea)
        T_s, F_s = x[:, -1, :].unsqueeze(1), x[:, 0:-1, :]
        # T_s [B, 1, 384]  F_s [B, 14*14, 384]

        q = self.q(F_s).reshape(B, N-1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-1, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea + fea[:, 0:-1, :]
        return infer_fea


class token_Transformer(nn.Module):
    def __init__(self, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_Transformer, self).__init__()

        # self.saliency_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        

    def forward(self, h_1_16, temb):
        B, _, _ = h_1_16.shape
        temb = temb.unsqueeze(dim=1) # B,1,384
        # saliency_tokens = self.saliency_token.expand(B, -1, -1) # B,1,384
        # h_1_16 = torch.cat((saliency_tokens, h_1_16), dim=1) # B,257,384

        h_1_16 = torch.cat((h_1_16, temb), dim=1) # B,258,384

        h_1_16 = self.encoderlayer(h_1_16) # B,258,384
        # saliency_tokens = h_1_16[:, 0, :].unsqueeze(1) # B,1,384

        saliency_h_1_16 = self.saliency_token_pre(h_1_16) #B,256,384

        return saliency_h_1_16, h_1_16

