from typing import Iterable, Literal
import einops
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from .Transformer import token_Transformer
from .Transformer import Transformer
from .t2t_vit import T2t_vit_t_14
from . import encoding, ops
import torch.nn.functional as F
from .Decoder import Decoder
import time
# from .CLIP.clip import clip

def _join(*tensors) -> torch.Tensor:
    return torch.cat(tensors, dim=1)


def _n_tuple(x: Iterable | int, N: int) -> tuple[int]:
    if isinstance(x, Iterable):
        assert len(x) == N
        return x
    else:
        return (x,) * N

class Transdiff(nn.Module):

    def __init__(
        self,
        in_channels: int,
        resolution: tuple[int, int] | int,
        out_channels: int | None = None,  # == in_channels if None
        base_channels: int = 128,
        temb_channels: int = None,
        channel_multiplier: tuple[int] | int = (1, 2, 4, 8),
        num_residual_blocks: tuple[int] | int = (3, 3, 3, 3),
        gn_num_groups: int = 32 // 4,
        gn_eps: float = 1e-6,
        attn_num_heads: int = 8,
        coords_embedding: Literal[
            "spherical_harmonics", "polar_coordinates", "fourier_features", None
        ] = "spherical_harmonics",
        ring: bool = True,
    ):
        super().__init__()
        self.resolution = _n_tuple(resolution, 2)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        temb_channels = base_channels * 4 if temb_channels is None else temb_channels
        self.rgb_backbone = T2t_vit_t_14(pretrained=False)
        # transformer
        self.transformer = Transformer(embed_dim=384, numbers=4, num_heads=4, mlp_ratio=3.)
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=[32, 1024])
        # spatial coords embedding
        coords = encoding.generate_polar_coords(*self.resolution)
        self.in_conv = ops.Conv2d(32, 64, 3, 1, 1, ring=ring)

        self.register_buffer("coords", coords)
        self.coords_embedding = None
        if coords_embedding == "spherical_harmonics":
            self.coords_embedding = encoding.SphericalHarmonics(levels=5)
            in_channels += self.coords_embedding.extra_ch
        elif coords_embedding == "polar_coordinates":
            self.coords_embedding = nn.Identity()
            in_channels += coords.shape[1]
        elif coords_embedding == "fourier_features":
            self.coords_embedding = encoding.FourierFeatures(self.resolution)
            in_channels += self.coords_embedding.extra_ch
        # self.coords_transfer = ops.Conv2d(30, 1, 3, 1, 1, ring=ring)
        # timestep embedding
        self.time_embedding = nn.Sequential(
            ops.SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(512, 384),
            nn.SiLU(),
            nn.Linear(384, 384),
        )
        
        # parameters for up/down-sampling blocks
        updown_levels = 4
        channel_multiplier = _n_tuple(channel_multiplier, updown_levels)
        C = [base_channels] + [base_channels * m for m in channel_multiplier]
        N = _n_tuple(num_residual_blocks, updown_levels)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        h = images

        # text embedding
        text_emb = self.text_embedding(text.float()) # B, 384

        # timestep embedding
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].repeat_interleave(h.shape[0], dim=0)
        temb = self.time_embedding(timesteps.to(h)) # B,384
        
        # spatial embedding
        if self.coords_embedding is not None:
            cemb = self.coords_embedding(self.coords)
            cemb = cemb.repeat_interleave(h.shape[0], dim=0) # B, 32, 64, 1024
            # cemb = self.coords_transfer(cemb) # B, 1, 64, 1024
            # h = torch.cat([h, cemb], dim=1)

        # Transformer encoding
        h = torch.cat([h, cemb], dim=1)
        h = self.in_conv(h)
        # start_time = time.time()
        h_1_16, h_1_8, h_1_4, h_1_2 = self.rgb_backbone(h, temb) # h_1_16=B,128(2*64),384  # h_1_8=B,512(4*128),64  # h_1_4=B,2048(8*256),64
        h_1_16 = self.transformer(h_1_16) # B, 128, 384
        saliency_h_1_16, h_1_16 = self.token_trans(h_1_16, temb, text_emb)
        output, output_multi = self.decoder(saliency_h_1_16, h_1_16, h_1_8, h_1_4, h_1_2, temb, text_emb)
        # end_time = time.time()
        # print("程序运行时间：%.2f秒" % (end_time - start_time))
        return output, output_multi
    