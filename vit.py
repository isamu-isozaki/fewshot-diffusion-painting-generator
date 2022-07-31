"""
Author: Isamu Isozaki (isamu.website@gmail.com)
Description: description
Created:  2022-07-10T18:43:41.662Z
Modified: !date!
Modified By: modifier
"""

import torch
from vit_pytorch.vit_for_small_dataset import *
import sys
sys.path.append("glid-3-xl")
from guided_diffusion.unet import timestep_embedding

class sSPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class sViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = sSPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.dim = dim

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, imgs, timestep):
        t_emb = timestep_embedding(timestep, self.dim, repeat_only=False)
        timestep_embed = self.time_embed(t_emb)
        set_b, b, c, h, w = imgs.shape 
        imgs = torch.reshape(imgs, (set_b*b, c, h, w))
        x = self.to_patch_embedding(imgs)
        x_b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x_b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += timestep_embed[:, None, :]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = torch.reshape(x, (set_b, b, -1))
        return torch.reshape(torch.mean(self.mlp_head(x), dim=1), (set_b, 1, -1))
