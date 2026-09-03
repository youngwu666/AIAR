import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
        embed_dim, torch.arange(length, dtype=torch.float32)
    )
    return pos_embed.unsqueeze(0)

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0

        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = torch.cat([emb_h, emb_w], dim=1)
        return emb

    grid_h = torch.arange(grid_size[0], dtype=torch.float32)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.unsqueeze(0)

def get_3d_sincos_pos_embed(embed_dim, grid_size):
    def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):

        dim_per_axis = (embed_dim // 3) // 2 * 2

        rem_dim = embed_dim - dim_per_axis * 3

        emb_z = get_1d_sincos_pos_embed_from_grid(dim_per_axis, grid[0])
        emb_y = get_1d_sincos_pos_embed_from_grid(dim_per_axis, grid[1])
        emb_x = get_1d_sincos_pos_embed_from_grid(dim_per_axis + rem_dim, grid[2])

        emb = torch.cat([emb_z, emb_y, emb_x], dim=1)
        return emb

    grid_w = torch.arange(grid_size[0], dtype=torch.float32)
    grid_h = torch.arange(grid_size[1], dtype=torch.float32)
    grid_l = torch.arange(grid_size[2], dtype=torch.float32)

    grid = torch.meshgrid(grid_w, grid_h, grid_l, indexing="ij")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.unsqueeze(0)

class PatchEmbed3D(nn.Module):

    def __init__(self, patch_size=(16,16), emb_dim=768, channel = 5, use_norm=False):
        super().__init__()

        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm
        self.channel = channel

        self.proj = nn.Conv3d(in_channels = channel,
                              out_channels = emb_dim,
                              kernel_size = patch_size,
                              stride = patch_size
                              )
        if use_norm:
            self.norm = nn.LayerNorm(emb_dim, eps=1e-5)

    def forward(self, inputs):

        x = self.proj(inputs)

        x = rearrange(x, 'b d w h l-> b (w h l) d')

        if self.use_norm:
            x = self.norm(x)

        return x

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, inputs):

        x = F.gelu(self.fc1(inputs))
        x = self.fc2(x)
        return x

class SelfAttnBlock(nn.Module):
    def __init__(self, num_heads, emb_dim, mlp_ratio, layer_norm_eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.mlp = MlpBlock(emb_dim * mlp_ratio, emb_dim)

    def forward(self, inputs):

        x = self.norm1(inputs)
        x, _ = self.attn(x, x, x)
        x = x + inputs

        y = self.mlp(self.norm2(x))
        return x + y

class CrossAttnBlock(nn.Module):
    def __init__(self, num_heads, emb_dim, mlp_ratio, layer_norm_eps=1e-5):
        super().__init__()
        self.norm_q = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.norm_kv = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
        self.mlp = MlpBlock(emb_dim * mlp_ratio, emb_dim)

    def forward(self, q_inputs, kv_inputs):

        q, kv = self.norm_q(q_inputs), self.norm_kv(kv_inputs)
        x, _ = self.attn(q, kv, kv)
        x = x + q_inputs

        y = self.mlp(self.norm_out(x))
        return x + y

class PerceiverBlock(nn.Module):
    def __init__(self, emb_dim, depth, num_heads=8, num_latents=64, mlp_ratio=1, layer_norm_eps=1e-5):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, emb_dim))

        self.cross_attn = nn.ModuleList([
            CrossAttnBlock(num_heads, emb_dim, mlp_ratio, layer_norm_eps) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)

    def forward(self, x):

        latents = repeat(self.latents, 'l d -> b l d', b=x.shape[0])

        for layer in self.cross_attn:
            latents = layer(latents, x)
        return self.norm(latents)

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.W_patch, self.H_patch, self.L_patch = config.patch_size
        self.W_img, self.H_img, self.L_img = config.grid_size
        self.emb_dim = config.emb_dim
        self.channel = config.channel

        self.patch_embed = PatchEmbed3D(config.patch_size,
                                      self.emb_dim,
                                      self.channel
                                      )

        self.pos_emb = nn.Parameter(
            get_3d_sincos_pos_embed(self.emb_dim,
                                    (self.W_img//self.W_patch,
                                     self.H_img//self.H_patch,
                                     self.L_img//self.L_patch)
                                    ),requires_grad = False
        )

        self.perceiver = PerceiverBlock(self.emb_dim,
                                        2,
                                        config.num_heads,
                                        config.num_latents
                                        )

        self.transformer = nn.ModuleList([
            SelfAttnBlock(config.num_heads,
                          self.emb_dim,
                          config.mlp_ratio,
                          config.layer_norm_eps
                           )
            for _ in range(config.depth)
        ])

        self.norm = nn.LayerNorm(self.emb_dim, eps=config.layer_norm_eps)

    def forward(self, x):

        B, C, W, H, L = x.shape

        x = self.patch_embed(x)

        pos_emb = self.pos_emb

        x = x + pos_emb

        x = self.perceiver(x)

        for layer in self.transformer:
            x = layer(x)

        return self.norm(x)

class FourierEmbs(nn.Module):

    def __init__(self, embed_scale, embed_dim, in_dim = 3):

        super().__init__()

        self.kernel = nn.Parameter(torch.randn(in_dim, embed_dim//2) * embed_scale)

    def forward(self, x):

        return torch.cat([torch.cos(x @ self.kernel), torch.sin(x @ self.kernel)], dim=-1)

class Mlp(nn.Module):
    def __init__(self, num_layers, hidden_dim, out_dim, layer_norm_eps=1e-5):

        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.fourier_freq = config.fourier_freq
        self.dec_emb_dim = config.dec_emb_dim
        self.dec_depth = config.dec_depth
        self.dec_num_heads = config.dec_num_heads
        self.mlp_ratio = config.mlp_ratio
        self.num_mlp_layers = config.num_mlp_layers
        self.out_dim = config.out_dim
        self.layer_norm_eps = config.layer_norm_eps

        self.fourier = FourierEmbs(self.fourier_freq, self.dec_emb_dim)

        self.proj = nn.Linear(self.dec_emb_dim, self.dec_emb_dim)

        self.cross_attn = nn.ModuleList([
            CrossAttnBlock(self.dec_num_heads, self.dec_emb_dim, self.mlp_ratio, self.layer_norm_eps)
            for _ in range(self.dec_depth)
        ])

        self.norm = nn.LayerNorm(self.dec_emb_dim, eps=self.layer_norm_eps)

        self.mlp = Mlp(self.num_mlp_layers, self.dec_emb_dim, self.out_dim, self.layer_norm_eps)

    def forward(self, x, coords):

        coords = self.fourier(coords)

        x = self.proj(x)

        for layer in self.cross_attn:
            coords = layer(coords, x)

        return self.mlp(self.norm(coords))