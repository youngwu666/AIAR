import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from einops import rearrange
import math

class TimestepEmbedder(nn.Module):

    def __init__(self, emb_dim, frequency_embedding_size=256):
        super().__init__()

        self.emb_dim = emb_dim
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def timestep_embedding(self, t, dim, max_period=10000):

        half = dim // 2

        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
            ).to(t.device)

        args = t[:, None].float() * freqs[None]

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t):

        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

        t_emb = self.mlp(t_freq)

        return t_emb

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

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=(16, 16), emb_dim=768, in_channels=3, use_norm=False):
        super().__init__()

        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.use_norm = use_norm

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        if use_norm:
            self.norm = nn.LayerNorm(emb_dim, eps=1e-5)

    def forward(self, inputs):

        x = rearrange(inputs, 'b h w c -> b c h w')

        x = self.proj(x)

        x = rearrange(x, 'b d h w -> b (h w) d')

        if self.use_norm:
            x = self.norm(x)
        return x

class MlpBlock(nn.Module):

    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.fc1 = nn.Linear(out_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, out_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs):

        x = self.fc1(inputs)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_heads,
        is_concat = True,
        dropout = 0.1,
        leaky_relu_negative_slope = 0.2,
        sigma = 0.3,
    ):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.sigma = sigma

        if is_concat:

            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:

            self.n_hidden = out_features

        self.proj = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        self.proj_attn = nn.Linear(in_features, self.n_hidden * n_heads // 2, bias=False)

        self.attn = nn.Linear(self.n_hidden, 1, bias=False)

        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def compute_dynamic_adj(self, positions):

        B, M, _ = positions.shape

        dist_sq = torch.cdist(positions, positions, p=2).pow(2)

        adj = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        eye = torch.eye(M, device=positions.device).unsqueeze(0)
        adj = adj * (1 - eye)

        return adj.unsqueeze(-1)

    def forward(self, h, positions):
        batch_size = h.shape[0]
        n_nodes = h.shape[1]

        adj_mat = self.compute_dynamic_adj(positions)

        g = self.proj(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        ga = self.proj_attn(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden // 2)

        g_repeat = ga.repeat(1, n_nodes, 1, 1)

        g_repeat_interleave = ga.repeat_interleave(n_nodes, dim=1)

        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)

        g_concat = g_concat.view(
            batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden
        )

        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        e = e * adj_mat

        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum("bijh,bjhf->bihf", a, g)

        if self.is_concat:

            return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:

            return attn_res.mean(dim=2)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()

        self.heads = heads
        self.inner_dim = dim

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(self, x, sensor_feat):

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)

        k, v = self.to_kv(sensor_feat).chunk(2, dim=-1)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def modulate(x, shift, scale):

    return x * (1 + scale[:, None, :]) + shift[:, None, :]

class DiTBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_ratio):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 12 * emb_dim)
        )

        nn.init.zeros_(self.cond_proj[1].weight)
        nn.init.zeros_(self.cond_proj[1].bias)

        self.cross_attn = CrossAttention(emb_dim, heads=num_heads)
        self.norm_cross = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-5)

        self.norm_cross_ffn = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-5)
        self.cross_ffn = MlpBlock(int(emb_dim * mlp_ratio), emb_dim)

        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, bias=True)
        self.norm_self = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-5)

        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)

        self.mlp = MlpBlock(int(emb_dim * mlp_ratio), emb_dim)
        self.norm_mlp = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-5)

    def forward(self, x, t, sensor_feat = None):

        (
            shift_cross, scale_cross, gate_cross,
            shift_cross_ffn, scale_cross_ffn, gate_cross_ffn,
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp
        ) = self.cond_proj(t).chunk(12, dim = -1)

        if sensor_feat is not None:

            x_cross = modulate(self.norm_cross(x), shift_cross, scale_cross)

            x_cross_out = self.cross_attn(x_cross, sensor_feat)
            x = x + gate_cross.unsqueeze(1) * x_cross_out

            x_cross_ffn = modulate(self.norm_cross_ffn(x), shift_cross_ffn, scale_cross_ffn)
            x = x + gate_cross_ffn.unsqueeze(1) * self.cross_ffn(x_cross_ffn)

        x_attn = modulate(self.norm_self(x), shift_msa, scale_msa)

        x_attn, _ = self.self_attn(x_attn, x_attn, x_attn)

        x = x + gate_msa.unsqueeze(1) * x_attn

        x_mlp = modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)

        return x

class FinalLayer(nn.Module):

    def __init__(self, out_dim, emb_dim):
        super().__init__()

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim)
        )

        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-5)
        self.proj = nn.Linear(emb_dim, out_dim)

        nn.init.zeros_(self.cond_proj[1].weight)
        nn.init.zeros_(self.cond_proj[1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, c):

        shift, scale = self.cond_proj(c).chunk(2, dim=-1)

        x = modulate(self.norm(x), shift, scale)

        x = self.proj(x)
        return x

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_dim = config.model.in_dim
        self.emb_dim = config.model.emb_dim

        self.phy_z, self.phy_y, self.phy_x = config.model.phy_size
        self.W_img, self.H_img, self.L_img = config.model.grid_size

        self.depth = config.model.depth
        self.num_heads = config.model.num_heads
        self.mlp_ratio = config.model.mlp_ratio
        self.out_dim = config.model.out_dim
        self.use_conditioning = config.training.random_sensor

        self.x_proj = nn.Linear(self.in_dim, self.emb_dim)

        self.cond_proj = nn.Linear(self.emb_dim, self.emb_dim) if self.use_conditioning else None

        self.t_embed = TimestepEmbedder(self.emb_dim)

        self.sensor_enc = nn.Sequential(
            nn.Linear(config.model.channels + 3, self.emb_dim),
            nn.LayerNorm(self.emb_dim)
            )

        self.blocks = nn.ModuleList([
            DiTBlock(self.emb_dim, self.num_heads, self.mlp_ratio)
            for _ in range(self.depth)
        ])

        self.final_layer = FinalLayer(self.out_dim, self.emb_dim)

    def forward(self,
                z_t,
                t,
                sensor_value = None,
                sensor_pos = None
                ):

        B, N, _ = z_t.shape

        z_t = self.x_proj(z_t)

        pos_emb = get_1d_sincos_pos_embed(self.emb_dim, N).to(z_t.device)
        z_t = z_t + pos_emb

        t_emb = self.t_embed(t)

        if sensor_value is not None and self.cond_proj is not None:

            sensor_value = torch.cat([sensor_value, sensor_pos], dim=-1)
            sensor_feat = self.sensor_enc(sensor_value)

            for block in self.blocks:
                z_t = block(z_t, t_emb, sensor_feat)
        else:

            for block in self.blocks:
                z_t = block(z_t, t_emb)

        z_t = self.final_layer(z_t, t_emb)

        return z_t