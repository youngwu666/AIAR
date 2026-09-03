import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
from torch.autograd import grad
import csv
import os

def highlight_sensor_overlap(ax, sens_x, sens_y, threshold=8, size=80, base_color='blue'):
    sensor_xy = np.stack([sens_x, sens_y], axis=1)
    used = np.zeros(len(sensor_xy), dtype=bool)
    colors = ['purple', 'deepskyblue', 'lime', 'gold', 'navy']
    for j in range(len(sensor_xy)):
        if used[j]:
            continue
        group = []
        queue = [j]
        used[j] = True
        while queue:
            idx = queue.pop(0)
            group.append(idx)
            dist = np.sqrt(((sensor_xy - sensor_xy[idx]) ** 2).sum(axis=1))
            neighbors = np.where((dist <= threshold) & (~used))[0]
            used[neighbors] = True
            queue.extend(neighbors.tolist())
        group = np.array(group)
        if len(group) > 1:
            ax.scatter(sens_x[group[0]], sens_y[group[0]], s=size, c=base_color, marker='*',
                       edgecolors='white', linewidths=0.5, zorder=12)
            radius = 4
            for color_idx, idx in enumerate(group[1:]):
                angle = 2 * np.pi * color_idx / max(len(group) - 1, 1)
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                ax.scatter(sens_x[idx] + dx, sens_y[idx] + dy, s=size * 0.9,
                           c=colors[color_idx % len(colors)], marker='*',
                           edgecolors='none', zorder=13)

def sample_sensor_value(data, y_sens, x_sens):
    B, C, H, W = data.shape

    x = x_sens * 2 - 1
    y = y_sens * 2 - 1

    grid = torch.stack([x, y], dim = -1)
    grid = grid.view(1, -1, 1, 2).expand(B, -1, -1, -1)

    sampled = torch.nn.functional.grid_sample(
        data,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )

    return sampled.squeeze(-1).permute(0, 2, 1)

def u_net(decoder, z, coords):
    u = decoder(z, coords)

    return u

def pde_net(u_pred, coords_input, scaler):

    u_pred = u_pred.permute(0,2,1).unsqueeze(2)
    u_pred = scaler.inverse(u_pred)

    u_pred_real = u_pred.squeeze(2).permute(0, 2, 1)

    Ux, Uy, p, k, epsilon = torch.chunk(u_pred_real, 5, dim=-1)

    k = F.softplus(k) + 1e-6
    eps = F.softplus(epsilon) + 1e-6

    def grad_yx(f):
        g = grad(
            f,
            coords_input,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        f_y = g[..., 0:1]
        f_x = g[..., 1:2]
        return f_y, f_x

    Ux_y, Ux_x = grad_yx(Ux)
    Uy_y, Uy_x = grad_yx(Uy)
    p_y,  p_x  = grad_yx(p)
    k_y, k_x = grad_yx(k)
    eps_y, eps_x = grad_yx(eps)

    Re = 5000
    nu = 1.0 / Re
    C_mu = 0.0845
    sigma_k = 1.0
    sigma_eps = 1.3
    C1 = 1.42
    C2 = 1.68
    eta0 = 4.38
    beta = 0.012

    nu_t = C_mu * k**2 / eps

    To_S_ijS_ij = 2 * (Ux_x**2 + Uy_y**2) + (Ux_y + Uy_x)**2
    Pk = nu_t * To_S_ijS_ij

    S_norm = torch.sqrt(To_S_ijS_ij)
    eta = (k / eps) * S_norm
    R = (eta * (1.0 - eta / eta0)) / (1.0 + beta * eta**3)

    mu_mom = nu + nu_t
    mu_k   = nu + nu_t / sigma_k
    mu_eps = nu + nu_t / sigma_eps

    def grad_x(f):
        g = grad(
            f,
            coords_input,
            grad_outputs=torch.ones_like(f),
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        return g[..., 1:2]

    def grad_y(f):
        g = grad(
            f,
            coords_input,
            grad_outputs=torch.ones_like(f),
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        return g[..., 0:1]

    term_ux_x = mu_mom * Ux_x
    term_ux_y = mu_mom * Ux_y
    term_uy_x = mu_mom * Uy_x
    term_uy_y = mu_mom * Uy_y
    term_k_x = mu_k * k_x
    term_k_y = mu_k * k_y
    term_eps_x = mu_eps * eps_x
    term_eps_y = mu_eps * eps_y

    ux_lap = grad_x(term_ux_x) + grad_y(term_ux_y)

    uy_lap = grad_x(term_uy_x) + grad_y(term_uy_y)

    k_lap = grad_x(term_k_x) + grad_y(term_k_y)

    eps_lap = grad_x(term_eps_x) + grad_y(term_eps_y)

    cont_res = Ux_x + Uy_y

    mom_x_res = Ux * Ux_x + Uy * Ux_y + p_x - ux_lap

    mom_y_res = Ux * Uy_x + Uy * Uy_y + p_y - uy_lap

    k_res = Ux * k_x + Uy * k_y - Pk + eps - k_lap

    eps_res = Ux * eps_x + Uy * eps_y \
              - (C1 - R) * (eps / k) * Pk \
              + C2 * (eps**2 / k) \
              - eps_lap

    cont_res = cont_res * 1.0
    mom_x_res = mom_x_res * 1.0
    mom_y_res = mom_y_res * 1.0
    k_res = k_res * 0.01
    eps_res = eps_res * 0.01

    del term_ux_x, term_ux_y, term_uy_x, term_uy_y, term_k_x, term_k_y, term_eps_x, term_eps_y
    del Ux_x, Ux_y, Uy_x, Uy_y, p_x, p_y, k_x, k_y, eps_x, eps_y
    del nu_t, Pk, R, mu_mom, mu_k, mu_eps
    del u_pred, u_pred_real

    residual = torch.cat([cont_res, mom_x_res, mom_y_res, k_res, eps_res], dim=-1)
    loss_res = l2_loss(residual, torch.zeros_like(residual))

    return loss_res

def loss_fn(encoder, decoder, batch, scaler, use_pde=True, pde_r = 0, max_steps=200, current_step = 0):

    coords, coords_out, x = batch

    coords_input = coords.detach().requires_grad_(True)

    latent_z = encoder(x)

    u_pred = u_net(decoder, latent_z, coords_input)

    loss_data = MSE(u_pred, coords_out)

    if current_step>= 50 and use_pde:

        alpha = pde_r * (current_step / max_steps)

        loss_res = pde_net(u_pred, coords_input, scaler)
        loss_res = pde_r * loss_res
    else:

        loss_res = 0 * loss_data
        loss_res = pde_r * loss_res

    loss = loss_data + loss_res

    return loss, loss_data, loss_res

def MRE(pred, true, eps=1e-8):
    abs_error = torch.abs(pred - true)
    denom = torch.abs(true) + eps
    mre = (abs_error / denom).mean()
    return mre
def l2_loss(x, y):
    return ((x - y)**2).mean((-1, -2)).sqrt().mean()
def MAE(x, y):
    return torch.abs(x - y).mean((-1, -2)).mean()
def MSE(pred, true):
    return (pred - true).square().mean()
def RMSE(x, y):
    return ((x - y)**2).mean().sqrt()

def relative_l2_error(pred, true):

    pred_flat = pred.flatten(start_dim=1)
    true_flat = true.flatten(start_dim=1)

    numerator = torch.norm(pred_flat - true_flat, p=2, dim=1)
    denominator = torch.norm(true_flat, p=2, dim=1)

    return numerator / (denominator + 1e-8)

def append_U_channel(field, channel_dim=-3):

    if isinstance(field, torch.Tensor):
        ux = field.select(channel_dim, 0)
        uy = field.select(channel_dim, 1)
        U = torch.sqrt(ux.square() + uy.square()).unsqueeze(channel_dim)
        return torch.cat([field, U], dim=channel_dim)

    ux = np.take(field, 0, axis=channel_dim)
    uy = np.take(field, 1, axis=channel_dim)
    U = np.sqrt(np.square(ux) + np.square(uy))
    return np.concatenate([field, np.expand_dims(U, axis=channel_dim)], axis=channel_dim)

def compute_U_statistics(mode, u_true, data, scaler):

    if mode == "uncertainty":
        stacked = data
        scale = scaler.std.to(stacked.device)[None, None, :, None, None]
        shift = scaler.mean.to(stacked.device)[None, None, :, None, None]
        stacked_phys = stacked * scale + shift

        field_mean = stacked_phys.mean(dim=0)
        field_var = stacked_phys.var(dim=0, unbiased=False)

        U_samples = torch.sqrt(stacked_phys[:, :, 0].square() + stacked_phys[:, :, 1].square())
        U_mean = U_samples.mean(dim=0).unsqueeze(1)
        U_var = U_samples.var(dim=0, unbiased=False).unsqueeze(1)

        true_field = scaler.inverse(u_true)

        return (
            append_U_channel(true_field),
            torch.cat([field_mean, U_mean], dim=1),
            torch.cat([field_var, U_var], dim=1),
        )

    return append_U_channel(u_true), append_U_channel(data), None

def relative_l2_error_per_channel(pred, true):
    pred_flat = pred.flatten(start_dim=2)
    true_flat = true.flatten(start_dim=2)

    numerator   = torch.norm(pred_flat - true_flat, p=2, dim=2)
    denominator = torch.norm(true_flat, p=2, dim=2)
    return numerator / (denominator + 1e-8)

def save_uncertainty(save_dir, sample_stacks, scaler, channel_names, csv_name='sample_summary.csv'):

    os.makedirs(save_dir, exist_ok=True)
    stacked = sample_stacks[0]
    num_samples = stacked.shape[0]
    H, W = stacked.shape[-2:]
    center_y, center_x = H // 2, W // 2
    scale = scaler.std.to(stacked.device)[None, None, :, None, None]
    shift = scaler.mean.to(stacked.device)[None, None, :, None, None]
    sample_phys = stacked * scale + shift
    sample_point = sample_phys[:, 0, :, center_y, center_x]
    U = torch.sqrt(sample_point[:, 0].square() + sample_point[:, 1].square()).unsqueeze(1)
    sample_point = torch.cat([sample_point, U], dim=1).detach().cpu().numpy()

    sample_summary = []
    for s in range(num_samples):
        sample_summary.append([s] + list(sample_point[s]))

    np.savetxt(
        os.path.join(save_dir, csv_name),
        np.array(sample_summary, dtype=object),
        delimiter=',',
        header=','.join(['sample_idx'] + channel_names),
        comments='',
        fmt=['%d'] + ['%.6f'] * len(channel_names)
    )
