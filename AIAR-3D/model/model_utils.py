import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
from torch.autograd import grad
import pyvista as pv
import os

def highlight_sensor_overlap(
    plotter,
    sens_z,
    sens_y,
    sens_x,
    threshold=0.08,
    sphere_radius=0.025,
    base_color="red",
):
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().reshape(-1)
        return np.asarray(value).reshape(-1)

    z = _to_numpy(sens_z)
    y = _to_numpy(sens_y)
    x = _to_numpy(sens_x)

    points = np.column_stack((x * 3.0, y, z))
    used = np.zeros(len(points), dtype=bool)
    colors = ["gold", "purple", "deepskyblue", "lime", "navy"]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for start_idx in range(len(points)):
        if used[start_idx]:
            continue
        group = []
        queue = [start_idx]
        used[start_idx] = True
        while queue:
            idx = queue.pop(0)
            group.append(idx)
            distances = np.linalg.norm(points - points[idx], axis=1)
            neighbors = np.where((distances <= threshold) & (~used))[0]
            used[neighbors] = True
            queue.extend(neighbors.tolist())

        if len(group) <= 1:
            continue

        plotter.add_mesh(
            pv.Sphere(radius=sphere_radius * 0.9, center=points[group[0]]),
            color=base_color,
            show_edges=True,
            edge_color="white",
        )
        for color_idx, idx in enumerate(group[1:]):
            angle = color_idx * golden_angle
            z_offset = 1.0 - 2.0 * (color_idx + 1) / len(group)
            radial = np.sqrt(max(0.0, 1.0 - z_offset**2))
            direction = np.array(
                [radial * np.cos(angle), radial * np.sin(angle), z_offset]
            )
            display_point = points[idx] + direction * sphere_radius * 2.2
            plotter.add_mesh(
                pv.Sphere(radius=sphere_radius, center=display_point),
                color=colors[color_idx % len(colors)],
                show_edges=True,
                edge_color="white",
            )

def sample_sensor_value(data, z_sens, y_sens, x_sens):

    batch_size = data.shape[0]
    x = x_sens.to(device=data.device, dtype=data.dtype) * 2.0 - 1.0
    y = y_sens.to(device=data.device, dtype=data.dtype) * 2.0 - 1.0
    z = z_sens.to(device=data.device, dtype=data.dtype) * 2.0 - 1.0
    grid = torch.stack([x, y, z], dim=-1)
    grid = grid.view(1, -1, 1, 1, 3).expand(batch_size, -1, -1, -1, -1)

    sampled = F.grid_sample(
        data,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

def u_net(decoder, z, coords):
    u = decoder(z, coords)

    return u

def pde_net(u_pred, coords_input, scaler):

    u_pred = u_pred.permute(0,2,1).unsqueeze(2).unsqueeze(3)
    u_pred = scaler.inverse(u_pred)

    u_pred_real = u_pred.squeeze(2).squeeze(2).permute(0, 2, 1)

    Ux, Uy, Uz, p, nut = torch.chunk(u_pred_real, 5, dim=-1)

    nut = F.softplus(nut) + 1e-6

    def grad_yxz(f):
        g = grad(
            f,
            coords_input,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        f_z = g[..., 0:1]
        f_y = g[..., 1:2]
        f_x = g[..., 2:3]
        return f_z, f_y, f_x

    Ux_z, Ux_y, Ux_x = grad_yxz(Ux)
    Uy_z, Uy_y, Uy_x = grad_yxz(Uy)
    Uz_z, Uz_y, Uz_x = grad_yxz(Uz)
    p_z, p_y, p_x  = grad_yxz(p)

    g2_11 = Ux_x**2 + Ux_y * Uy_x + Ux_z * Uz_x
    g2_22 = Uy_x * Ux_y + Uy_y**2 + Uy_z * Uz_y
    g2_33 = Uz_x * Ux_z + Uz_y * Uy_z + Uz_z**2
    g2_12 = Ux_x * Ux_y + Ux_y * Uy_y + Ux_z * Uz_y
    g2_21 = Uy_x * Ux_x + Uy_y * Uy_x + Uy_z * Uz_x
    g2_13 = Ux_x * Ux_z + Ux_y * Uy_z + Ux_z * Uz_z
    g2_31 = Uz_x * Ux_x + Uz_y * Uy_x + Uz_z * Uz_x
    g2_23 = Uy_x * Ux_z + Uy_y * Uy_z + Uy_z * Uz_z
    g2_32 = Uz_x * Ux_y + Uz_y * Uy_y + Uz_z * Uz_y

    tr_g2 = g2_11 + g2_22 + g2_33

    Sd_11 = g2_11 - tr_g2 / 3.0
    Sd_22 = g2_22 - tr_g2 / 3.0
    Sd_33 = g2_33 - tr_g2 / 3.0
    Sd_12 = (g2_12 + g2_21) / 2.0
    Sd_13 = (g2_13 + g2_31) / 2.0
    Sd_23 = (g2_23 + g2_32) / 2.0

    SdSd = Sd_11**2 + Sd_22**2 + Sd_33**2 + 2.0 * (Sd_12**2 + Sd_13**2 + Sd_23**2)

    S11 = Ux_x
    S22 = Uy_y
    S33 = Uz_z
    S12 = (Ux_y + Uy_x) / 2.0
    S13 = (Ux_z + Uz_x) / 2.0
    S23 = (Uy_z + Uz_y) / 2.0

    SS = S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2)

    Cw = 0.325
    Delta = 0.01
    CwD2 = (Cw * Delta)**2
    nut_wale = CwD2 * SdSd**1.5 / (SS**2.5 + SdSd**1.25 + 1e-12)

    Re = 5000
    nu = 1.0 / Re

    mu_mom = nu + nut

    def grad_z(f):
        g = grad(
            f,
            coords_input,
            grad_outputs=torch.ones_like(f),
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        return g[..., 0:1]

    def grad_y(f):
        g = grad(
            f,
            coords_input,
            grad_outputs=torch.ones_like(f),
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        return g[..., 1:2]

    def grad_x(f):
        g = grad(
            f,
            coords_input,
            grad_outputs=torch.ones_like(f),
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        return g[..., 2:3]

    term_ux_z = mu_mom * Ux_z
    term_ux_y = mu_mom * Ux_y
    term_ux_x = mu_mom * Ux_x

    term_uy_z = mu_mom * Uy_z
    term_uy_y = mu_mom * Uy_y
    term_uy_x = mu_mom * Uy_x

    term_uz_z = mu_mom * Uz_z
    term_uz_y = mu_mom * Uz_y
    term_uz_x = mu_mom * Uz_x

    ux_lap = grad_z(term_ux_z) + grad_y(term_ux_y) + grad_x(term_ux_x)
    uy_lap = grad_z(term_uy_z) + grad_y(term_uy_y) + grad_x(term_uy_x)
    uz_lap = grad_z(term_uz_z) + grad_y(term_uz_y) + grad_x(term_uz_x)

    cont_res = Ux_x + Uy_y + Uz_z

    mom_x_res = Ux * Ux_x + Uy * Ux_y + Uz * Ux_z + p_x - ux_lap
    mom_y_res = Ux * Uy_x + Uy * Uy_y + Uz * Uy_z + p_y - uy_lap
    mom_z_res = Ux * Uz_x + Uy * Uz_y + Uz * Uz_z + p_z - uz_lap

    nut_res = nut - nut_wale

    cont_res  = cont_res * 100.0
    mom_x_res = mom_x_res
    mom_y_res = mom_y_res
    mom_z_res = mom_z_res
    nut_res   = nut_res

    del term_ux_z, term_ux_y, term_ux_x
    del term_uy_z, term_uy_y, term_uy_x
    del term_uz_z, term_uz_y, term_uz_x
    del Ux_z, Ux_y, Ux_x, Uy_z, Uy_y, Uy_x, Uz_z, Uz_y, Uz_x
    del p_z, p_y, p_x
    del g2_11, g2_22, g2_33, g2_12, g2_21, g2_13, g2_31, g2_23, g2_32
    del Sd_11, Sd_22, Sd_33, Sd_12, Sd_13, Sd_23
    del S11, S22, S33, S12, S13, S23
    del nut_wale, mu_mom
    del u_pred, u_pred_real

    residual = torch.cat([cont_res, mom_x_res, mom_y_res, mom_z_res, nut_res], dim=-1)

    loss_res = l2_loss(residual, torch.zeros_like(residual))

    return loss_res

def loss_fn(encoder, decoder, batch_data, scaler, use_pde=True, pde_r = 0, max_steps=200, current_step = 0):

    coords, coords_out, x = batch_data

    coords_input = coords.detach().requires_grad_(True)

    latent_z = encoder(x)

    u_pred = u_net(decoder, latent_z, coords_input)

    loss_data = MSE(u_pred, coords_out)

    if current_step >= 150 and use_pde:

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

def append_U_channel(field, channel_dim=None):

    if channel_dim is None:
        channel_dim = 1 if field.ndim >= 4 else 0

    if isinstance(field, torch.Tensor):
        ux = field.select(channel_dim, 0)
        uy = field.select(channel_dim, 1)
        uz = field.select(channel_dim, 2)
        U = torch.sqrt(ux.square() + uy.square() + uz.square()).unsqueeze(channel_dim)
        return torch.cat([field, U], dim=channel_dim)

    ux = np.take(field, 0, axis=channel_dim)
    uy = np.take(field, 1, axis=channel_dim)
    uz = np.take(field, 2, axis=channel_dim)
    U = np.sqrt(np.square(ux) + np.square(uy) + np.square(uz))
    return np.concatenate([field, np.expand_dims(U, axis=channel_dim)], axis=channel_dim)

def compute_U_statistics(mode, u_true, data, scaler):

    if mode == "uncertainty":
        stacked = data
        scale_shape = [1] * stacked.ndim
        scale_shape[2] = -1
        scale = scaler.std.to(stacked.device).reshape(scale_shape)
        shift = scaler.mean.to(stacked.device).reshape(scale_shape)
        stacked_phys = stacked * scale + shift

        field_mean = stacked_phys.mean(dim=0)
        field_var = stacked_phys.var(dim=0, unbiased=False)

        U_samples = torch.sqrt(
            stacked_phys[:, :, 0].square()
            + stacked_phys[:, :, 1].square()
            + stacked_phys[:, :, 2].square()
        )
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

def save_uncertainty(save_dir, sample_stacks, scaler, channel_names, csv_name="sample_summary.csv"):

    if not sample_stacks:
        raise ValueError("sample_stacks must contain at least one sampled batch.")

    os.makedirs(save_dir, exist_ok=True)
    stacked = sample_stacks[0]
    if stacked.ndim != 6:
        raise ValueError(
            "Expected sample stack [num_samples, B, C, z, y, x], "
            f"got {tuple(stacked.shape)}."
        )
    if stacked.shape[2] < 3:
        raise ValueError("3D uncertainty export requires Ux, Uy, and Uz channels.")

    num_samples = stacked.shape[0]
    depth, height, width = stacked.shape[-3:]
    center_z, center_y, center_x = depth // 2, height // 2, width // 2
    scale = scaler.std.to(stacked.device)[None, None, :, None, None, None]
    shift = scaler.mean.to(stacked.device)[None, None, :, None, None, None]
    sample_phys = stacked * scale + shift
    sample_point = sample_phys[:, 0, :, center_z, center_y, center_x]
    velocity_magnitude = torch.sqrt(
        sample_point[:, 0].square()
        + sample_point[:, 1].square()
        + sample_point[:, 2].square()
    ).unsqueeze(1)
    sample_point = torch.cat([sample_point, velocity_magnitude], dim=1)
    sample_point = sample_point.detach().cpu().numpy()

    if len(channel_names) != sample_point.shape[1]:
        raise ValueError(
            "channel_names must include every field channel plus U: "
            f"expected {sample_point.shape[1]}, got {len(channel_names)}."
        )

    rows = [[sample_idx] + list(sample_point[sample_idx]) for sample_idx in range(num_samples)]
    np.savetxt(
        os.path.join(save_dir, csv_name),
        np.asarray(rows, dtype=object),
        delimiter=",",
        header=",".join(["sample_idx"] + list(channel_names)),
        comments="",
        fmt=["%d"] + ["%.6f"] * len(channel_names),
    )

