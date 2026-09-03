import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange,repeat
import random
from models.EncoderDecoder import Encoder, Decoder
from models.flow import DiT
from utils.checkpoint_utils import load_checkpoint, load_dit_checkpoint, save_sensor_checkpoint,load_sensor_checkpoint
from data_utils import create_dataloader, BatchParser
from model.model_utils import compute_U_statistics, relative_l2_error, relative_l2_error_per_channel, sample_sensor_value, save_uncertainty, highlight_sensor_overlap
from model.train_flow import sample_ode, plot_ode_trajectory
from model.inference_recons import visualize_results, visualize_vector_field
from utils.model_utils import rand_sensor_indices
from utils.computational_efficiency import EfficiencyTracker, TimedModule, save_efficiency

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_DIR = os.path.join(PROJECT_DIR, "inference")

def plot_sensor_optimization(
        true_field,
        rand_sens_x,
        rand_sens_y,
        opt_sens_x,
        opt_sens_y,
        channel_names,
        save_path,
    ):

    C = len(channel_names)

    fig, axes = plt.subplots(C, 1, figsize=(6, C * 4))
    if C == 1:
        axes = [axes]
    for i in range(C):
        ax = axes[i]
        im = ax.imshow(true_field[i], cmap='jet', origin='lower')
        ax.scatter(rand_sens_x, rand_sens_y, s=80, c='red', marker='*', edgecolors='white', linewidths=1, zorder=10, label='Random')
        ax.scatter(opt_sens_x, opt_sens_y, s=80, c='blue', marker='*', edgecolors='white', linewidths=1, zorder=10, label='Optimized')

        highlight_sensor_overlap(ax, opt_sens_x, opt_sens_y, size=80)

        ax.set_title(f'{channel_names[i]} - Sensor Positions', fontsize=11, fontweight='bold')
        ax.axis('off')
        if i == C-1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=2)

        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

    import csv

    field_csv_path = save_path.replace('.png', '_field.csv')
    Ny, Nx = true_field[0].shape
    x_coords = np.linspace(0, 3.0, Nx)
    y_coords = np.linspace(0, 1.0, Ny)

    with open(field_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Channel'] + channel_names)
        writer.writerow([])

        for c, ch_name in enumerate(channel_names):
            writer.writerow([f'=== {ch_name} ==='])
            writer.writerow(['y \\ x'] + [f'{x:.4f}' for x in x_coords])

            for iy, y_val in enumerate(y_coords):
                row = [f'{y_val:.4f}'] + true_field[c][iy, :].tolist()
                writer.writerow(row)

            writer.writerow([])

    sensor_csv_path = save_path.replace('.png', '_sensor_positions.csv')
    with open(sensor_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sensor_type', 'sensor_index', 'x_pixel', 'y_pixel', 'x_coord', 'y_coord'])

        for i, (x, y) in enumerate(zip(rand_sens_x, rand_sens_y)):
            x_coord = float(x) / max(Nx - 1, 1) * 3.0
            y_coord = float(y) / max(Ny - 1, 1) * 1.0
            writer.writerow(['random', i, x, y, x_coord, y_coord])

        for i, (x, y) in enumerate(zip(opt_sens_x, opt_sens_y)):
            x_coord = float(x) / max(Nx - 1, 1) * 3.0
            y_coord = float(y) / max(Ny - 1, 1) * 1.0
            writer.writerow(['optimized', i, x, y, x_coord, y_coord])

def inference(config, device):

    sensor_numer = config.sensor.sensor_numer
    noise_level = config.sensor.noise_level
    num_ode_steps = config.sensor.num_ode_steps

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, H, W = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, device)

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    dit.x_sens = torch.nn.Parameter(
        torch.zeros(sensor_numer, dtype=torch.float32).to(device),
        requires_grad=False
        )
    dit.y_sens = torch.nn.Parameter(
        torch.zeros(sensor_numer, dtype=torch.float32).to(device),
        requires_grad=False
        )

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_sensor_checkpoint(config, sensor_numer, dit, device)
    print(f"x_sens 是否全零: {(dit.x_sens == 0).all()}")

    encoder.eval()
    decoder.eval()
    dit.eval()

    efficiency = EfficiencyTracker(
        device,
        {"encoder": encoder, "decoder": decoder, "dit": dit},
    )
    decoder = TimedModule(decoder, efficiency)

    opt_errors, rand_errors = [], []
    opt_ch_errors, rand_ch_errors = [], []

    vis_batch = 0

    inference_dir = INFERENCE_DIR
    job_name = f"inference_sensor_{sensor_numer}_noise_{noise_level}_ode{num_ode_steps}"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    channel_names = ['Ux', 'Uy', 'p', 'k', 'epsilon', 'U']

    efficiency.start()
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)

            B, C, H, W = data.shape

            coords, coords_outputs, batch = batch_parser.query_all(data)

            z_u = encoder(batch)
            z0 = torch.randn(z_u.shape, device = device)

            opt_x_sens = dit.x_sens[:sensor_numer].to(device)
            opt_y_sens = dit.y_sens[:sensor_numer].to(device)

            sensor_pos = torch.stack([opt_y_sens, opt_x_sens], dim = -1).expand(B, -1, -1).to(device)

            sensor_value = sample_sensor_value(data, opt_y_sens, opt_x_sens)

            noise = torch.randn_like(sensor_value, device = device)
            sensor_value = sensor_value + noise * noise_level

            indices_rand = rand_sensor_indices(H, W, sensor_numer, device, config.seed)

            row_rand = indices_rand // W
            col_rand = indices_rand % W

            sensor_rand_value = data[:, :, row_rand, col_rand].permute(0, 2, 1)
            sensor_rand_pos = batch_parser.coords[indices_rand].expand(B, -1, -1).to(device)

            sensor_rand_value = sensor_rand_value + noise * noise_level

            z_pred, traj = sample_ode(
                    dit = dit,
                    z0 = z0,
                    sensor_value = sensor_value,
                    sensor_pos = sensor_pos,
                    num_steps = num_ode_steps,
                    use_conditioning = True
                )

            pred_rand, _ = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_rand_value,
                sensor_pos = sensor_rand_pos,
                num_steps = num_ode_steps,
                use_conditioning = True
                )

            u_pred = decoder(z_pred, coords)
            u_pred = rearrange(u_pred, "b (h w) c -> b c h w", h = H, w = W)
            u_true, u_pred, _ = compute_U_statistics(
                "inference",
                scaler.inverse(data),
                scaler.inverse(u_pred),
                scaler
            )

            errors = relative_l2_error(u_pred[:, :5], u_true[:, :5])
            opt_errors.extend(errors.cpu().numpy())

            opt_ch_errors.extend(relative_l2_error_per_channel(u_pred, u_true).cpu().numpy())

            rand_pred = decoder(pred_rand, coords)
            rand_pred = rearrange(rand_pred, "b (h w) c -> b c h w", h = H, w = W)
            _, rand_pred, _ = compute_U_statistics(
                "inference",
                scaler.inverse(data),
                scaler.inverse(rand_pred),
                scaler
            )

            rand_ch_errors.extend(relative_l2_error_per_channel(rand_pred, u_true).cpu().numpy())

            errors_rand = relative_l2_error(rand_pred[:, :5], u_true[:, :5])
            rand_errors.extend(errors_rand.cpu().numpy())

            if batch_idx == vis_batch:

                efficiency.pause()

                true_field = u_true.cpu().numpy()[vis_batch]

                plot_sensor_optimization(
                    true_field = true_field,
                    rand_sens_x = col_rand.cpu().numpy(),
                    rand_sens_y = row_rand.cpu().numpy(),
                    opt_sens_x = opt_x_sens.cpu().numpy()*(W-1),
                    opt_sens_y = opt_y_sens.cpu().numpy()*(H-1),
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/sensor_positions_{sensor_numer}.png"
                )

                visualize_results(
                    noise_level = noise_level,
                    row_indices = opt_y_sens*H,
                    col_indices = opt_x_sens*W,
                    u_true = u_true,
                    u_pred = u_pred,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/opt_sensor_{sensor_numer}.png",
                    vis_idx = vis_batch,
                    clolor = 1
                )

                visualize_results(
                    noise_level=noise_level,
                    row_indices = row_rand,
                    col_indices = col_rand,
                    u_true = u_true,
                    u_pred = rand_pred,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/rand_sensor_{sensor_numer}.png",
                    vis_idx = vis_batch
                )

                visualize_vector_field(
                    u_true = u_true,
                    u_pred = u_pred,
                    row_indices = opt_y_sens*(H-1),
                    col_indices = opt_x_sens*(W-1),
                    save_path = f"{save_vis_dir}/opt_vector_{sensor_numer}.png",
                    vis_idx=vis_batch
                )

                visualize_vector_field(
                    u_true = u_true,
                    u_pred = rand_pred,
                    row_indices = row_rand,
                    col_indices = col_rand,
                    save_path = f"{save_vis_dir}/rand_vector_{sensor_numer}.png",
                    vis_idx=vis_batch
                )
                efficiency.resume()

    save_efficiency(
        os.path.join(save_vis_dir, "computer_efficiency.csv"),
        efficiency.finish(),
    )

    promotion = (np.mean(rand_errors) - np.mean(opt_errors))/np.mean(rand_errors) * 100
    print(f"优化后的传感器数量 = {len(dit.x_sens):.4f}")
    print(f"优化误差 = {np.mean(opt_errors):.4f}")
    print(f"随机误差 = {np.mean(rand_errors):.4f}")
    print(f"相对提升 = {np.mean(promotion):.4f}%")

    opt_ch_mean= np.mean(opt_ch_errors,axis=0)
    rand_ch_mean = np.mean(rand_ch_errors, axis=0)
    for name, o, r in zip(['Ux', 'Uy', 'p', 'k', 'epsilon','U'],
                          opt_ch_mean,
                          rand_ch_mean):
        print(f"{name:>8s}: opt={o:.4f}rand={r:.4f}")

    np.savetxt(os.path.join(save_vis_dir, 'error_summary.csv'),
           np.array(list(zip(['overall','Ux', 'Uy', 'p', 'k', 'epsilon','U'],
                             [np.mean(opt_errors)] + list(opt_ch_mean),
                             [np.mean(rand_errors)] + list(rand_ch_mean)
                             )), dtype=object),
           delimiter=',', fmt=['%s', '%.6f', '%.6f'])

def inference_uncertainty(config, device, num_samples=10):

    sensor_numer = config.sensor.sensor_numer
    noise_level = config.sensor.noise_level
    num_ode_steps = config.sensor.num_ode_steps

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, H, W = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, device)

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    dit.x_sens = torch.nn.Parameter(
        torch.zeros(sensor_numer, dtype=torch.float32).to(device),
        requires_grad=False
        )
    dit.y_sens = torch.nn.Parameter(
        torch.zeros(sensor_numer, dtype=torch.float32).to(device),
        requires_grad=False
        )

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_sensor_checkpoint(config, sensor_numer, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    opt_sample_stacks, rand_sample_stacks = [], []

    opt_errors, rand_errors = [], []
    opt_ch_errors, rand_ch_errors = [], []
    vis_batch = 0

    inference_dir = INFERENCE_DIR
    job_name = f"inference_sensor_{sensor_numer}_noise_{noise_level}_ode{num_ode_steps}_uncertainty-{num_samples}"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    channel_names = ['Ux', 'Uy', 'p', 'k', 'epsilon', 'U']

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing', leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, H, W = data.shape

            coords, coords_outputs, batch = batch_parser.query_all(data)
            z_u = encoder(batch)

            opt_x_sens = dit.x_sens[:sensor_numer].to(device)
            opt_y_sens = dit.y_sens[:sensor_numer].to(device)

            sensor_pos = torch.stack([opt_y_sens, opt_x_sens], dim=-1).expand(B, -1, -1).to(device)

            sensor_value = sample_sensor_value(data, opt_y_sens, opt_x_sens)

            noise = torch.randn_like(sensor_value, device=device)
            sensor_value = sensor_value + noise * noise_level

            indices_rand = rand_sensor_indices(H, W, sensor_numer, device, config.seed)

            row_rand = indices_rand // W
            col_rand = indices_rand % W
            sensor_rand_value = data[:, :, row_rand, col_rand].permute(0, 2, 1)
            sensor_rand_pos = batch_parser.coords[indices_rand].expand(B, -1, -1).to(device)
            sensor_rand_value = sensor_rand_value + noise * noise_level

            opt_collection = []
            rand_collection = []

            for s in range(num_samples):

                z0 = torch.randn(z_u.shape, device=device)

                z_pred, _ = sample_ode(
                    dit=dit, z0=z0,
                    sensor_value=sensor_value, sensor_pos=sensor_pos,
                    num_steps=num_ode_steps, use_conditioning=True
                )
                pred_rand, _ = sample_ode(
                    dit=dit, z0=z0,
                    sensor_value=sensor_rand_value, sensor_pos=sensor_rand_pos,
                    num_steps=num_ode_steps, use_conditioning=True
                )

                u_pred = decoder(z_pred, coords)
                u_pred = rearrange(u_pred, "b (h w) c -> b c h w", h=H, w=W)
                opt_collection.append(u_pred)

                rand_pred = decoder(pred_rand, coords)
                rand_pred = rearrange(rand_pred, "b (h w) c -> b c h w", h=H, w=W)
                rand_collection.append(rand_pred)

            opt_stacked = torch.stack(opt_collection)
            opt_sample_stacks.append(opt_stacked)
            u_true, u_mean_opt, u_var_opt = compute_U_statistics(
                "uncertainty",
                batch,
                opt_stacked,
                scaler
            )
            errors = relative_l2_error(u_mean_opt[:, :5], u_true[:, :5])
            opt_errors.extend(errors.cpu().numpy())

            opt_ch_errors.extend(relative_l2_error_per_channel(u_mean_opt, u_true).cpu().numpy())

            rand_stacked = torch.stack(rand_collection)
            rand_sample_stacks.append(rand_stacked)
            _, u_mean_rand, u_var_rand = compute_U_statistics(
                "uncertainty",
                batch,
                rand_stacked,
                scaler
            )
            errors_rand = relative_l2_error(u_mean_rand[:, :5], u_true[:, :5])
            rand_errors.extend(errors_rand.cpu().numpy())

            rand_ch_errors.extend(relative_l2_error_per_channel(u_mean_rand, u_true).cpu().numpy())

            if batch_idx == vis_batch:

                var_clim = (float(u_var_rand[vis_batch].cpu().numpy().min()),
                        float(u_var_rand[vis_batch].cpu().numpy().max()))

                visualize_results(
                    noise_level = noise_level,
                    row_indices = opt_y_sens*(H-1),
                    col_indices = opt_x_sens*(W-1),
                    u_true = u_true,
                    u_pred = u_mean_opt,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/opt_sensor_{sensor_numer}.png",
                    vis_idx = vis_batch,
                    u_var = u_var_opt,
                    clolor = 1,
                    var_clim=var_clim
                )

                visualize_results(
                    noise_level = noise_level,
                    row_indices = row_rand,
                    col_indices = col_rand,
                    u_true = u_true,
                    u_pred = u_mean_rand,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/rand_sensor_{sensor_numer}.png",
                    vis_idx = vis_batch,
                    u_var = u_var_rand,
                    var_clim=var_clim
                )

                visualize_vector_field(
                    u_true = u_true,
                    u_pred = u_mean_opt,
                    row_indices = opt_y_sens*(H-1),
                    col_indices = opt_x_sens*(W-1),
                    save_path = f"{save_vis_dir}/opt_vector_{sensor_numer}.png",
                    vis_idx = vis_batch
                )

                visualize_vector_field(
                    u_true = u_true,
                    u_pred = u_mean_rand,
                    row_indices = row_rand,
                    col_indices = col_rand,
                    save_path = f"{save_vis_dir}/rand_vector_{sensor_numer}.png",
                    vis_idx = vis_batch
                )

                plot_U_statistics_compare(
                    u_true = u_true,
                    u_mean_opt = u_mean_opt,
                    u_mean_rand = u_mean_rand,
                    u_var_opt = u_var_opt,
                    u_var_rand = u_var_rand,
                    opt_row = opt_y_sens*(H-1),
                    opt_col = opt_x_sens*(W-1),
                    rand_row = row_rand,
                    rand_col = col_rand,
                    save_path = f"{save_vis_dir}/U_statistics_compare_{sensor_numer}.png",
                    vis_idx = vis_batch
                )

    promotion = (np.mean(rand_errors) - np.mean(opt_errors)) / np.mean(rand_errors) * 100
    print(f"优化后的传感器数量 = {len(dit.x_sens):.4f}")
    print(f"优化误差 = {np.mean(opt_errors):.4f}")
    print(f"随机误差 = {np.mean(rand_errors):.4f}")
    print(f"相对提升 = {promotion:.4f}%")

    opt_ch_mean= np.mean(opt_ch_errors,axis=0)
    opt_ch_std = np.std(opt_ch_errors, axis=0, ddof=0)

    rand_ch_mean = np.mean(rand_ch_errors, axis=0)
    for name, o, r in zip(['Ux', 'Uy', 'p', 'k', 'epsilon','U'], opt_ch_mean, rand_ch_mean):
        print(f"{name:>8s}: opt={o:.4f}rand={r:.4f}")

    np.savetxt(os.path.join(save_vis_dir, 'error_summary.csv'),
           np.array(
               list(zip(
                    ['overall','Ux', 'Uy', 'p', 'k', 'epsilon','U'],
                    [np.mean(opt_errors)] + list(opt_ch_mean),
                    [np.std(opt_errors, ddof=0)] + list(opt_ch_std),
                    [np.mean(rand_errors)] + list(rand_ch_mean)
                )), dtype=object
               ),
           delimiter=',', fmt=['%s', '%.6f','%.6f', '%.6f'])

    save_uncertainty(save_vis_dir, opt_sample_stacks, scaler,
                     ['Ux', 'Uy', 'p', 'k', 'epsilon', 'U'],
                     csv_name='sample_opt_summary.csv')

def plot_U_statistics_compare(
        u_true,
        u_mean_opt,
        u_mean_rand,
        u_var_opt,
        u_var_rand,
        opt_row,
        opt_col,
        rand_row,
        rand_col,
        save_path,
        vis_idx,
        sample_row=None,
        sample_col=None,
    ):

    true = u_true[vis_idx].cpu().numpy()
    mean_opt = u_mean_opt[vis_idx].cpu().numpy()
    mean_rand = u_mean_rand[vis_idx].cpu().numpy()
    var_opt = u_var_opt[vis_idx].cpu().numpy()
    var_rand = u_var_rand[vis_idx].cpu().numpy()

    U_true = true[5]
    U_opt = mean_opt[5]
    U_rand = mean_rand[5]
    U_std_opt = np.sqrt(np.clip(var_opt[5], 0, None))
    U_std_rand = np.sqrt(np.clip(var_rand[5], 0, None))

    Ny, Nx = U_true.shape

    if sample_col is not None:
        sample_col = int(np.clip(sample_col, 0, Nx - 1))
        U_true_line = U_true[:, sample_col]
        U_opt_line = U_opt[:, sample_col]
        U_rand_line = U_rand[:, sample_col]
        U_std_opt_line = U_std_opt[:, sample_col]
        U_std_rand_line = U_std_rand[:, sample_col]
        coords = np.arange(Ny)
        line_label = f'x={sample_col}'
    else:
        if sample_row is None:
            sample_row = Ny // 2
        sample_row = int(np.clip(sample_row, 0, Ny - 1))
        U_true_line = U_true[sample_row, :]
        U_opt_line = U_opt[sample_row, :]
        U_rand_line = U_rand[sample_row, :]
        U_std_opt_line = U_std_opt[sample_row, :]
        U_std_rand_line = U_std_rand[sample_row, :]
        coords = np.arange(Nx)
        line_label = f'y={sample_row}'

    opt_sens_x = opt_col.cpu().numpy()
    opt_sens_y = opt_row.cpu().numpy()
    rand_sens_x = rand_col.cpu().numpy()
    rand_sens_y = rand_row.cpu().numpy()

    fig = plt.figure(figsize=(14, 10))

    ax0 = fig.add_axes([0.25, 0.72, 0.5, 0.2])
    im0 = ax0.imshow(U_true, cmap='jet', origin='lower', aspect='auto')
    if sample_col is not None:
        ax0.axvline(x=sample_col, color='red', linestyle='--', linewidth=2)
    else:
        ax0.axhline(y=sample_row, color='red', linestyle='--', linewidth=2)
    ax0.scatter(rand_sens_x, rand_sens_y, s=80, c='red', marker='*',
                edgecolors='white', linewidths=0.5, zorder=10, label='Random')
    ax0.scatter(opt_sens_x, opt_sens_y, s=80, c='blue', marker='*',
                edgecolors='white', linewidths=0.5, zorder=10, label='Optimized')

    highlight_sensor_overlap(ax0, opt_sens_x, opt_sens_y, size=80)

    ax0.legend(fontsize=8, loc='upper right')
    ax0.set_title(f'U - Ground Truth  (line: {line_label})', fontsize=11)
    ax0.set_xlabel('x', fontsize=10)
    ax0.set_ylabel('y', fontsize=10)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_axes([0.08, 0.3, 0.4, 0.3])
    ax1.plot(coords, U_true_line, 'b-', linewidth=1.5, label='Ground Truth')
    ax1.plot(coords, U_opt_line, 'r-', linewidth=1.5, label='Optimized')
    ax1.fill_between(coords, U_opt_line - 3 * U_std_opt_line, U_opt_line + 3 * U_std_opt_line,
                     color='red', alpha=0.15, label='Opt ± 3σ')
    ax1.plot(coords, U_rand_line, 'g-', linewidth=1.5, label='Random')
    ax1.fill_between(coords, U_rand_line - 3 * U_std_rand_line, U_rand_line + 3 * U_std_rand_line,
                     color='green', alpha=0.15, label='Rand ± 3σ')
    ax1.set_xlabel('Pixel index', fontsize=11)
    ax1.set_ylabel('U', fontsize=11)
    ax1.set_title(f'U along line {line_label}', fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    from scipy.stats import gaussian_kde
    ax2 = fig.add_axes([0.55, 0.3, 0.4, 0.3])
    all_vals = np.concatenate([U_true_line, U_opt_line, U_rand_line])
    bins = np.linspace(all_vals.min(), all_vals.max(), 40)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    ax2.bar(bin_centers, np.histogram(U_true_line, bins=bins)[0], width=bin_width,
            facecolor='none', edgecolor='blue', linewidth=1.2, label='Ground Truth')
    ax2.bar(bin_centers, np.histogram(U_true_line, bins=bins)[0], width=bin_width,
            facecolor='blue', edgecolor='none', alpha=0.2)
    ax2.bar(bin_centers, np.histogram(U_opt_line, bins=bins)[0], width=bin_width,
            facecolor='none', edgecolor='red', linewidth=1.2, label='Optimized')
    ax2.bar(bin_centers, np.histogram(U_opt_line, bins=bins)[0], width=bin_width,
            facecolor='red', edgecolor='none', alpha=0.2)
    ax2.bar(bin_centers, np.histogram(U_rand_line, bins=bins)[0], width=bin_width,
            facecolor='none', edgecolor='green', linewidth=1.2, label='Random')
    ax2.bar(bin_centers, np.histogram(U_rand_line, bins=bins)[0], width=bin_width,
            facecolor='green', edgecolor='none', alpha=0.2)

    kde_x = np.linspace(all_vals.min(), all_vals.max(), 200)
    ax2.plot(kde_x, gaussian_kde(U_true_line)(kde_x) * len(U_true_line) * bin_width, 'b-', linewidth=2, alpha=0.8)
    ax2.plot(kde_x, gaussian_kde(U_opt_line)(kde_x) * len(U_opt_line) * bin_width, 'r-', linewidth=2, alpha=0.8)
    ax2.plot(kde_x, gaussian_kde(U_rand_line)(kde_x) * len(U_rand_line) * bin_width, 'g-', linewidth=2, alpha=0.8)

    ax2.set_xlabel('U', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'U distribution along line {line_label}', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

    import csv
    csv_dir = os.path.dirname(save_path)

    csv_line_path = os.path.join(csv_dir, 'U_statistics_compare_line.csv')
    with open(csv_line_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pixel_index', 'U_ground_truth', 'U_optimized', 'U_opt_std',
                         'U_random', 'U_rand_std'])
        for i in range(len(coords)):
            writer.writerow([coords[i], U_true_line[i], U_opt_line[i], U_std_opt_line[i],
                             U_rand_line[i], U_std_rand_line[i]])

    csv_hist_path = os.path.join(csv_dir, 'U_statistics_compare_hist.csv')
    counts_true, _ = np.histogram(U_true_line, bins=bins)
    counts_opt, _ = np.histogram(U_opt_line, bins=bins)
    counts_rand, _ = np.histogram(U_rand_line, bins=bins)
    with open(csv_hist_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['U_bin_center', 'count_ground_truth', 'count_optimized', 'count_random'])
        for i in range(len(bin_centers)):
            writer.writerow([bin_centers[i], counts_true[i], counts_opt[i], counts_rand[i]])
