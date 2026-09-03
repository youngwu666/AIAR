import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
import random
from models.EncoderDecoder import Encoder, Decoder
from models.flow import DiT
from utils.checkpoint_utils import load_checkpoint, load_dit_checkpoint
from data_utils import create_dataloader, BatchParser
from model.model_utils import compute_U_statistics, relative_l2_error, relative_l2_error_per_channel, save_uncertainty, highlight_sensor_overlap
from model.train_flow import sample_ode, plot_ode_trajectory
from utils.model_utils import rand_sensor_indices
from utils.computational_efficiency import EfficiencyTracker, TimedModule, save_efficiency

NOISE_LEVEL = 0.0
NUM_ODE_STEPS = 1000
SENSOR_NUMBER = 10
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_DIR = os.path.join(PROJECT_DIR, "inference")

def visualize_results(
        noise_level,
        row_indices,
        col_indices,
        u_true,
        u_pred,
        channel_names,
        save_path,
        vis_idx,
        u_var = None,
        clolor = None,
        var_clim = None
    ):

    true = u_true[vis_idx].cpu().numpy()
    pred = u_pred[vis_idx].cpu().numpy()
    err = np.abs(pred - true)

    if u_var is not None:
        var = u_var[vis_idx].cpu().numpy()

    sens_y = row_indices.cpu().numpy()
    sens_x = col_indices.cpu().numpy()

    cmap = 'jet'
    origin = 'lower'

    C = len(channel_names)
    ncols = 4 if u_var is not None else 3

    fig, axes = plt.subplots(C, ncols, figsize=(4.5 * ncols, C * 3))

    for i in range(C):

        vmin = true[i].min()
        vmax = true[i].max()

        ax0 = axes[i, 0]
        im0 = ax0.imshow(true[i], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
        ax0.set_title(f'{channel_names[i]} - Ground Truth', fontsize=10)
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axes[i, 1]
        im1 = ax1.imshow(pred[i], cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
        ax1.set_title(f'{channel_names[i]} - Reconstruction - Noisy sensor{noise_level * 100}%', fontsize=10)

        if clolor is not None:
            c = "blue"
        else:
            c = "red"
        ax1.scatter(sens_x, sens_y, s=60, c = c, marker='*',
                    edgecolors='white', linewidths=0.5, zorder=10)

        highlight_sensor_overlap(ax1, sens_x, sens_y, size=60)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[i, 2]
        im2 = ax2.imshow(err[i], cmap=cmap, origin=origin)
        ax2.set_title(f'{channel_names[i]} - Abs Error', fontsize=10)

        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        if u_var is not None:
            ax3 = axes[i, 3]
            vmin, vmax = var_clim if var_clim is not None else (var[i].min(), var[i].max())
            im3 = ax3.imshow(var[i], cmap='hot', origin=origin,
                             vmin=vmin, vmax=vmax)
            ax3.set_title(f'{channel_names[i]} - Uncertainty', fontsize=10)

            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

def inference_uncertainty(config, device, num_samples=3):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, H, W = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, device)

    noise_level = NOISE_LEVEL
    num_ode_steps = NUM_ODE_STEPS
    sensor_number = SENSOR_NUMBER
    inference_dir = INFERENCE_DIR

    job_name = f"{config.model_name}_sensor_number{sensor_number}_noise_{noise_level}_ode{num_ode_steps}_uncertainty-{num_samples}"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    sample_stacks = []
    all_errors, all_ch_errors = [], []
    vis_sample_idx = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing', leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, H, W = data.shape

            indices = rand_sensor_indices(H, W, sensor_number, device, config.seed)

            row_indices = indices // W
            col_indices = indices % W
            sensor_value = data[:, :, row_indices, col_indices].permute(0, 2, 1)
            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            noise = torch.randn_like(sensor_value, device=device)
            sensor_value = sensor_value + noise * noise_level

            coords, coords_outputs, batch = batch_parser.query_all(data)

            z_u = encoder(batch)

            sample_collection = []
            for s in range(num_samples):
                z0 = torch.randn(z_u.shape, device=device)
                z_pred, traj = sample_ode(
                    dit=dit,
                    z0=z0,
                    sensor_value=sensor_value,
                    sensor_pos=sensor_pos,
                    num_steps=num_ode_steps,
                    use_conditioning=config.training.random_sensor
                )
                u_pred = decoder(z_pred, coords)
                u_pred = rearrange(u_pred, "b (h w) c -> b c h w", h=H, w=W)
                sample_collection.append(u_pred)

            stacked = torch.stack(sample_collection)
            sample_stacks.append(stacked)
            u_true, u_mean, u_var = compute_U_statistics(
                "uncertainty",
                batch,
                stacked,
                scaler
            )

            errors = relative_l2_error(u_mean[:, :5], u_true[:, :5])
            all_errors.extend(errors.cpu().numpy())

            ch_errors = relative_l2_error_per_channel(u_mean, u_true)
            all_ch_errors.extend(ch_errors.cpu().numpy())

            if batch_idx == vis_sample_idx:

                channel_names = ['Ux', 'Uy', 'p', 'k', 'epsilon', 'U']

                visualize_results(
                    noise_level = noise_level,
                    row_indices = row_indices,
                    col_indices = col_indices,
                    u_true = u_true,
                    u_pred = u_mean,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/inference_result.png",
                    vis_idx = vis_sample_idx,
                    u_var = u_var
                )

                visualize_vector_field(
                    u_true = u_true,
                    u_pred = u_mean,
                    row_indices = row_indices,
                    col_indices = col_indices,
                    save_path = f"{save_vis_dir}/vector_field.png",
                    vis_idx = vis_sample_idx
                )

                plot_U_statistics(
                    u_true = u_true,
                    u_mean = u_mean,
                    u_var = u_var,
                    row_indices = row_indices,
                    col_indices = col_indices,
                    save_path=f"{save_vis_dir}/U_statistics.png",
                    vis_idx = vis_sample_idx
                )

    print(f"Inference l2 Relative Error: {np.mean(all_errors):.4f} ± {np.std(all_errors, ddof=0):.4f}")

    ch_mean = np.mean(all_ch_errors, axis=0)
    ch_std = np.std(all_ch_errors, axis=0, ddof=0)
    for name, mean, std in zip(['Ux', 'Uy', 'p', 'k', 'epsilon','U'], ch_mean, ch_std):
        print(f"{name:>8s}: {mean:.4f} ± {std:.4f}")

    np.savetxt(os.path.join(save_vis_dir, 'error_summary.csv'),
           np.array(
               list(zip(
                    ['overall_mean','Ux', 'Uy', 'p', 'k', 'epsilon','U'],
                    [np.mean(all_errors)] + list(ch_mean),
                    [np.std(all_errors, ddof=0)] + list(ch_std)

                   )),dtype=object
                ),
           delimiter=',', fmt=['%s','%.6f','%.6f'])

    save_uncertainty(save_vis_dir, sample_stacks, scaler,
                     ['Ux', 'Uy', 'p', 'k', 'epsilon', 'U'],
                     csv_name='sample_summary.csv')

def inference(config, device):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    efficiency = EfficiencyTracker(
        device,
        {"encoder": encoder, "decoder": decoder, "dit": dit},
    )
    decoder = TimedModule(decoder, efficiency)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, H, W = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, device)

    noise_level = NOISE_LEVEL
    num_ode_steps = NUM_ODE_STEPS
    sensor_number = SENSOR_NUMBER
    inference_dir = INFERENCE_DIR

    job_name = f"{config.model_name}_sensor_number{sensor_number}_noise_{noise_level}_ode{num_ode_steps}"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    all_errors, all_ch_errors = [],[]
    vis_sample_idx = 0

    efficiency.start()
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, H, W = data.shape

            indices = rand_sensor_indices(H, W, sensor_number, device, config.seed)

            row_indices = indices // W
            col_indices = indices % W

            sensor_value = data[:, :, row_indices, col_indices].permute(0, 2, 1)

            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            noise = torch.randn_like(sensor_value, device = device)
            sensor_value = sensor_value + noise * noise_level

            coords, coords_outputs, batch = batch_parser.query_all(data)

            z_u = encoder(batch)
            z0 = torch.randn(z_u.shape, device = device)

            z_pred, traj = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_value,
                sensor_pos = sensor_pos,
                num_steps = num_ode_steps,
                use_conditioning = config.training.random_sensor
            )

            u_pred = decoder(z_pred, coords)
            u_pred = rearrange(u_pred, "b (h w) c -> b c h w", h=H, w=W)

            u_true, u_pred, _ = compute_U_statistics(
                "inference",
                scaler.inverse(batch),
                scaler.inverse(u_pred),
                scaler
            )

            errors = relative_l2_error(u_pred[:, :5], u_true[:, :5])
            all_errors.extend(errors.cpu().numpy())

            ch_errors = relative_l2_error_per_channel(u_pred, u_true)
            all_ch_errors.extend(ch_errors.cpu().numpy())

            if batch_idx == vis_sample_idx:

                efficiency.pause()

                channel_names = ['Ux', 'Uy', 'p', 'k', 'epsilon','U']

                visualize_results(
                    noise_level = noise_level,
                    row_indices = row_indices,
                    col_indices = col_indices,
                    u_true = u_true,
                    u_pred = u_pred,
                    channel_names = channel_names,
                    save_path=f"{save_vis_dir}/inference_result.png",
                    vis_idx = vis_sample_idx
                )

                visualize_vector_field(
                    u_true = u_true,
                    u_pred = u_pred,
                    row_indices = row_indices,
                    col_indices = col_indices,
                    save_path = f"{save_vis_dir}/vector_field.png",
                    vis_idx = vis_sample_idx
                )

                efficiency.resume()

    save_efficiency(
        os.path.join(save_vis_dir, "computer_efficiency.csv"),
        efficiency.finish(),
    )

    print(f"Inference l2 Relative Error: {np.mean(all_errors):.4f}")

    ch_mean = np.mean(all_ch_errors, axis=0)
    for name, err in zip(['Ux', 'Uy', 'p', 'k', 'epsilon','U'], ch_mean):
        print(f"{name:>8s}: {err:.4f}")

    np.savetxt(os.path.join(save_vis_dir, 'error_summary.csv'),
           np.array(list(zip(['overall','Ux', 'Uy', 'p', 'k', 'epsilon','U'],
                             [np.mean(all_errors)] + list(np.mean(all_ch_errors, axis=0)))),
                    dtype=object),
           delimiter=',', fmt=['%s','%.6f'])

def visualize_vector_field(
        u_true,
        u_pred,
        row_indices,
        col_indices,
        save_path,
        vis_idx,
        skip=None
    ):

    true = u_true[vis_idx].cpu().numpy()
    pred = u_pred[vis_idx].cpu().numpy()

    U_true = true[5]
    U_pred = pred[5]

    Ny, Nx = U_true.shape
    if skip is None:
        skip = max(Nx // 20, 1)

    qx = np.arange(0, Nx, skip)
    qy = np.arange(0, Ny, skip)
    QX, QY = np.meshgrid(qx, qy)

    Ux_true = true[0][QY, QX]
    Uy_true = true[1][QY, QX]

    Ux_pred = pred[0][QY, QX]
    Uy_pred = pred[1][QY, QX]

    sens_x = col_indices.cpu().numpy()
    sens_y = row_indices.cpu().numpy()

    cmap = 'jet'
    origin = 'lower'

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax0 = axes[0]
    im0 = ax0.imshow(U_true, cmap=cmap, origin=origin)
    ax0.quiver(QX, QY, Ux_true, Uy_true, color='white',
               scale=None, alpha=0.8, width=0.003, zorder=5)

    ax0.set_title('U - Ground Truth', fontsize=10)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = axes[1]
    im1 = ax1.imshow(U_pred, cmap=cmap, origin=origin)
    ax1.quiver(QX, QY, Ux_pred, Uy_pred, color='white',
               scale=None, alpha=0.8, width=0.003, zorder=5)

    ax1.set_title('U - Reconstruction', fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

def plot_U_statistics(
        u_true,
        u_mean,
        u_var,
        row_indices,
        col_indices,
        save_path,
        vis_idx,
        sample_row=None,
        sample_col=None,
    ):

    true = u_true[vis_idx].cpu().numpy()
    mean = u_mean[vis_idx].cpu().numpy()
    var  = u_var[vis_idx].cpu().numpy()

    U_true = true[5]
    U_mean = mean[5]
    U_std = np.sqrt(np.clip(var[5], 0, None))

    Ny, Nx = U_true.shape

    if sample_col is not None:
        sample_col = int(np.clip(sample_col, 0, Nx - 1))
        U_true_line = U_true[:, sample_col]
        U_mean_line = U_mean[:, sample_col]
        U_std_line  = U_std[:, sample_col]
        coords = np.arange(Ny)
        line_label = f'x={sample_col}'
    else:
        if sample_row is None:
            sample_row = Ny // 2
        sample_row = int(np.clip(sample_row, 0, Ny - 1))
        U_true_line = U_true[sample_row, :]
        U_mean_line = U_mean[sample_row, :]
        U_std_line  = U_std[sample_row, :]
        coords = np.arange(Nx)
        line_label = f'y={sample_row}'

    sens_x = col_indices.cpu().numpy()
    sens_y = row_indices.cpu().numpy()

    fig = plt.figure(figsize=(14, 10))

    ax0 = fig.add_axes([0.25, 0.72, 0.5, 0.2])
    im0 = ax0.imshow(U_true, cmap='jet', origin='lower', aspect='auto')
    if sample_col is not None:
        ax0.axvline(x=sample_col, color='red', linestyle='--', linewidth=2)
    else:
        ax0.axhline(y=sample_row, color='red', linestyle='--', linewidth=2)
    ax0.scatter(sens_x, sens_y, s=80, c='red', marker='*',
                edgecolors='white', linewidths=0.5, zorder=10, label='Sensors')

    highlight_sensor_overlap(ax0, sens_x, sens_y, size=80)

    ax0.legend(fontsize=8, loc='upper right')
    ax0.set_title(f'U - Ground Truth  (line: {line_label})', fontsize=11)
    ax0.set_xlabel('x', fontsize=10)
    ax0.set_ylabel('y', fontsize=10)
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_axes([0.08, 0.3, 0.4, 0.3])
    ax1.plot(coords, U_true_line, 'b-', linewidth=1.5, label='Ground Truth')
    ax1.plot(coords, U_mean_line, 'r-', linewidth=1.5, label='Mean')
    ax1.fill_between(coords, U_mean_line - 3 * U_std_line, U_mean_line + 3 * U_std_line,
                     color='red', alpha=0.25, label='Mean ± 3σ')
    ax1.set_xlabel('Pixel index', fontsize=11)
    ax1.set_ylabel('U', fontsize=11)
    ax1.set_title(f'U along line {line_label}', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    from scipy.stats import gaussian_kde
    ax2 = fig.add_axes([0.55, 0.3, 0.4, 0.3])
    all_vals = np.concatenate([U_true_line, U_mean_line])
    bins = np.linspace(all_vals.min(), all_vals.max(), 40)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    counts_true, _ = np.histogram(U_true_line, bins=bins)
    counts_pred, _ = np.histogram(U_mean_line, bins=bins)
    ax2.bar(bin_centers, counts_true, width=bin_width,
            facecolor='none', edgecolor='blue', linewidth=1.2, label='Ground Truth')
    ax2.bar(bin_centers, counts_true, width=bin_width,
            facecolor='blue', edgecolor='none', alpha=0.3)
    ax2.bar(bin_centers, counts_pred, width=bin_width,
            facecolor='none', edgecolor='red', linewidth=1.2, label='Prediction')
    ax2.bar(bin_centers, counts_pred, width=bin_width,
            facecolor='red', edgecolor='none', alpha=0.3)

    kde_x = np.linspace(all_vals.min(), all_vals.max(), 200)
    kde_true = gaussian_kde(U_true_line)(kde_x) * len(U_true_line) * bin_width
    kde_pred = gaussian_kde(U_mean_line)(kde_x) * len(U_mean_line) * bin_width
    ax2.plot(kde_x, kde_true, 'b-', linewidth=2, alpha=0.8)
    ax2.plot(kde_x, kde_pred, 'r-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('U', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'U distribution along line {line_label}', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

    import csv
    csv_dir = os.path.dirname(save_path)

    csv_line_path = os.path.join(csv_dir, 'U_statistics_line.csv')
    with open(csv_line_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pixel_index', 'U_ground_truth', 'U_prediction_mean', 'U_std', 'U_mean_minus_3sigma', 'U_mean_plus_3sigma'])
        for i in range(len(coords)):
            writer.writerow([coords[i], U_true_line[i], U_mean_line[i], U_std_line[i],
                             U_mean_line[i] - 3 * U_std_line[i], U_mean_line[i] + 3 * U_std_line[i]])

    csv_hist_path = os.path.join(csv_dir, 'U_statistics_hist.csv')
    with open(csv_hist_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['U_bin_center', 'count_ground_truth', 'count_prediction'])
        for i in range(len(bin_centers)):
            writer.writerow([bin_centers[i], counts_true[i], counts_pred[i]])
