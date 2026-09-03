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
import pyvista as pv
from utils.model_utils import rand_sensor_indices
from utils.computational_efficiency import EfficiencyTracker, TimedModule, save_efficiency

NOISE_LEVEL = 0.2
NUM_ODE_STEPS = 50
SENSOR_NUMBER = 80
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_DIR = os.path.join(PROJECT_DIR, "inference")
NUM_SAMPLES = 10

def make_error_dataset(u_true, u_pred, variable):

    if variable == 'Velocity_Magnitude':
        U_true = torch.sqrt(u_true[:,0]**2 + u_true[:,1]**2 + u_true[:,2]**2)
        U_pred = torch.sqrt(u_pred[:,0]**2 + u_pred[:,1]**2 + u_pred[:,2]**2)
        err = torch.zeros_like(u_true)
        err[:,0] = torch.abs(U_pred - U_true)
        return err
    return torch.abs(u_pred - u_true)

def _scalar_range(dataset_np, t, var):

    Ux, Uy, Uz = dataset_np[t,0], dataset_np[t,1], dataset_np[t,2]
    p, nut = dataset_np[t,3], dataset_np[t,4]
    vel = np.sqrt(Ux**2 + Uy**2 + Uz**2)
    d = {'Velocity_Magnitude': vel, 'Ux': Ux, 'Uy': Uy, 'Uz': Uz, 'p': p, 'nut': nut}
    s = d[var]
    return float(np.nanmin(s)), float(np.nanmax(s))

def plot_3d_flow_simple(
        dataset,
        time_idx,
        variable='Velocity_Magnitude',
        plot_type='isosurface',
        iso_vals=None,
        cmap='jet',
        clim = None,
        title_label='',
        rand_z=None, rand_y=None, rand_x=None,
        opt_z=None, opt_y=None, opt_x=None,
        save_path = None,
        show_interactive = True
    ):

    Ux = dataset[time_idx, 0, :, :, :]
    Uy = dataset[time_idx, 1, :, :, :]
    Uz = dataset[time_idx, 2, :, :, :]
    p  = dataset[time_idx, 3, :, :, :]
    nut = dataset[time_idx, 4, :, :, :]

    Nz, Ny, Nx = Ux.shape

    x = np.linspace(0, 3.0, Nx)
    y = np.linspace(0, 1.0, Ny)
    z = np.linspace(0, 1.0, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid = pv.StructuredGrid(X, Y, Z)

    vel_mag = np.sqrt(Ux**2 + Uy**2 + Uz**2)
    scalar_dict = {
        'Velocity_Magnitude': vel_mag,
        'Ux': Ux,
        'Uy': Uy,
        'Uz': Uz,
        'p': p,
        'nut': nut
    }
    scalar = scalar_dict[variable]

    sbar_args = {'title': ''}

    scalar_ordered = scalar.transpose(2, 1, 0)
    grid['scalar'] = scalar_ordered.flatten(order='F')

    plotter = pv.Plotter(off_screen=not show_interactive, window_size=(1200, 1000))

    if plot_type == 'isosurface':
        if iso_vals is None:

            if clim is not None:
                vmin, vmax = clim
            else:
                vmin, vmax = float(np.nanmin(scalar)), float(np.nanmax(scalar))

            iso_vals = np.linspace(vmin + 0.2*(vmax-vmin), vmax*0.8, 4)

        for val in iso_vals:
            contour = grid.contour(isosurfaces=[val], scalars='scalar')
            if contour.n_points == 0:
                continue
            plotter.add_mesh(contour, cmap=cmap,
                             clim=clim, opacity=1.0,
                             scalars='scalar', show_scalar_bar=True,
                             scalar_bar_args=sbar_args)

    elif plot_type == 'vortex':

        Ux_ord = Ux.transpose(2, 1, 0).flatten(order='F')
        Uy_ord = Uy.transpose(2, 1, 0).flatten(order='F')
        Uz_ord = Uz.transpose(2, 1, 0).flatten(order='F')
        grid['vectors'] = np.column_stack([Ux_ord, Uy_ord, Uz_ord])

        grid = grid.compute_derivative(scalars='vectors', gradient=True, qcriterion=True)

        q_key = None
        for k in grid.array_names:
            if 'q' in k.lower() and 'criterion' in k.lower():
                q_key = k
                break
        if q_key is None:
            print(f"Available arrays: {grid.array_names}")
            plotter.add_text('Q criterion not found in dataset', position='upper_right', font_size=10)
        else:
            Q = grid[q_key]
            Q_pos = Q[Q > 0]
            if len(Q_pos) > 0:

                if iso_vals is not None:
                    q_threshold = iso_vals
                else:
                    q_threshold = np.percentile(Q_pos, 92)
                    computed_q_vals = q_threshold
                contour = grid.contour(isosurfaces=[q_threshold], scalars=q_key)

                plotter.add_mesh(contour, cmap='plasma', opacity=1.0, scalars='scalar', clim=clim,
                                 show_scalar_bar=True, scalar_bar_args=sbar_args)
            else:
                plotter.add_text('No vortex (Q<=0 everywhere)', position='upper_right', font_size=10)

    elif plot_type == 'slice':
        plotter.add_mesh(grid.slice_orthogonal(), scalars='scalar', cmap=cmap, clim=clim,
                         show_scalar_bar=True, lighting=False, scalar_bar_args=sbar_args)

    elif plot_type == 'glyph':

        Ux_ord = Ux.transpose(2, 1, 0).flatten(order='F')
        Uy_ord = Uy.transpose(2, 1, 0).flatten(order='F')
        Uz_ord = Uz.transpose(2, 1, 0).flatten(order='F')
        grid['vectors'] = np.column_stack([Ux_ord, Uy_ord, Uz_ord])
        arrows = grid.glyph(orient='vectors', scale='scalar', factor=0.15)
        plotter.add_mesh(arrows, scalars='scalar', cmap=cmap,
                         clim=clim, show_scalar_bar=True,
                         scalar_bar_args=sbar_args)

    elif plot_type == 'streamline':

        Ux_ord = Ux.transpose(2, 1, 0).flatten(order='F')
        Uy_ord = Uy.transpose(2, 1, 0).flatten(order='F')
        Uz_ord = Uz.transpose(2, 1, 0).flatten(order='F')
        grid['vectors'] = np.column_stack([Ux_ord, Uy_ord, Uz_ord])

        streamlines = grid.streamlines(
            vectors='vectors',
            n_points=30,
            source_center=(1.5, 0.5, 0.5),
            source_radius=0.6,
            integration_direction='both',
            max_length=20.0
        )
        if streamlines.n_points > 0:

            tubes = streamlines.tube(radius=0.005)
            plotter.add_mesh(tubes, scalars='scalar',
                             cmap=cmap, clim=clim, show_scalar_bar=True,
                             scalar_bar_args=sbar_args)
        else:
            plotter.add_text('No streamlines generated', position='upper_right', font_size=10)

    elif plot_type == 'volume':

        vol = pv.ImageData(
            dimensions=(Nx, Ny, Nz),
            spacing=(3.0 / (Nx - 1), 1.0 / (Ny - 1), 1.0 / (Nz - 1)),
            origin=(0.0, 0.0, 0.0)
        )
        vol['scalar'] = scalar_ordered.flatten(order='F')

        plotter.add_volume(vol, scalars='scalar', cmap=cmap,
                           clim=clim, opacity='linear', show_scalar_bar=True,
                           scalar_bar_args=sbar_args)

    elif plot_type == 'volumeplus':

        vol = pv.ImageData(
            dimensions=(Nx, Ny, Nz),
            spacing=(3.0 / (Nx - 1), 1.0 / (Ny - 1), 1.0 / (Nz - 1)),
            origin=(0.0, 0.0, 0.0)
        )
        vol['scalar'] = scalar_ordered.flatten(order='F')

        plotter.add_volume(vol, scalars='scalar', cmap=cmap,
                           clim=clim, opacity='sigmoid', show_scalar_bar=True,
                           scalar_bar_args=sbar_args)

    elif plot_type == 'surface':

        plotter.add_mesh(grid, scalars='scalar', cmap=cmap, clim=clim,
                         show_scalar_bar=True, lighting=False, scalar_bar_args=sbar_args)

    elif plot_type == 'clip':

        clipped = grid.clip_box(
            bounds=(0.0, 3.0, 0.5, 1.0, 0.5, 1.0),
            invert=True
        )
        plotter.add_mesh(clipped, scalars='scalar', cmap=cmap, clim=clim,
                         show_scalar_bar=True, lighting=False,
                         scalar_bar_args=sbar_args)

    elif plot_type == 'sensor_only':

        pass
    plotter.add_mesh(grid.outline(), color='k', line_width=3, scalar_bar_args=sbar_args)

    has_sensors = (rand_z is not None and rand_y is not None and rand_x is not None) or \
                  (opt_z is not None and opt_y is not None and opt_x is not None)
    if has_sensors:

        if rand_z is not None and rand_y is not None and rand_x is not None:

            rand_x_phy = rand_x / (Nx - 1) * 3.0
            rand_y_phy = rand_y / (Ny - 1) * 1.0
            rand_z_phy = rand_z / (Nz - 1) * 1.0
            rand_points = pv.PolyData(np.column_stack((rand_x_phy.cpu().numpy(), rand_y_phy.cpu().numpy(), rand_z_phy.cpu().numpy())))

            plotter.add_mesh(
                rand_points,
                color = 'blue',
                point_size = 15,
                render_points_as_spheres = True,
                edge_color = 'white',
                line_width = 1
            )

            highlight_sensor_overlap(
                plotter,
                rand_z / max(Nz - 1, 1),
                rand_y / max(Ny - 1, 1),
                rand_x / max(Nx - 1, 1),
                base_color='blue',
            )

        if opt_z is not None and opt_y is not None and opt_x is not None:

            opt_x_phy = opt_x.cpu().numpy() * 3.0
            opt_y_phy = opt_y.cpu().numpy() * 1.0
            opt_z_phy = opt_z.cpu().numpy() * 1.0
            opt_points = pv.PolyData(np.column_stack((opt_x_phy, opt_y_phy, opt_z_phy)))

            plotter.add_mesh(
                opt_points,
                color = 'red',
                point_size = 15,
                render_points_as_spheres = True,
                edge_color = 'white',
                line_width = 1,
            )
            highlight_sensor_overlap(
                plotter,
                opt_z,
                opt_y,
                opt_x,
                base_color='red',
            )

    plotter.add_text(f'{title_label} - {variable} ({plot_type})'
                     if title_label else f'{variable} - {plot_type}', font_size=14, position='upper_left')

    plotter.camera_position = [
        (5, 3, 5.0),
        (1.5, 0.5, 0.5),
        (0, 1, 0)
    ]

    plotter.screenshot(save_path, transparent_background=True)
    plotter.show()
    plotter.close()

def visualize_results(
        noise_level,
        u_true,
        u_pred,
        channel_names,
        save_path,
        vis_idx,
        u_var = None
    ):

    true = u_true[vis_idx].cpu().numpy()
    pred = u_pred[vis_idx].cpu().numpy()
    err = np.abs(pred - true)

    if u_var is not None:
        var = u_var[vis_idx].cpu().numpy()

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
        ax1.set_title(f'{channel_names[i]} - Reconstruction- Noisy sensor{noise_level * 100}%', fontsize=10)

        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[i, 2]
        im2 = ax2.imshow(err[i], cmap=cmap, origin=origin)
        ax2.set_title(f'{channel_names[i]} - Abs Error', fontsize=10)

        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        if u_var is not None:
            ax3 = axes[i, 3]
            im3 = ax3.imshow(var[i], cmap='hot', origin=origin)
            ax3.set_title(f'{channel_names[i]} - Uncertainty', fontsize=10)
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()

    import csv
    data_types = {
        'true': true,
        'pred': pred,
        'error': err
    }
    if u_var is not None:
        data_types['variance'] = var

    for data_name, data_array in data_types.items():
        csv_path = save_path.replace('.png', f'_{data_name}.csv')

        Ny, Nx = data_array[0].shape

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['Channel'] + channel_names)
            writer.writerow([])

            x_coords = np.linspace(0, 3.0, Nx)
            y_coords = np.linspace(0, 1.0, Ny)

            for c, ch_name in enumerate(channel_names):
                writer.writerow([f'=== {ch_name} ==='])
                writer.writerow(['y \\ x'] + [f'{x:.4f}' for x in x_coords])

                for iy, y_val in enumerate(y_coords):
                    row = [f'{y_val:.4f}'] + data_array[c][iy, :].tolist()
                    writer.writerow(row)

                writer.writerow([])

def visualize_vector_field(
        u_true,
        u_pred,
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
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_U_statistics(
        u_true,
        u_mean,
        u_var,
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

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, :])
    im0 = ax0.imshow(U_true, cmap='jet', origin='lower', aspect='auto')
    if sample_col is not None:
        ax0.axvline(x=sample_col, color='red', linestyle='--', linewidth=2)
    else:
        ax0.axhline(y=sample_row, color='red', linestyle='--', linewidth=2)
    ax0.set_title(f'U - Ground Truth  (line: {line_label})', fontsize=13)
    ax0.set_xlabel('x', fontsize=11)
    ax0.set_ylabel('y', fontsize=11)
    plt.colorbar(im0, ax=ax0, fraction=0.03, pad=0.02)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(coords, U_true_line, 'b-', linewidth=1.5, label='Ground Truth')
    ax1.plot(coords, U_mean_line, 'r-', linewidth=1.5, label='Mean')
    ax1.fill_between(coords, U_mean_line - 3 * U_std_line, U_mean_line + 3 * U_std_line,
                     color='red', alpha=0.25, label='Mean ± 3σ')
    ax1.set_xlabel('Pixel index', fontsize=11)
    ax1.set_ylabel('U', fontsize=11)
    ax1.set_title(f'U along line {line_label}', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
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

def inference_uncertainty(config, device, num_samples = NUM_SAMPLES):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    noise_level = NOISE_LEVEL
    num_ode_steps = NUM_ODE_STEPS
    sensor_number = SENSOR_NUMBER

    use_conditioning = config.training.random_sensor

    inference_dir = INFERENCE_DIR

    slice_dim = 'z'

    job_name = f"{config.model_name}_sensor_number{sensor_number}_noise_{noise_level}_ode{num_ode_steps}_uncertainty-{num_samples}"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L  = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    slice_idx = W // 2

    all_errors, all_ch_errors = [], []
    vis_sample_idx = 0
    sample_stacks = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            indices = rand_sensor_indices(W, H, L, sensor_number, device, config.seed)

            wid_indices = indices // (H * L)
            row_indices = (indices % (H * L)) // L
            col_indices = (indices % (H * L)) % L

            sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)

            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            noise = torch.randn_like(sensor_value, device = device)
            sensor_value = sensor_value + noise * noise_level

            slice_coords, slice_outputs, batch_inputs, slice_shape = batch_parser.query_slice(
                data, slice_dim = slice_dim, slice_idx=None)

            z_u = encoder(batch_inputs)

            sample_collection = []
            for s in range(num_samples):
                z0 = torch.randn(z_u.shape, device=device)
                z_pred, traj = sample_ode(
                    dit = dit,
                    z0 = z0,
                    sensor_value = sensor_value,
                    sensor_pos = sensor_pos,
                    num_steps = num_ode_steps,
                    use_conditioning = use_conditioning
                )

                u_pred_slice = decoder(z_pred, slice_coords)

                u_pred = u_pred_slice.permute(0,2,1).unsqueeze(2).unsqueeze(3)
                sample_collection.append(u_pred)

            stacked = torch.stack(sample_collection)
            sample_stacks.append(stacked)

            Slice_data = slice_outputs.permute(0,2,1).unsqueeze(2).unsqueeze(3)

            u_true, u_mean, u_var = compute_U_statistics(
                "uncertainty",
                Slice_data,
                stacked,
                scaler
            )

            errors = relative_l2_error(u_mean[:, :5], u_true[:, :5])
            all_errors.extend(errors.cpu().numpy())

            ch_errors = relative_l2_error_per_channel(u_mean, u_true)
            all_ch_errors.extend(ch_errors.cpu().numpy())

            if batch_idx == vis_sample_idx:

                h, w = slice_shape

                u_slice_pred = u_mean.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_slice = u_true.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_var_slice = u_var.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)

                channel_names = ['Ux','Uy','Uz','p','nut','U']

                visualize_results(
                    noise_level = noise_level,
                    u_true = u_slice,
                    u_pred = u_slice_pred,
                    channel_names = channel_names,
                    save_path=f"{save_vis_dir}/inference_result.png",
                    vis_idx = vis_sample_idx,
                    u_var = u_var_slice

                )

                visualize_vector_field(
                    u_true = u_slice,
                    u_pred = u_slice_pred,
                    save_path=f"{save_vis_dir}/vector_field.png",
                    vis_idx = vis_sample_idx
                )

                plot_U_statistics(
                    u_true = u_slice,
                    u_mean = u_slice_pred,
                    u_var = u_var_slice,
                    save_path=f"{save_vis_dir}/U_statistics.png",
                    vis_idx = vis_sample_idx,
                    sample_row = None,
                    sample_col = None,
                )

    print(f"Inference l2 Relative Error: {np.mean(all_errors):.4f} ± {np.std(all_errors, ddof=0):.4f}")

    ch_mean = np.mean(all_ch_errors, axis=0)
    ch_std = np.std(all_ch_errors, axis=0, ddof=0)
    for name, mean, std in zip(['Ux','Uy','Uz','p','nut','U'], ch_mean, ch_std):
        print(f"{name:>8s}: {mean:.4f} ± {std:.4f}")

    np.savetxt(os.path.join(save_vis_dir, 'error_summary.csv'),
           np.array(
               list(zip(
                    ['overall_mean','Ux','Uy','Uz','p','nut','U'],
                    [np.mean(all_errors)] + list(ch_mean),
                    [np.std(all_errors, ddof=0)] + list(ch_std)
                )),
                dtype=object),
           delimiter=',', fmt=['%s','%.6f','%.6f'])

    save_uncertainty(
        save_vis_dir,
        sample_stacks,
        scaler,
        ['Ux', 'Uy', 'Uz', 'p', 'nut', 'U'],
        csv_name='sample_summary.csv',
    )

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

    noise_level = NOISE_LEVEL
    num_ode_steps = NUM_ODE_STEPS
    sensor_number = SENSOR_NUMBER

    use_conditioning = config.training.random_sensor

    inference_dir = INFERENCE_DIR

    slice_dim = 'z'

    job_name = f"{config.model_name}_sensor_number{sensor_number}_noise_{noise_level}_ode{num_ode_steps}"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L  = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    slice_idx = W // 2

    all_errors, all_ch_errors = [], []
    vis_sample_idx = 0

    efficiency.start()
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            indices = rand_sensor_indices(W, H, L, sensor_number, device, config.seed)

            wid_indices = indices // (H * L)
            row_indices = (indices % (H * L)) // L
            col_indices = (indices % (H * L)) % L
            sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)
            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            noise = torch.randn_like(sensor_value, device = device)
            sensor_value = sensor_value + noise * noise_level

            slice_coords, slice_outputs, batch_inputs, slice_shape = batch_parser.query_slice(
                data, slice_dim = slice_dim, slice_idx=None)

            z_u = encoder(batch_inputs)
            z0 = torch.randn(z_u.shape, device = device)

            z_pred, traj = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_value,
                sensor_pos = sensor_pos,
                num_steps = num_ode_steps,
                use_conditioning = use_conditioning
            )

            u_pred_slice = decoder(z_pred, slice_coords)

            u_pred = u_pred_slice.permute(0,2,1).unsqueeze(2).unsqueeze(3)
            Slice_data = slice_outputs.permute(0,2,1).unsqueeze(2).unsqueeze(3)

            u_true, u_pred, _ = compute_U_statistics(
                "inference",
                scaler.inverse(Slice_data),
                scaler.inverse(u_pred),
                scaler
            )

            errors = relative_l2_error(u_pred[:, :5], u_true[:, :5])
            all_errors.extend(errors.cpu().numpy())

            ch_errors = relative_l2_error_per_channel(u_pred, u_true)
            all_ch_errors.extend(ch_errors.cpu().numpy())

            if batch_idx == vis_sample_idx:
                efficiency.pause()

                h, w = slice_shape
                u_slice_pred = u_pred.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_slice = u_true.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)

                channel_names = ['Ux','Uy','Uz','p','nut','U']

                visualize_results(
                    noise_level = noise_level,
                    u_true = u_slice,
                    u_pred = u_slice_pred,
                    channel_names = channel_names,
                    save_path=f"{save_vis_dir}/inference_result.png",
                    vis_idx = vis_sample_idx
                )

                visualize_vector_field(
                    u_true = u_slice,
                    u_pred = u_slice_pred,
                    save_path=f"{save_vis_dir}/vector_field.png",
                    vis_idx = vis_sample_idx
                )
                efficiency.resume()

    print(f"Mean Relative Error: {np.mean(all_errors):.4f}")

    ch_mean = np.mean(all_ch_errors, axis=0)
    for name, err in zip(['Ux','Uy','Uz','p','nut','U'], ch_mean):
        print(f"{name:>8s}: {err:.4f}")

    np.savetxt(os.path.join(save_vis_dir, 'error_summary.csv'),
           np.array(list(zip(['overall_mean','Ux','Uy','Uz','p','nut','U'],
                             [np.mean(all_errors)] + list(ch_mean)
                             )),
                    dtype=object),
           delimiter=',', fmt=['%s','%.6f'])

    save_efficiency(
        os.path.join(save_vis_dir, 'computer_efficiency.csv'),
        efficiency.finish(),
    )

def visualize_3d_slices(
        u_true_list,
        u_pred_list,
        z_positions,
        channel_names,
        save_path,
        vis_idx,
        u_var_list = None
    ):

    num_slices = len(u_true_list)

    true_list, pred_list, err_list = [], [], []
    for s in range(num_slices):
        true = u_true_list[s][vis_idx].cpu().numpy()
        pred = u_pred_list[s][vis_idx].cpu().numpy()
        err  = np.abs(pred - true)

        true_list.append(true)
        pred_list.append(pred)
        err_list.append(err)

    import csv
    for s, z_pos in enumerate(z_positions):
        for c, ch_name in enumerate(channel_names):
            csv_path = save_path.replace('.png', f'_z{z_pos:.2f}_{ch_name}.csv')

            data_2d = {
                'true': true_list[s][c],
                'pred': pred_list[s][c],
                'error': err_list[s][c]
            }
            if u_var_list is not None:
                var_data = u_var_list[s][vis_idx].cpu().numpy()
                if c < var_data.shape[0]:
                    data_2d['variance'] = var_data[c]

            Ny, Nx = data_2d['true'].shape
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                x_coords = np.linspace(0, 3.0, Nx)
                header = ['y'] + [f'x={x:.4f}' for x in x_coords]
                writer.writerow([''] + header)

                for data_type, data_array in data_2d.items():
                    writer.writerow([f'--- {data_type} ---'])
                    y_coords = np.linspace(0, 1.0, Ny)
                    for iy, y_val in enumerate(y_coords):
                        row = [f'{y_val:.4f}'] + data_array[iy, :].tolist()
                        writer.writerow(row)
                    writer.writerow([])

    outline_box = pv.Box(bounds=(0, 3.0, 0, 1.0, 0, 1.0)).outline()

    for i in range(len(channel_names)):

        vmin_tp = min(np.min(true_list[s][i]) for s in range(num_slices))
        vmax_tp = max(np.max(true_list[s][i]) for s in range(num_slices))
        vmin_e  = min(np.min(err_list[s][i])  for s in range(num_slices))
        vmax_e  = max(np.max(err_list[s][i])  for s in range(num_slices))

        for g_label, data_list, cmap, clim_range in [
            ('Ground Truth',   true_list, 'jet', [vmin_tp, vmax_tp]),
            ('Reconstruction', pred_list, 'jet', [vmin_tp, vmax_tp]),
            ('Abs Error',      err_list,  'hot', [vmin_e,  vmax_e]),
            ]:
            plotter = pv.Plotter(window_size=(1600, 1200))

            for s in range(num_slices):

                scalar = data_list[s][i]
                Ny, Nx = scalar.shape
                x = np.linspace(0, 3.0, Nx)
                y = np.linspace(0, 1.0, Ny)
                X, Y = np.meshgrid(x, y, indexing='ij')
                Z = np.full_like(X, 1 - z_positions[s])
                grid = pv.StructuredGrid(X, Y, Z)
                grid['scalar'] = scalar.T.flatten(order='F')
                opacity = 0.7 if s == 1 else 1
                plotter.add_mesh(grid, scalars='scalar', cmap=cmap, opacity=opacity,
                                 show_scalar_bar=(s == num_slices - 1),
                                 lighting=False,
                                 clim=clim_range)

            plotter.add_mesh(outline_box, color='k', line_width=3)
            plotter.add_text(f'{channel_names[i]} - {g_label}', position='upper_left', font_size=10)
            plotter.camera_position = [
                (5, 3, 5.0),
                (1.5, 0.5, 0.5),
                (0, 1, 0)
            ]

            plotter.screenshot(save_path, transparent_background=True)
            plotter.show()
            plotter.close()

def inference_3d(config, device):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    noise_level = NOISE_LEVEL
    num_ode_steps = NUM_ODE_STEPS
    sensor_number = SENSOR_NUMBER

    use_conditioning = config.training.random_sensor

    inference_dir = INFERENCE_DIR

    slice_dim = 'z'

    job_name = f"{config.model_name}_sensor_number{sensor_number}_noise_{noise_level}_ode{num_ode_steps}_3slice"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L  = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    z_indices = [0, W // 2, W - 1]
    z_positions = [zi / (W - 1) for zi in z_indices]

    all_errors = []
    vis_sample_idx = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            indices = rand_sensor_indices(W, H, L, sensor_number, device, config.seed)

            wid_indices = indices // (H * L)
            row_indices = (indices % (H * L)) // L
            col_indices = (indices % (H * L)) % L
            sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)
            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            noise = torch.randn_like(sensor_value, device = device)
            sensor_value = sensor_value + noise * noise_level

            z_u = encoder(data)
            z0 = torch.randn(z_u.shape, device = device)

            z_pred, _ = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_value,
                sensor_pos = sensor_pos,
                num_steps = num_ode_steps,
                use_conditioning = use_conditioning
            )

            u_pred_list, u_true_list  = [],[]
            for zi in z_indices:

                slice_coords, slice_outputs, _, slice_shape = batch_parser.query_slice(
                    data, slice_dim = slice_dim, slice_idx=zi)

                u_pred_slice = decoder(z_pred, slice_coords)
                u_pred    = u_pred_slice.permute(0,2,1).unsqueeze(2).unsqueeze(3)
                Slice_data = slice_outputs.permute(0,2,1).unsqueeze(2).unsqueeze(3)

                u_true, u_pred, _ = compute_U_statistics(
                    "inference",
                    scaler.inverse(Slice_data),
                    scaler.inverse(u_pred),
                    scaler
                )

                errors = relative_l2_error(u_pred[:, :5], u_true[:, :5])
                all_errors.extend(errors.cpu().numpy())

                h, w = slice_shape
                u_slice_pred = u_pred.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_slice      = u_true.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)

                u_true_list.append(u_slice)
                u_pred_list.append(u_slice_pred)

            channel_names = ['Ux', 'Uy', 'Uz', 'p', 'nut', 'U']
            visualize_3d_slices(
                u_true_list          = u_true_list,
                u_pred_list          = u_pred_list,
                z_positions          = z_positions,
                channel_names        = channel_names,
                save_path            = f"{save_vis_dir}/inference_result.png",
                vis_idx              = vis_sample_idx,
            )

            break

def inference_3d_uncertainty(config, device, num_samples = NUM_SAMPLES):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    noise_level = NOISE_LEVEL
    num_ode_steps = NUM_ODE_STEPS
    sensor_number = SENSOR_NUMBER

    use_conditioning = config.training.random_sensor

    inference_dir = INFERENCE_DIR

    slice_dim = 'z'

    job_name = f"{config.model_name}_sensor_number{sensor_number}_noise_{noise_level}_ode{num_ode_steps}_3slice_uncertainty"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L  = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    z_indices = [0, W // 2, W - 1]
    z_positions = [zi / (W - 1) for zi in z_indices]

    all_errors = []
    vis_sample_idx = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            indices = rand_sensor_indices(W, H, L, sensor_number, device, config.seed)

            wid_indices = indices // (H * L)
            row_indices = (indices % (H * L)) // L
            col_indices = (indices % (H * L)) % L
            sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)
            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            noise = torch.randn_like(sensor_value, device = device)
            sensor_value = sensor_value + noise * noise_level

            z_u = encoder(data)

            sample_collection = []
            slice_outputs_list = []
            slice_shapes = []
            for s in range(num_samples):

                z0 = torch.randn(z_u.shape, device=device)
                z_pred, traj = sample_ode(
                    dit = dit,
                    z0 = z0,
                    sensor_value = sensor_value,
                    sensor_pos = sensor_pos,
                    num_steps = num_ode_steps,
                    use_conditioning = use_conditioning
                )

                for slice_i, zi in enumerate(z_indices):
                    slice_coords, slice_outputs, _, slice_shape = batch_parser.query_slice(
                        data, slice_dim = slice_dim, slice_idx=zi)

                    u_pred_slice = decoder(z_pred, slice_coords)
                    u_pred    = u_pred_slice.permute(0,2,1).unsqueeze(2).unsqueeze(3)
                    sample_collection[slice_i].append(u_pred)
                    slice_outputs_list[slice_i] = slice_outputs
                    slice_shapes[slice_i] = slice_shape

            u_true_list, u_pred_mean_list, u_var_list = [], [], []
            for slice_i in range(len(z_indices)):
                stacked = torch.stack(sample_collection[slice_i])

                Slice_data = slice_outputs_list[slice_i].permute(0,2,1).unsqueeze(2).unsqueeze(3)
                u_true, u_mean, u_var = compute_U_statistics(
                    "uncertainty",
                    Slice_data,
                    stacked,
                    scaler
                )

                errors = relative_l2_error(u_mean[:, :5], u_true[:, :5])
                all_errors.extend(errors.cpu().numpy())

                h, w = slice_shapes[slice_i]
                u_true_list.append(u_true.squeeze(2).squeeze(2).reshape(B, C + 1, h, w))
                u_pred_mean_list.append(u_mean.squeeze(2).squeeze(2).reshape(B, C + 1, h, w))
                u_var_list.append(u_var.squeeze(2).squeeze(2).reshape(B, C + 1, h, w))

            channel_names = ['Ux', 'Uy', 'Uz', 'p', 'nut', 'U']
            visualize_3d_slices(
                u_true_list          = u_true_list,
                u_pred_list          = u_pred_mean_list,
                z_positions          = z_positions,
                channel_names        = channel_names,
                save_path            = f"{save_vis_dir}/inference_result.png",
                vis_idx              = vis_sample_idx,
                u_var_list           = u_var_list
            )

            break

def inference_flow(config, device):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    noise_level = NOISE_LEVEL
    num_ode_steps = NUM_ODE_STEPS
    sensor_number = SENSOR_NUMBER

    use_conditioning = config.training.random_sensor

    inference_dir = INFERENCE_DIR

    job_name = f"{config.model_name}_sensor_number{sensor_number}_noise_{noise_level}_ode{num_ode_steps}_flow"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L  = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    all_errors = []
    vis_sample_idx = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            indices = rand_sensor_indices(W, H, L, sensor_number, device, config.seed)

            wid_indices = indices // (H * L)
            row_indices = (indices % (H * L)) // L
            col_indices = (indices % (H * L)) % L
            sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)
            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            noise = torch.randn_like(sensor_value, device = device)
            sensor_value = sensor_value + noise * noise_level

            z_u = encoder(data)
            z0 = torch.randn(z_u.shape, device = device)

            z_pred, _ = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_value,
                sensor_pos = sensor_pos,
                num_steps = num_ode_steps,
                use_conditioning = use_conditioning
            )

            result_cache_path = os.path.join(save_vis_dir, "pred3d.pt")
            if os.path.exists(result_cache_path):

                print(f"Loading result cache: .........")

                rc = torch.load(result_cache_path, map_location='cpu', weights_only=False)
                u_pred_3d = rc['u_pred_3d']
                u_true_3d = rc['u_true_3d']

            else:

                true_slices, pred_slices = [],[]
                for zi in range(W):

                    slice_coords, slice_outputs, _, slice_shape = batch_parser.query_slice(
                        data, slice_dim='z', slice_idx=zi)

                    u_slice = decoder(z_pred, slice_coords)

                    pred_slices.append(u_slice)
                    true_slices.append(slice_outputs)

                u_pred = torch.cat(pred_slices, dim=1)
                u_true = torch.cat(true_slices, dim=1)

                u_pred_3d = scaler.inverse(u_pred.permute(0,2,1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)
                u_true_3d = scaler.inverse(u_true.permute(0,2,1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)

                torch.save({'u_pred_3d': u_pred_3d.cpu(),
                            'u_true_3d': u_true_3d.cpu()
                            }, result_cache_path)

            errors = relative_l2_error(u_pred_3d, u_true_3d)
            all_errors.extend(errors.cpu().numpy())

            gt_np = u_true_3d.cpu().numpy()

            plot_phy = ['Velocity_Magnitude', 'Ux', 'Uy', 'Uz', 'p', 'nut']
            for variable in plot_phy:

                vmin, vmax = _scalar_range(gt_np, vis_sample_idx, variable)
                clim = (vmin, vmax)

                u_err_3d = make_error_dataset(u_true_3d, u_pred_3d, variable)

                for label, dataset_3d in [('Ground_Truth', u_true_3d),
                                          ('Reconstruction', u_pred_3d),
                                          ('Abs_Error', u_err_3d)]:

                    plot_3d_flow_simple(
                        dataset = dataset_3d.cpu().numpy(),
                        time_idx = vis_sample_idx,
                        variable = variable,
                        plot_type = 'slice',
                        clim =  None if label=='Abs_Error' else clim ,
                        title_label = label,
                        save_path = f"{save_vis_dir}/{label}_{variable}.png",
                        show_interactive = False
                    )

            plot_3d_flow_simple(
                dataset = u_true_3d.cpu().numpy(),
                time_idx = vis_sample_idx,
                variable = 'Velocity_Magnitude',
                plot_type = 'sensor_only',
                title_label = 'Sensor Positions',
                rand_z = wid_indices, rand_y = row_indices, rand_x = col_indices,
                save_path = f"{save_vis_dir}/Sensor_Positions.png",
                show_interactive = False
            )

            break

def inference_flow_uncertainty(config, device, num_samples = NUM_SAMPLES):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    noise_level = NOISE_LEVEL
    num_ode_steps = NUM_ODE_STEPS
    sensor_number = SENSOR_NUMBER

    use_conditioning = config.training.random_sensor

    inference_dir = INFERENCE_DIR

    job_name = f"{config.model_name}_sensor_number{sensor_number}_noise_{noise_level}_ode{num_ode_steps}_flow_uncertainty"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L  = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    all_errors = []
    vis_sample_idx = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            indices = rand_sensor_indices(W, H, L, sensor_number, device, config.seed)

            wid_indices = indices // (H * L)
            row_indices = (indices % (H * L)) // L
            col_indices = (indices % (H * L)) % L
            sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)
            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            noise = torch.randn_like(sensor_value, device = device)
            sensor_value = sensor_value + noise * noise_level

            z_u = encoder(data)

            result_cache_path = os.path.join(save_vis_dir, "pred3d_uncertainty.pt")
            if os.path.exists(result_cache_path):

                print(f"Loading result cache: ..........")

                rc = torch.load(result_cache_path, map_location='cpu', weights_only=False)
                u_pred_mean = rc['u_pred_mean']
                u_true_3d = rc['u_true_3d']

            else:

                sample_3d_collection = []
                for s in range(num_samples):

                    z0 = torch.randn(z_u.shape, device=device)

                    z_pred, traj = sample_ode(
                        dit = dit,
                        z0 = z0,
                        sensor_value = sensor_value,
                        sensor_pos = sensor_pos,
                        num_steps = num_ode_steps,
                        use_conditioning = use_conditioning
                    )

                    pred_slices, true_slices = [],[]
                    for zi in range(W):

                        slice_coords, slice_outputs, _, slice_shape = batch_parser.query_slice(
                            data, slice_dim='z', slice_idx=zi)

                        u_slice = decoder(z_pred, slice_coords)
                        pred_slices.append(u_slice)
                        true_slices.append(slice_outputs)

                    u_pred = torch.cat(pred_slices, dim=1)
                    u_true = torch.cat(true_slices, dim=1)

                    u_pred_3d = scaler.inverse(u_pred.permute(0,2,1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)
                    u_true_3d = scaler.inverse(u_true.permute(0,2,1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)

                    sample_3d_collection.append(u_pred_3d)

                stacked = torch.stack(sample_3d_collection)
                u_pred_mean = stacked.mean(dim=0)

                torch.save({'u_pred_mean': u_pred_mean.cpu(),
                            'u_true_3d': u_true_3d.cpu()
                            }, result_cache_path)

            errors = relative_l2_error(u_pred_mean, u_true_3d)
            all_errors.extend(errors.cpu().numpy())

            gt_np = u_true_3d.cpu().numpy()

            plot_phy = ['Velocity_Magnitude', 'Ux', 'p']
            for variable in plot_phy:

                vmin, vmax = _scalar_range(gt_np, vis_sample_idx, variable)
                clim = (vmin, vmax)

                u_err_3d = make_error_dataset(u_true_3d, u_pred_3d, variable)

                for label, dataset_3d in [('Ground_Truth', u_true_3d),
                                          ('Reconstruction', u_pred_mean),
                                          ('Abs_Error', u_err_3d)]:

                    plot_3d_flow_simple(
                        dataset = dataset_3d.cpu().numpy(),
                        time_idx = vis_sample_idx,
                        variable = variable,
                        plot_type = 'vortex',
                        clim =  None if label=='Abs_Error' else clim ,
                        title_label = label,
                        save_path = f"{save_vis_dir}/{label}_{variable}.png",
                        show_interactive = False
                    )

            plot_3d_flow_simple(
                dataset = u_true_3d.cpu().numpy(),
                time_idx = vis_sample_idx,
                variable = 'Velocity_Magnitude',
                plot_type = 'sensor_only',
                title_label = 'Sensor Positions',
                rand_z = wid_indices, rand_y = row_indices, rand_x = col_indices,
                save_path = f"{save_vis_dir}/Sensor_Positions.png",
                show_interactive = False
            )

            break
