import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from models.EncoderDecoder import Encoder, Decoder
from utils.model_utils import create_optimizer, compute_total_params, save_loss_to_csv, plot_and_save_loss_curve
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, save_dit_checkpoint, load_dit_checkpoint
from data_utils import create_dataloader, BatchParser
from models.flow import DiT
from model.model_utils import MSE, RMSE, append_U_channel, compute_U_statistics, relative_l2_error
import matplotlib.pyplot as plt
from einops import rearrange
import imageio
import random
import seaborn as sns
import io

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def diffusion_loss_fn(dit, diffusion_batch, sensor_value, sensor_pos, use_conditioning):

    if use_conditioning:

        z_t, t, target = diffusion_batch

        pred = dit(z_t, t, sensor_value, sensor_pos)

    else:
        z_t, t, target = diffusion_batch
        pred = dit(z_t, t)

    loss = MSE(pred, target)

    return loss

def get_diffusion_batch(z_u):

    batch_size = z_u.shape[0]

    z0 = torch.randn_like(z_u)

    t = torch.rand(batch_size, 1, 1, device = z_u.device)

    z_t = t * z_u + (1 - t) * z0

    target = z_u - z0

    t_flat = t.flatten()

    return z_t, t_flat, target

def train_diffusion_epoch(dit, encoder, train_loader, batch_parser, optimizer, config, device):

    dit.train()
    encoder.eval()

    total_loss = 0.0
    use_conditioning = config.training.random_sensor

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training',leave=False)
    for batch_idx, data in progress_bar:

        data = data.to(device)

        B, C, W, H, L = data.shape

        sensor_value, sensor_pos = None, None
        if use_conditioning:

            m = random.randint(20, 200)
            indices = torch.randperm(W * H * L)[:m].to(device)
            wid_indices = indices // (H * L)
            row_indices = (indices % (H * L)) // L
            col_indices = (indices % (H * L)) % L

            sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)

            sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

        z_u = encoder(data)

        diffusion_batch = get_diffusion_batch(z_u)

        optimizer.zero_grad()

        loss = diffusion_loss_fn(
            dit, diffusion_batch,
            sensor_value, sensor_pos,
            use_conditioning = use_conditioning
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            dit.parameters(),
            config.optim.clip_norm
            )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def sample_ode(dit, z0, sensor_value = None, sensor_pos = None, num_steps=50, use_conditioning=False):

    dt = 1.0 / num_steps
    z = z0
    traj = [z.detach().cpu()]

    for i in tqdm(range(num_steps), desc="ODE Sampling",leave=False):
        t = torch.full((z.shape[0],), i / num_steps, device=z.device)

        pred = dit(z, t, sensor_value, sensor_pos) if use_conditioning else dit(z, t)

        z = z + pred * dt

        traj.append(z.detach().cpu())

    return z, traj

def validate_epoch(dit, encoder, decoder, dev_loader, batch_parser, device, config, scaler):

    dit.eval()
    encoder.eval()
    decoder.eval()

    num_steps = config.training.num_ode_steps
    total_rmse = 0.0
    use_conditioning = config.training.random_sensor

    progress_bar = tqdm(enumerate(dev_loader), total=len(dev_loader), desc='Validating',leave=False)
    with torch.no_grad():
        for batch_idx, data in progress_bar:

            data = data.to(device)

            B, C, W, H, L = data.shape

            z_u = encoder(data)

            z0 = torch.randn_like(z_u, device = device)

            sensor_value, sensor_pos = None, None
            if use_conditioning:

                m = random.randint(20, 200)
                indices = torch.randperm(W * H * L)[:m].to(device)
                wid_indices = indices // (H * L)
                row_indices = (indices % (H * L)) // L
                col_indices = (indices % (H * L)) % L

                sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)

                sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            z_pred, traj = sample_ode(
                    dit = dit,
                    z0 = z0,
                    sensor_value = sensor_value,
                    sensor_pos = sensor_pos,
                    num_steps = num_steps,
                    use_conditioning = use_conditioning
                    )

            rmse = RMSE(z_pred, z_u)
            total_rmse += rmse.item()

    avg_rmse = total_rmse / len(dev_loader)
    return avg_rmse

def validate_epoch_decoder(dit, encoder, decoder, dev_loader, batch_parser, device, config, scaler):

    dit.eval()
    encoder.eval()
    decoder.eval()

    num_steps = config.training.num_ode_steps
    total_rmse = 0.0
    slice_dim = 'z'
    use_conditioning = config.training.random_sensor

    progress_bar = tqdm(enumerate(dev_loader), total=len(dev_loader), desc='Validating',leave=False)
    with torch.no_grad():
        for batch_idx, data in progress_bar:

            data = data.to(device)

            B, C, W, H, L = data.shape

            slice_coords, slice_outputs, batch_inputs, slice_shape = batch_parser.query_slice(
                data, slice_dim = slice_dim, slice_idx=None)

            z_u = encoder(batch_inputs)

            z0 = torch.randn_like(z_u, device = device)

            sensor_value, sensor_pos = None, None
            if use_conditioning:

                m = random.randint(20, 200)
                indices = torch.randperm(W * H * L)[:m].to(device)
                wid_indices = indices // (H * L)
                row_indices = (indices % (H * L)) // L
                col_indices = (indices % (H * L)) % L

                sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)

                sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

            z_pred, traj = sample_ode(
                    dit = dit,
                    z0 = z0,
                    sensor_value = sensor_value,
                    sensor_pos = sensor_pos,
                    num_steps = num_steps,
                    use_conditioning = use_conditioning
                    )

            u_pred_slice = decoder(z_pred, slice_coords)

            u_pred = u_pred_slice.permute(0,2,1).unsqueeze(2).unsqueeze(3)
            Slice_data = slice_outputs.permute(0,2,1).unsqueeze(2).unsqueeze(3)

            rmse = RMSE(scaler.inverse(u_pred), scaler.inverse(Slice_data))

            total_rmse += rmse.item()

    avg_rmse = total_rmse / len(dev_loader)
    return avg_rmse

def test_diffusion(config, batch_parser, test_loader, scaler, device):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    total_test_rmse = []
    vis_batch = 0
    save_vis_dir = PROJECT_DIR
    slice_dim = 'z'
    num_ode_steps = config.training.num_ode_steps

    num_sensor = 20

    use_conditioning = config.training.random_sensor

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing Diffusion',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            slice_coords, slice_outputs, batch_inputs, slice_shape = batch_parser.query_slice(
                data, slice_dim = slice_dim, slice_idx=None)

            z_u = encoder(batch_inputs)

            z0 = torch.randn(z_u.shape, device = device)

            sensor_value, sensor_pos = None, None
            if use_conditioning:

                indices = torch.randperm(W * H * L)[:num_sensor].to(device)
                wid_indices = indices // (H * L)
                row_indices = (indices % (H * L)) // L
                col_indices = (indices % (H * L)) % L

                sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)

                sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

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

            rmse = relative_l2_error(scaler.inverse(u_pred), scaler.inverse(Slice_data))
            total_test_rmse.extend(rmse.cpu().numpy())

            if batch_idx == vis_batch:

                h, w = slice_shape

                u_slice_pred = scaler.inverse(u_pred).squeeze(2).squeeze(2).reshape(B, C, h, w)
                u_slice = scaler.inverse(Slice_data).squeeze(2).squeeze(2).reshape(B, C, h, w)

                u_slice = append_U_channel(u_slice)
                u_slice_pred = append_U_channel(u_slice_pred)

                true_field = u_slice.cpu().numpy()[vis_batch]
                pred_field = u_slice_pred.cpu().numpy()[vis_batch]
                error_field = np.abs(true_field - pred_field)

                channel_names = ['Ux','Uy','Uz','p','nut','U']
                C = len(channel_names)
                fig, axes = plt.subplots(C, 3, figsize=(3 * 4, C * 3))

                for i in range(C):

                    vmin = true_field[i].min()
                    vmax = true_field[i].max()

                    im1 = axes[i, 0].imshow(true_field[i, :, :], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                    axes[i, 0].set_title(f'{channel_names[i]} - Ground Truth')
                    axes[i, 0].axis('off')
                    plt.colorbar(im1, ax=axes[i, 0])

                    im2 = axes[i, 1].imshow(pred_field[i, :, :], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                    axes[i, 1].set_title(f'{channel_names[i]} - pred')
                    axes[i, 1].axis('off')
                    plt.colorbar(im2, ax=axes[i, 1])

                    im3 = axes[i, 2].imshow(error_field[i, :, :], cmap='jet', origin='lower')
                    axes[i, 2].set_title(f'{channel_names[i]} - Absolute Error')
                    axes[i, 2].axis('off')
                    plt.colorbar(im3, ax=axes[i, 2])

                plt.tight_layout()
                vis_path = os.path.join(save_vis_dir, f"{config.model_name}_sensor{num_sensor}.png")
                plt.savefig(vis_path, bbox_inches='tight', dpi=300, transparent=True)
                plt.close()

    print(f"Test l2 Relative Error: {np.mean(total_test_rmse):.4f}")

def test_diffusion_uncertainty(config, batch_parser, test_loader, scaler, device, num_samples=10):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    encoder.eval()
    decoder.eval()
    dit.eval()

    total_test_rmse = []
    vis_batch = 0
    save_vis_dir = PROJECT_DIR
    slice_dim = 'z'
    num_ode_steps = config.training.num_ode_steps

    num_sensor = 20

    use_conditioning = config.training.random_sensor

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing Diffusion',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            slice_coords, slice_outputs, batch_inputs, slice_shape = batch_parser.query_slice(
                data, slice_dim = slice_dim, slice_idx=None)

            z_u = encoder(batch_inputs)

            sensor_value, sensor_pos = None, None
            if use_conditioning:

                indices = torch.randperm(W * H * L)[:num_sensor].to(device)
                wid_indices = indices // (H * L)
                row_indices = (indices % (H * L)) // L
                col_indices = (indices % (H * L)) % L

                sensor_value = data[:, :, wid_indices, row_indices, col_indices].permute(0, 2, 1)

                sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

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

            Slice_data = slice_outputs.permute(0,2,1).unsqueeze(2).unsqueeze(3)

            u_true, u_mean, u_var = compute_U_statistics(
                "uncertainty",
                Slice_data,
                stacked,
                scaler
            )

            rmse = relative_l2_error(u_mean[:, :5], u_true[:, :5])
            total_test_rmse.extend(rmse.cpu().numpy())

            if batch_idx == vis_batch:

                h, w = slice_shape

                u_slice_pred = u_mean.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_slice = u_true.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_var_slice = u_var.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)

                true_field = u_slice.cpu().numpy()[vis_batch]
                pred_field = u_slice_pred.cpu().numpy()[vis_batch]
                error_field = np.abs(true_field - pred_field)
                var_field = u_var_slice.cpu().numpy()[vis_batch]

                channel_names = ['Ux','Uy','Uz','p','nut','U']
                C = len(channel_names)
                fig, axes = plt.subplots(C, 4, figsize=(4 * 4, C * 3))

                for i in range(C):

                    vmin = true_field[i].min()
                    vmax = true_field[i].max()

                    im1 = axes[i, 0].imshow(true_field[i, :, :], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                    axes[i, 0].set_title(f'{channel_names[i]} - Ground Truth')
                    axes[i, 0].axis('off')
                    plt.colorbar(im1, ax=axes[i, 0])

                    im2 = axes[i, 1].imshow(pred_field[i, :, :], cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
                    axes[i, 1].set_title(f'{channel_names[i]} - pred')
                    axes[i, 1].axis('off')
                    plt.colorbar(im2, ax=axes[i, 1])

                    im3 = axes[i, 2].imshow(error_field[i, :, :], cmap='jet', origin='lower')
                    axes[i, 2].set_title(f'{channel_names[i]} - Absolute Error')
                    axes[i, 2].axis('off')
                    plt.colorbar(im3, ax=axes[i, 2])

                    im4 = axes[i, 3].imshow(var_field[i, :, :], cmap='hot', origin='lower')
                    axes[i, 3].set_title(f'{channel_names[i]} - Uncertainty')
                    axes[i, 3].axis('off')
                    plt.colorbar(im4, ax=axes[i, 3])

                plt.tight_layout()
                vis_path = os.path.join(save_vis_dir, f"{config.model_name}_sensor{num_sensor}_uncertainty.png")
                plt.savefig(vis_path, bbox_inches='tight', dpi=300, transparent=True)
                plt.close()

    print(f"Test l2 Relative Error (mean of {num_samples} samples): {np.mean(total_test_rmse):.4f}")

def plot_ode_trajectory(traj, decoder, z_u, coords, scaler, H, W, channel_names, save_path, fps = 10):

    gif_save_dir = os.path.join(save_path, "gifs")
    pca_save_dir = os.path.join(save_path, "pca")
    os.makedirs(gif_save_dir, exist_ok=True)
    os.makedirs(pca_save_dir, exist_ok=True)
    C = len(channel_names)

    global_mins = [float('inf')] * C
    global_maxs = [-float('inf')] * C

    with torch.no_grad():
        for z in tqdm(traj, desc="预计算颜色范围", leave=False):

            z = z.to(coords.device)
            u_pred = decoder(z, coords)
            u_pred = rearrange(u_pred, "b (h w) c -> b c h w", h=H, w=W).unsqueeze(-1)
            u_pred_scaled = scaler.inverse(u_pred).squeeze(-1)
            U = torch.sqrt(u_pred_scaled[:, 0]**2 + u_pred_scaled[:, 1]**2 + u_pred_scaled[:, 2]**2)
            u_pred_scaled = torch.cat([u_pred_scaled, U.unsqueeze(1)], dim=1)

            for c in range(C):
                field = u_pred_scaled[0, c].cpu().numpy()
                global_mins[c] = min(global_mins[c], field.min())
                global_maxs[c] = max(global_maxs[c], field.max())

    for channel_idx in range(C):
        channel_name = channel_names[channel_idx]
        frame_paths = []

        with torch.no_grad():
            for i, z in enumerate(traj):

                z = z.to(coords.device)
                u_pred = decoder(z, coords)
                u_pred = rearrange(u_pred, "b (h w) c -> b c h w", h=H, w=W).unsqueeze(-1)
                u_pred_scaled = scaler.inverse(u_pred).squeeze(-1)
                U = torch.sqrt(u_pred_scaled[:, 0]**2 + u_pred_scaled[:, 1]**2 + u_pred_scaled[:, 2]**2)
                u_pred_scaled = torch.cat([u_pred_scaled, U.unsqueeze(1)], dim=1)

                field = u_pred_scaled[0, channel_idx].cpu().numpy()

                plt.figure(figsize=(6, 5))
                im = plt.imshow(field, cmap='jet', origin='lower', vmin=global_mins[channel_idx], vmax=global_maxs[channel_idx])
                plt.title(f'{channel_name} - Step {i}', fontsize=14)
                plt.axis('off')
                plt.colorbar(im, fraction=0.046, pad=0.04)

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
                plt.close()
                buf.seek(0)
                frame_paths.append(imageio.imread(buf))

        gif_path = os.path.join(gif_save_dir, f"{channel_name}.gif")
        imageio.mimsave(gif_path, frame_paths, fps=fps, loop=0)

    num_steps = len(traj)
    B = traj[0].shape[0]

    traj_np = []
    for z in traj:
        traj_np.append(z.cpu().numpy().reshape(B, -1).flatten())

    true_z_flat = z_u.cpu().numpy().reshape(B, -1).flatten()
    gaussian_ref = np.random.randn(traj_np[0].shape[0])

    kde_frames = []
    X_LIMIT = (-5, 5)
    Y_LIMIT = (0, 0.7)

    for i in range(0, num_steps, 1):
        plt.figure(figsize=(10, 6))

        sns.kdeplot(gaussian_ref, label='Standard Gaussian',
                    color='gray', linestyle=':', linewidth=2, alpha=0.7)
        sns.kdeplot(true_z_flat, label='Ground Truth',
                    color='black', linestyle='--', linewidth=3, alpha=0.8)

        current_data = traj_np[i]

        if i == 0:
            current_color = '#d62728'
            current_label = 'Current (Start)'
        elif i == num_steps - 1:
            current_color = '#2ca02c'
            current_label = 'Current (Final)'
        else:
            current_color = '#1f77b4'
            current_label = f'Current Step {i}'

        sns.kdeplot(current_data, label=current_label,
                    color=current_color, linewidth=3.5)

        plt.title(f'Latent Distribution Shift | Step {i}/{num_steps-1}', fontsize=15, pad=15)
        plt.xlabel('Latent Feature Value', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.xlim(X_LIMIT)
        plt.ylim(Y_LIMIT)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, transparent=True)
        plt.close()
        buf.seek(0)
        kde_frames.append(imageio.imread(buf))

    kde_gif_path = os.path.join(pca_save_dir, "latent_distribution_shift.gif")
    imageio.mimsave(kde_gif_path, kde_frames, fps=fps, loop = 1)

def train_and_evaluate(config, device):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)

    for param in encoder.parameters():
        param.requires_grad = False

    dit = DiT(config).to(device)
    compute_total_params(dit)

    optimizer, lr_scheduler = create_optimizer(config, dit)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(train_loader))
    B, C, W, H, L = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    job_name = f"{config.model_name}_pde_{config.fae.training.use_pde}_ode_{config.training.num_ode_steps}"
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    exp_dir = os.path.join(base_dir, job_name)
    ckpt_path = os.path.join(exp_dir, "ckpt")

    os.makedirs(ckpt_path, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=4)

    total_steps = 0
    max_steps = config.training.max_steps
    log_interval = config.logging.log_interval

    best_dev_rmse = float('inf')
    epoch_list = []
    train_loss_list = []
    val_rmse_list = []
    for epoch in range(max_steps):

        avg_loss = train_diffusion_epoch(
            dit, encoder, train_loader, batch_parser,
            optimizer, config, device
        )

        val_rmse = 0

        lr_scheduler.step()

        total_steps += 1
        epoch_list.append(total_steps)
        train_loss_list.append(avg_loss)
        val_rmse_list.append(val_rmse)

        save_loss_to_csv(epoch_list, train_loss_list, val_rmse_list, None, exp_dir)
        plot_and_save_loss_curve(epoch_list, train_loss_list, val_rmse_list, None, exp_dir)

        if total_steps % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"step: {total_steps} | LR: {current_lr:.3e} | train loss: {avg_loss:.3e} | valide loss: {val_rmse:.3e}")

        save_interval = 50
        if total_steps % save_interval == 0:
            save_dit_checkpoint(ckpt_path, dit, optimizer, total_steps)