import os
import json
import time
import torch
import numpy as np
from models.EncoderDecoder import Encoder, Decoder
from utils.model_utils import create_optimizer, compute_total_params, save_loss_to_csv, plot_and_save_loss_curve
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from data_utils import create_dataloader, BatchParser
from model.model_utils import append_U_channel, loss_fn, u_net, RMSE, l2_loss, relative_l2_error
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_epoch(encoder, decoder, train_loader,
                batch_parser, optimizer,
                config, device, scaler, epoch):

    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_loss_data = 0.0
    total_loss_res = 0.0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training',leave=False)
    for batch_idx, data in progress_bar:

        data = data.to(device)

        if config.training.random_resolution:
            batch_data = batch_parser.random_query(data)
        else:
            batch_data = batch_parser.query_all(data)

        optimizer.zero_grad()

        loss, loss_data, loss_res = loss_fn(encoder, decoder,
                                            batch_data, scaler,
                                            config.training.use_pde, config.training.pde_r, config.training.max_steps, epoch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            config.optim.clip_norm
        )

        optimizer.step()

        total_loss += loss.item()
        total_loss_data += loss_data.item()
        total_loss_res += loss_res.item()

    avg_loss = total_loss / len(train_loader)
    avg_loss_data = total_loss_data / len(train_loader)
    avg_loss_res = total_loss_res / len(train_loader)

    return avg_loss, avg_loss_data, avg_loss_res

def validate(config, encoder, decoder, val_loader, batch_parser, device, scaler):

    encoder.eval()
    decoder.eval()

    total_rmse = 0.0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validating',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)

            B, C, W, H, L = data.shape

            if config.training.random_resolution:

                batch_coords, coords_outputs, batch_inputs = batch_parser.random_query(data)
            else:
                batch_coords, coords_outputs, batch_inputs = batch_parser.query_all(data)

            z = encoder(batch_inputs)

            u_pred = u_net(decoder, z, batch_coords)

            u_pred = u_pred.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
            coords_outputs = coords_outputs.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)

            rmse = RMSE(scaler.inverse(u_pred), scaler.inverse(coords_outputs))

            total_rmse += rmse.item()

    avg_rmse = total_rmse / len(val_loader)

    return avg_rmse

def test(config, test_loader, scaler, device):

    encoder = Encoder(config.model.encoder).to(device)
    decoder = Decoder(config.model.decoder).to(device)
    encoder, decoder = load_checkpoint(config, encoder, decoder, device)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L  = sample_batch.shape
    batch_parser = BatchParser(config, H, W, L, device)

    encoder.eval()
    decoder.eval()
    total_test_rmse = []
    vis_batch = 0
    save_vis_dir = PROJECT_DIR
    slice_dim = 'z'

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing')
        for batch_idx, data in progress_bar:

            data = data.to(device)

            B, C, W, H, L = data.shape

            slice_coords, slice_outputs, batch_inputs, slice_shape = batch_parser.query_slice(
                data, slice_dim = slice_dim, slice_idx=None)

            latent_z = encoder(batch_inputs)

            u_pred_slice = u_net(decoder, latent_z, slice_coords)

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

                    im1 = axes[i, 0].imshow(true_field[i, :, :], cmap='jet', origin='lower')
                    axes[i,0].set_title(f'{channel_names[i]} - real field')
                    axes[i,0].axis('off')
                    plt.colorbar(im1, ax=axes[i,0])

                    im2 = axes[i, 1].imshow(pred_field[i, :, :], cmap='jet', origin='lower')
                    axes[i,1].set_title(f'{channel_names[i]} - pred field')
                    axes[i,1].axis('off')
                    plt.colorbar(im2, ax=axes[i,1])

                    im3 = axes[i, 2].imshow(error_field[i, :, :], cmap='jet', origin='lower')
                    axes[i,2].set_title(f'{channel_names[i]} - error')
                    axes[i,2].axis('off')
                    plt.colorbar(im3, ax=axes[i,2])

                plt.tight_layout()
                vis_path = os.path.join(save_vis_dir, f"{config.model_name}_pde-{config.training.use_pde}.png")
                plt.savefig(vis_path, bbox_inches='tight', dpi=300, transparent=True)
                plt.close()

    print(f"l2 Relative Error: {np.mean(total_test_rmse):.6f}")

def train_and_evaluate(config, device):

    encoder = Encoder(config.model.encoder).to(device)
    decoder = Decoder(config.model.decoder).to(device)

    compute_total_params(encoder,decoder)

    optimizer, lr_scheduler = create_optimizer(config,encoder,decoder)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.train_batch_size)

    sample_batch = next(iter(train_loader))
    B, C, W, H, L  = sample_batch.shape
    batch_parser = BatchParser(config, H, W, L, device)

    job_name = f"{config.model_name}_pde-{config.training.use_pde}-{config.training.pde_r}_random-{config.training.random_resolution}_bs-{config.dataset.train_batch_size}"
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
    data_loss_list = []
    val_rmse_list = []
    phy_loss_list = []

    for epoch in range(max_steps):

        avg_train_loss, avg_train_loss_data, avg_train_loss_pde = train_epoch(
            encoder, decoder, train_loader,
            batch_parser, optimizer, config, device, scaler, epoch
        )

        val_rmse = validate(config,
            encoder, decoder, dev_loader,
            batch_parser, device, scaler
        )

        lr_scheduler.step()

        total_steps += 1
        epoch_list.append(total_steps)
        data_loss_list.append(avg_train_loss_data)
        phy_loss_list.append(avg_train_loss_pde)
        val_rmse_list.append(val_rmse)

        save_loss_to_csv(epoch_list, data_loss_list, val_rmse_list, phy_loss_list, exp_dir)
        plot_and_save_loss_curve(epoch_list, data_loss_list, val_rmse_list, phy_loss_list, exp_dir)

        if total_steps % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"step: {total_steps} | LR: {current_lr:.3e} | \
                  all loss: {avg_train_loss:.3e} | phy loss: {avg_train_loss_pde:.3e} |valide loss: {val_rmse:.3e}")

        if val_rmse < best_dev_rmse:
            best_dev_rmse = val_rmse
            save_checkpoint(ckpt_path, encoder, decoder, optimizer, total_steps)

