import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import random
from models.EncoderDecoder import Encoder, Decoder
from models.flow import DiT
from utils.checkpoint_utils import load_checkpoint, load_dit_checkpoint, save_sensor_checkpoint,load_sensor_checkpoint
from data_utils import create_dataloader, BatchParser
from model.model_utils import relative_l2_error, RMSE, MSE, sample_sensor_value
from model.train_flow import sample_ode, plot_ode_trajectory, get_diffusion_batch
from utils.model_utils import create_optimizer, compute_total_params, save_loss_to_csv, plot_and_save_loss_curve

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_v(config, encoder, decoder, dit,
                    dataloader, scaler,
                    optimizer, device,
                    batch_parser, num_ode_steps,
                    latent_bank=None, local_bases=None):

    dit.eval()
    encoder.eval()
    decoder.eval()

    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training Epoch',leave=False)
    for batch_idx, data in progress_bar:

        data = data.to(device)

        B, C, W, H, L = data.shape

        optimizer.zero_grad()

        z_sens = dit.z_sens
        x_sens = dit.x_sens
        y_sens = dit.y_sens

        z_u = encoder(data)

        sensor_pos = torch.stack([z_sens, y_sens, x_sens], dim = -1).expand(B, -1, -1).to(device)

        sensor_value = decoder(z_u, sensor_pos)

        K = 4
        loss_sum = 0.0
        for _ in range(K):
            z_t, t, target = get_diffusion_batch(z_u)
            pred_v = dit(z_t, t, sensor_value, sensor_pos)

            residual = (pred_v - target).reshape(B, -1)

            query = z_t.reshape(B, -1).detach().cpu()
            distances = torch.cdist(query, latent_bank)
            nearest_indices = torch.argmin(distances, dim=1)
            basis = local_bases[nearest_indices].to(device)

            coeff = torch.einsum("bd,brd->br", residual, basis)
            tangent = torch.einsum("br,brd->bd", coeff, basis)
            normal = residual - tangent
            loss_sum += tangent.square().mean() + 0.15 * normal.square().mean()
        loss = loss_sum / K

        loss.backward()

        torch.nn.utils.clip_grad_norm_([dit.z_sens, dit.x_sens, dit.y_sens], 1.0)

        optimizer.step()

        dit.z_sens.data.clamp_(0, 1)
        dit.x_sens.data.clamp_(0, 1)
        dit.y_sens.data.clamp_(0, 1)

        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)

def train_and_evaluate(config, device):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_dit_checkpoint(config, dit, device)

    for param in encoder.parameters(): param.requires_grad = False
    for param in decoder.parameters(): param.requires_grad = False
    for param in dit.parameters(): param.requires_grad = False

    encoder.eval()
    decoder.eval()
    dit.eval()

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.sensor.batch_size)

    sample_batch = next(iter(train_loader))

    B, C, W, H, L = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L,device)

    sensor_numer = config.sensor.sensor_numer
    n_epochs = config.sensor.n_epochs
    lr = config.sensor.lr
    num_ode_steps = config.sensor.num_ode_steps
    log_interval = config.logging.log_interval

    latent_bank, local_bases = None, None

    latent_list = []
    sample_count = 0
    with torch.no_grad():
        for data in tqdm(train_loader, desc="Building latent bank", leave=False):
            data = data.to(device)
            z_u = encoder(data)
            latent_list.append(z_u.reshape(z_u.shape[0], -1).detach().cpu())
            sample_count += data.shape[0]
            if sample_count >= 256:
                break

    latent_bank = torch.cat(latent_list, dim=0)[:256].contiguous()
    local_k = min(64, latent_bank.shape[0])
    distances = torch.cdist(latent_bank, latent_bank)
    _, neighbor_indices = torch.topk(distances, k=local_k, largest=False)

    basis_list = []
    for i in tqdm(range(latent_bank.shape[0]), desc="Precomputing local PCA", leave=False):
        local_samples = latent_bank[neighbor_indices[i]]
        local_samples = local_samples - local_samples.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(local_samples, full_matrices=False)
        basis_list.append(vh[:min(16, vh.shape[0])].contiguous())
    local_bases = torch.stack(basis_list, dim=0).contiguous()

    dit.z_sens = torch.nn.Parameter(
        torch.rand(sensor_numer, device=device),
        requires_grad=True
        )
    dit.x_sens = torch.nn.Parameter(
        torch.rand(sensor_numer, device=device),
        requires_grad=True
        )
    dit.y_sens = torch.nn.Parameter(
        torch.rand(sensor_numer, device=device),
        requires_grad=True
        )

    optimizer = torch.optim.Adam(
        [dit.x_sens, dit.y_sens, dit.z_sens],
        lr = lr,
        weight_decay = 1e-4
        )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = n_epochs,
        eta_min = 0
        )

    job_name = f"Sensor_{sensor_numer}"
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    exp_dir = os.path.join(base_dir, job_name)
    ckpt_path = os.path.join(exp_dir, "ckpt")
    os.makedirs(ckpt_path, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=4)

    total_steps = 0
    epoch_list = []
    train_loss_list = []
    best_dev_loss = float('inf')
    for epoch in range(n_epochs):

        avg_loss = train_v(
            config, encoder, decoder, dit,
            train_loader, scaler,
            optimizer, device,
            batch_parser, num_ode_steps,
            latent_bank, local_bases
        )

        lr_scheduler.step()

        total_steps += 1
        epoch_list.append(total_steps)
        train_loss_list.append(avg_loss)

        save_loss_to_csv(epoch_list, train_loss_list, None, None, exp_dir)
        plot_and_save_loss_curve(epoch_list, train_loss_list, None, None, exp_dir)

        if total_steps % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"step: {total_steps} | LR: {current_lr:.3e} | train loss: {avg_loss:.3e}")

        if total_steps % n_epochs == 0:
            save_sensor_checkpoint(ckpt_path, sensor_numer, dit, optimizer, total_steps)

