import os
import torch
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

def save_checkpoint(ckpt_path, encoder, decoder, optimizer, step, filename='ckpt_step_final.pth'):
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }, os.path.join(ckpt_path, filename))

def load_checkpoint(config, encoder, decoder, device, filename='ckpt_step_final.pth'):

    fae_job_name = f"{config.model_name}_pde-{config.training.use_pde}-{config.training.pde_r}_random-{config.training.random_resolution}_bs-{config.dataset.train_batch_size}"
    fae_ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", fae_job_name, "ckpt")
    load_path = os.path.join(fae_ckpt_path, filename)

    checkpoint = torch.load(load_path, map_location=device)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    return encoder, decoder

def save_dit_checkpoint(ckpt_path, dit, optimizer, step, filename='dit_final.pth'):
    torch.save({
        'dit': dit.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }, os.path.join(ckpt_path, filename))

def load_dit_checkpoint(config, dit, device, filename='dit_final.pth'):

    dit_job_name = f"{config.model_name}_pde_{config.fae.training.use_pde}_ode_{config.training.num_ode_steps}"
    dit_ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", dit_job_name, "ckpt")
    load_path = os.path.join(dit_ckpt_path, filename)

    checkpoint = torch.load(load_path, map_location=device)

    dit.load_state_dict(checkpoint['dit'])

    return dit

def save_sensor_checkpoint(ckpt_path, sensor_numer, dit, optimizer, step, filename='sensor_final.pth'):
    torch.save({
        'sensor': dit.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        "opt_x_sens": dit.x_sens,
        "opt_y_sens": dit.y_sens,
        "opt_z_sens": dit.z_sens,
        "sensor_numer": sensor_numer
    }, os.path.join(ckpt_path, filename))

def load_sensor_checkpoint(config, sensor_numer, dit, device, filename='sensor_final.pth'):

    dit_job_name = f"Sensor_{sensor_numer}"
    dit_ckpt_path = os.path.join(PROJECT_DIR, "checkpoints", dit_job_name, "ckpt")
    load_path = os.path.join(dit_ckpt_path, filename)

    checkpoint = torch.load(load_path, map_location=device)

    dit.load_state_dict(checkpoint['sensor'])

    return dit
