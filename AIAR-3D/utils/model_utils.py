import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import os
import matplotlib.pyplot as plt

def rand_sensor_indices(depth: int, height: int, width: int, sensor_number: int, device, seed: int, batch_idx: int = 0) -> torch.Tensor:

    total = depth * height * width
    if sensor_number > total:
        raise ValueError(f"sensor_number={sensor_number} exceeds grid size {total}.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(batch_idx))
    return torch.randperm(total, generator=generator)[:sensor_number].to(device)

def create_optimizer(config, *models):

    all_params = []
    for model in models:
        all_params += list(model.parameters())

    optimizer = AdamW(
        all_params,
        lr = config.lr.peak_value,
        betas = (config.optim.beta1, config.optim.beta2),
        eps = config.optim.eps,
        weight_decay = config.optim.weight_decay
    )

    def lr_lambda(step):
        if step < config.lr.warmup_steps:
            return step / config.lr.warmup_steps
        else:
            return 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = config.training.max_steps - config.lr.warmup_steps,
        eta_min = config.lr.min_value
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.lr.warmup_steps]
    )

    return optimizer, lr_scheduler

def compute_total_params(*models):

    total_params = 0
    for model in models:

        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params += model_params

    if total_params >= 1_000_000_000:
        print(f"总参数量: {total_params / 1_000_000_000:.2f} 十亿")
    else:
        print(f"总参数量: {total_params / 1_000_000:.2f} 百万")
    return total_params

def save_loss_to_csv(epoch_list, train_loss_list, val_rmse_list, phy_loss_list, exp_dir):
    loss_df = pd.DataFrame({
        'epoch': epoch_list,
        'train_loss': train_loss_list,
        'phy_loss': phy_loss_list,
        'val_rmse': val_rmse_list
    })
    csv_path = os.path.join(exp_dir, 'loss_log.csv')
    loss_df.to_csv(csv_path, index=False)

def plot_and_save_loss_curve(epoch_list, data_loss_list, val_rmse_list, phy_loss_list, exp_dir):

    plt.figure(figsize=(12, 5))
    if phy_loss_list:
        ax1 = plt.subplot(1, 2, 1)
        ax2 = ax1.twinx()
        ax1.plot(epoch_list, data_loss_list, label='Data Loss', color='#1f77b4', linewidth=2)
        ax2.plot(epoch_list, phy_loss_list, label='phy Loss', color="#1fb44e", linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Data Loss', fontsize=12)
        ax2.set_ylabel('Phy Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14)
        ax1.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)

    else:
        plt.subplot(1, 2, 1)
        plt.plot(epoch_list, data_loss_list, label='Data Loss', color='#1f77b4', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Data Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14)
        plt.legend(fontsize=10)

    plt.grid(True, alpha=0.3)

    if val_rmse_list:

        plt.subplot(1, 2, 2)
        plt.plot(epoch_list, val_rmse_list, label='Val RMSE', color='#ff7f0e', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('Validation RMSE Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(exp_dir, 'loss_curve.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()