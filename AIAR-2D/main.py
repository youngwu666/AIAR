import torch
import os
import numpy as np
import random
from easydict import EasyDict
from model import train_encoder
from model import train_flow
from model import inference_recons
from model import train_sensor
from model import inference_sensor
import yaml
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

PROJECT_DIR = Path(__file__).resolve().parent

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    with open(PROJECT_DIR / config_path, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return EasyDict(config_dict)

def main():
    train_model = "inference_sensor"

    train_fae = "configs/encoder.yml"
    train_dit = "configs/flow.yml"

    fae_config = load_config(train_fae)
    dit_config = load_config(train_dit)

    fae_config.dataset.data_path = str((PROJECT_DIR / fae_config.dataset.data_path).resolve())
    dit_config.dataset.data_path = str((PROJECT_DIR / dit_config.dataset.data_path).resolve())

    dit_config.fae = fae_config

    setup_seed(fae_config.seed)

    if train_model == "train_encoder":
        train_encoder.train_and_evaluate(fae_config, device)

    elif train_model == "train_flow":
        train_flow.train_and_evaluate(dit_config, device)

    elif train_model == "inference_recons":
        inference_recons.inference(dit_config, device)

    elif train_model == "train_sensor":
        train_sensor.train_and_evaluate(dit_config, device)

    elif train_model == "inference_sensor":
        inference_sensor.inference_uncertainty(dit_config, device)

if __name__ == "__main__":
    main()
