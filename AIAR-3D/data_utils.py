from einops import rearrange, repeat
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
_raw_data_cache = {}

def _load_raw_data(path):
    if path not in _raw_data_cache:
        dataload = np.load(path)
        _raw_data_cache[path] = dataload['data']
    return _raw_data_cache[path]

class BaseDataset(Dataset):

    def __init__(self, x, downsample_factor=1, num_samples=None):
        super().__init__()

        self.downsample_factor = downsample_factor
        self.num_samples = num_samples

        self.x = x[:, :, ::downsample_factor, ::downsample_factor, ::downsample_factor]

        self.x = torch.tensor(self.x, dtype=torch.float32)

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

class StdScaler(object):

    def __init__(self, mean, std):

        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, x):

        mean = self.mean.to(x.device)[:, None, None, None]
        std = self.std.to(x.device)[:, None, None, None]

        return (x - mean) / std

    def inverse(self, x):

        mean = self.mean.to(x.device)[:, None, None, None]
        std = self.std.to(x.device)[:, None, None, None]

        return x * std + mean

def create_dataset(config, path, process=None):

    data = _load_raw_data(path)
    Nt, C, Nz, Ny, Nx = data.shape
    print(f'原始数据形状: 时间步:{Nt}, 通道:{C}, 高:{Ny}, 宽:{Nz}, 长:{Nx}')

    data_mean, data_scale = np.mean(data, axis=(0,2,3,4)), np.std(data, axis=(0,2,3,4))
    print(f'分通道均值: {data_mean}, 分通道标准差: {data_scale}, 均值尺寸: {data_mean.shape}')

    train_ratio = 0.7
    dev_ratio = 0.1
    train_end = int(Nt * train_ratio)
    dev_end = train_end + int(Nt * dev_ratio)

    if process == 'train':
        data_slice = data[:train_end, ...]
        print(f'[Train] 选取时间步: 0 ~ {train_end-1}')

    elif process == 'dev':
        data_slice = data[train_end:dev_end, ...]
        print(f'[Dev]   选取时间步: {train_end} ~ {dev_end-1}')

    elif process == 'test':
        data_slice = data[dev_end:, ...]
        print(f'[Test]  选取时间步: {dev_end} ~ {Nt-1}')

    else:
        raise ValueError("please choose which dataset you are using (train, dev, or test)")

    print(f'单通道数据格式: {data.shape}')
    dataset = BaseDataset(data_slice, 1)

    return dataset, data_mean, data_scale

class FlowDataset(Dataset):

    def __init__(self, config, path, process, transform=False):

        self.data, self.mean, self.sd = create_dataset(config, path, process)

        if transform == 'std':

            self.transform = StdScaler(self.mean, self.sd)

        elif transform is None:
            self.transform = None

        else:
            raise ValueError("Invalid normalization method specified. Choose 'std' or 'maxmin'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_sample = self.data[idx]

        if self.transform:

            data_sample = self.transform(data_sample)

        return data_sample

def create_dataloader(config, batch_size, shuffle=True, drop_last=True):

    train_dataset = FlowDataset(config,
                                path=config.dataset.data_path,
                        process='train',
                        transform=config.dataset.transform
                        )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last
    )

    scaler = train_dataset.transform

    dev_dataset = FlowDataset(config,
                              path=config.dataset.data_path,
                        process='dev',
                        transform=config.dataset.transform
                        )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle = False,
        drop_last = False
    )

    test_dataset = FlowDataset(config,
                              path=config.dataset.data_path,
                        process='test',
                        transform=config.dataset.transform
                        )

    test_loader = DataLoader(
        test_dataset,
        batch_size = config.dataset.test_size,
        shuffle = False,
        drop_last = False
    )

    return train_loader, dev_loader, test_loader, scaler

class BatchParser:

    def __init__(self, config, H, W, L, device):

        self.config = config
        self.num_query_points = 4096
        self.device = device

        self.phy_z, self.phy_y, self.phy_x = config.model.encoder.phy_size

        z_star = torch.linspace(0, self.phy_z, W, device= device)
        y_star = torch.linspace(0, self.phy_y, H, device= device)
        x_star = torch.linspace(0, self.phy_x, L, device= device)

        z_grid, y_grid, x_grid = torch.meshgrid(z_star, y_star, x_star, indexing="ij")

        self.coords = torch.cat([
            z_grid.flatten().unsqueeze(-1),
            y_grid.flatten().unsqueeze(-1),
            x_grid.flatten().unsqueeze(-1)
        ], dim = -1)

        self.coords_3d = torch.stack([z_grid, y_grid, x_grid], dim=-1)

    def random_query(self, batch):

        batch_inputs = batch

        batch_outputs = rearrange(batch_inputs, "b c w h l-> b (w h l) c")

        query_index = torch.randperm(batch_outputs.shape[1])[:self.num_query_points]

        batch_coords = self.coords[query_index]
        batch_coords = batch_coords.expand(batch.shape[0], -1, -1)

        coords_outputs = batch_outputs[:, query_index]

        return batch_coords, coords_outputs, batch_inputs

    def query_all(self, batch):

        batch_inputs = batch

        coords_outputs = rearrange(batch_inputs, "b c w h l-> b (w h l) c")

        batch_coords = self.coords
        batch_coords = batch_coords.expand(batch.shape[0], -1, -1)

        return batch_coords, coords_outputs, batch_inputs

    def query_slice(self, batch, slice_dim='z', slice_idx=None):
        B, C, W, H, L = batch.shape

        batch_inputs = batch

        if slice_idx is None:
            if slice_dim == 'z': slice_idx = W // 2
            elif slice_dim == 'y': slice_idx = H // 2
            elif slice_dim == 'x': slice_idx = L // 2

        if slice_dim == 'z':

            slice_coords_3d = self.coords_3d[slice_idx, :, :, :]

            slice_coords = slice_coords_3d.reshape(-1, 3)

            slice_outputs = batch[:, :, slice_idx, :, :].permute(0, 2, 3, 1).reshape(B, -1, C)

            slice_shape = (H, L)

        elif slice_dim == 'y':

            slice_coords_3d = self.coords_3d[:, slice_idx, :, :]
            slice_coords = slice_coords_3d.reshape(-1, 3)
            slice_outputs = batch[:, :, :, slice_idx, :].permute(0, 2, 3, 1).reshape(B, -1, C)

            slice_shape = (W, L)

        elif slice_dim == 'x':

            slice_coords_3d = self.coords_3d[:, :, slice_idx, :]
            slice_coords = slice_coords_3d.reshape(-1, 3)
            slice_outputs = batch[:, :, :, :, slice_idx].permute(0, 2, 3, 1).reshape(B, -1, C)

            slice_shape = (W, H)

        slice_coords = slice_coords.expand(B, -1, -1)

        return slice_coords, slice_outputs, batch_inputs, slice_shape

