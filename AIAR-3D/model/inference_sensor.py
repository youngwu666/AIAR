import os
import torch
import numpy as np
from tqdm import tqdm
from models.EncoderDecoder import Encoder, Decoder
from models.flow import DiT
from utils.checkpoint_utils import load_checkpoint, load_dit_checkpoint, load_sensor_checkpoint
from data_utils import create_dataloader, BatchParser
from model.model_utils import compute_U_statistics, relative_l2_error, relative_l2_error_per_channel, sample_sensor_value, save_uncertainty
from model.train_flow import sample_ode, plot_ode_trajectory
from model.inference_recons import visualize_results, visualize_vector_field, plot_U_statistics, plot_3d_flow_simple, _scalar_range, make_error_dataset
from utils.model_utils import rand_sensor_indices
from utils.computational_efficiency import EfficiencyTracker, TimedModule, save_efficiency

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUM_SAMPLES = 10

def get_rand_sensor(data, sensor_numer, batch_parser, device, seed):

    B, C, W, H, L = data.shape
    indices = rand_sensor_indices(W, H, L, sensor_numer, device, seed)

    wid = indices // (H * L)
    row = (indices % (H * L)) // L
    col = (indices % (H * L)) % L

    sensor_value = data[:, :, wid, row, col].permute(0, 2, 1)

    sensor_pos = batch_parser.coords[indices].expand(B, -1, -1).to(device)

    return sensor_pos, sensor_value, wid, row, col

def get_models(config, sensor_numer, device):

    encoder = Encoder(config.fae.model.encoder).to(device)
    decoder = Decoder(config.fae.model.decoder).to(device)
    dit = DiT(config).to(device)

    dit.z_sens = torch.nn.Parameter(
        torch.zeros(sensor_numer, dtype=torch.float32).to(device), requires_grad=False)
    dit.x_sens = torch.nn.Parameter(
        torch.zeros(sensor_numer, dtype=torch.float32).to(device), requires_grad=False)
    dit.y_sens = torch.nn.Parameter(
        torch.zeros(sensor_numer, dtype=torch.float32).to(device), requires_grad=False)

    encoder, decoder = load_checkpoint(config.fae, encoder, decoder, device)
    dit = load_sensor_checkpoint(config, sensor_numer, dit, device)

    print(f"x_sens 是否全零: {(dit.x_sens == 0).all()}")

    encoder.eval()
    decoder.eval()
    dit.eval()
    return encoder, decoder, dit

def plot_sensor_move_demo(
        dataset,
        time_idx,
        rand_z, rand_y, rand_x,
        opt_z, opt_y, opt_x,
        save_path,
        show_interactive=True,
        num_arrows=8
    ):

    import pyvista as pv

    Ux = dataset[time_idx, 0, :, :, :]
    Uy = dataset[time_idx, 1, :, :, :]
    Uz = dataset[time_idx, 2, :, :, :]
    Nz, Ny, Nx = Ux.shape
    x = np.linspace(0, 3.0, Nx)
    y = np.linspace(0, 1.0, Ny)
    z = np.linspace(0, 1.0, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = pv.StructuredGrid(X, Y, Z)
    vel_mag = np.sqrt(Ux**2 + Uy**2 + Uz**2)
    scalar_ordered = vel_mag.transpose(2, 1, 0)

    def to_numpy(value):
        if hasattr(value, 'detach'):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    rand_z_np = to_numpy(rand_z).reshape(-1)
    rand_y_np = to_numpy(rand_y).reshape(-1)
    rand_x_np = to_numpy(rand_x).reshape(-1)
    opt_z_np = to_numpy(opt_z).reshape(-1)
    opt_y_np = to_numpy(opt_y).reshape(-1)
    opt_x_np = to_numpy(opt_x).reshape(-1)

    rand_idx = np.random.randint(len(rand_z_np))
    opt_idx = np.random.randint(len(opt_z_np))

    rand_points = np.column_stack((
        rand_x_np / (Nx - 1) * 3.0,
        rand_y_np / (Ny - 1) * 1.0,
        rand_z_np / (Nz - 1) * 1.0
    ))
    opt_points = np.column_stack((
        opt_x_np * 3.0,
        opt_y_np * 1.0,
        opt_z_np * 1.0
    ))

    start = np.array([
        rand_x_np[rand_idx] / (Nx - 1) * 3.0,
        rand_y_np[rand_idx] / (Ny - 1) * 1.0,
        rand_z_np[rand_idx] / (Nz - 1) * 1.0
    ], dtype=float)
    end = np.array([
        opt_x_np[opt_idx] * 3.0,
        opt_y_np[opt_idx] * 1.0,
        opt_z_np[opt_idx] * 1.0
    ], dtype=float)

    direction = end - start
    if np.linalg.norm(direction) < 1e-8:
        end = np.array([
            min(start[0] + 0.6, 3.0),
            min(start[1] + 0.2, 1.0),
            min(start[2] + 0.2, 1.0)
        ], dtype=float)
        direction = end - start

    plotter = pv.Plotter(off_screen=not show_interactive, window_size=(1200, 1000))
    vol = pv.ImageData(
        dimensions=(Nx, Ny, Nz),
        spacing=(3.0 / (Nx - 1), 1.0 / (Ny - 1), 1.0 / (Nz - 1)),
        origin=(0.0, 0.0, 0.0)
    )
    vol['scalar'] = scalar_ordered.flatten(order='F')
    plotter.add_volume(
        vol,
        scalars='scalar',
        cmap='jet',
        opacity='sigmoid',
        show_scalar_bar=True,
        scalar_bar_args={'title': ''}
    )
    plotter.add_mesh(grid.outline(), color='k', line_width=3)
    plotter.add_mesh(pv.PolyData(rand_points), color='blue', point_size=15, render_points_as_spheres=True)
    plotter.add_mesh(pv.PolyData(opt_points), color='red', point_size=15, render_points_as_spheres=True)

    arrow_count = max(int(num_arrows), 1)
    mid = (start + end) / 2.0
    curve_offset = np.array([
        -0.20 * direction[1],
        0.18 * direction[0],
        0.25
    ], dtype=float)
    control = np.clip(mid + curve_offset, [0.0, 0.0, 0.0], [3.0, 1.0, 1.0])
    path_points = []
    for i in range(arrow_count + 1):
        t = i / arrow_count
        point = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end
        wobble = np.array([
            0.0,
            0.08 * np.sin(t * np.pi * 3.0),
            0.06 * np.sin(t * np.pi * 5.0)
        ], dtype=float)
        path_points.append(np.clip(point + wobble, [0.0, 0.0, 0.0], [3.0, 1.0, 1.0]))

    for i in range(arrow_count):
        arrow_start = path_points[i]
        segment = path_points[i + 1] - path_points[i]
        if np.linalg.norm(segment) < 1e-8:
            continue
        arrow = pv.Arrow(
            start=arrow_start,
            direction=segment,
            tip_length=0.25,
            tip_radius=0.08,
            shaft_radius=0.035,
            scale=np.linalg.norm(segment) * 1.15
        )
        plotter.add_mesh(arrow, color='red')

    plotter.add_text('Sensor Movement Demo', position='upper_left', font_size=14)
    plotter.camera_position = [
        (5, 3, 5.0),
        (1.5, 0.5, 0.5),
        (0, 1, 0)
    ]
    plotter.screenshot(save_path, transparent_background=True)
    plotter.show()
    plotter.close()

def inference(config, device):

    sensor_numer = config.sensor.sensor_numer
    noise_level = config.sensor.noise_level
    num_ode_steps = config.sensor.num_ode_steps

    inference_dir = os.path.join(PROJECT_DIR, "inference")
    slice_dim = 'z'

    job_name = f"sensor_{sensor_numer}_noise_{noise_level}_ode{num_ode_steps}"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    encoder, decoder, dit = get_models(config, sensor_numer, device)

    efficiency = EfficiencyTracker(
        device,
        {"encoder": encoder, "decoder": decoder, "dit": dit},
    )
    decoder = TimedModule(decoder, efficiency)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    opt_errors, opt_ch_errprs = [], []
    rand_errors, rand_ch_errprs = [], []
    vis_sample_idx = 0

    channel_names = ['Ux', 'Uy', 'Uz', 'p', 'nut', 'U']

    efficiency.start()
    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            slice_coords, slice_outputs, batch_inputs, slice_shape = batch_parser.query_slice(
                data, slice_dim=slice_dim, slice_idx=None)

            z_u = encoder(data)
            z0 = torch.randn(z_u.shape, device=device)

            opt_z_sens = dit.z_sens[:sensor_numer].to(device)
            opt_x_sens = dit.x_sens[:sensor_numer].to(device)
            opt_y_sens = dit.y_sens[:sensor_numer].to(device)
            sensor_pos = torch.stack([opt_z_sens, opt_y_sens, opt_x_sens], dim=-1).expand(B, -1, -1).to(device)

            sensor_value = sample_sensor_value(
                data, opt_z_sens, opt_y_sens, opt_x_sens
            )

            noise = torch.randn_like(sensor_value, device=device)
            sensor_value = sensor_value + noise * noise_level

            sensor_pos_rand, sensor_value_rand,_,_,_ = get_rand_sensor(
                data, sensor_numer, batch_parser, device, config.seed
            )
            noise_rand = torch.randn_like(sensor_value_rand, device=device)
            sensor_value_rand = sensor_value_rand + noise_rand * noise_level

            z_pred_opt, _ = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_value,
                sensor_pos = sensor_pos,
                num_steps = num_ode_steps,
                use_conditioning = True
            )

            z_pred_rand, _ = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_value_rand,
                sensor_pos = sensor_pos_rand,
                num_steps = num_ode_steps,
                use_conditioning = True
            )

            Slice_data = slice_outputs.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)

            u_pred_slice_opt = decoder(z_pred_opt, slice_coords)
            u_pred_opt = u_pred_slice_opt.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
            u_true, u_pred_opt, _ = compute_U_statistics(
                "inference",
                scaler.inverse(Slice_data),
                scaler.inverse(u_pred_opt),
                scaler
            )
            errors_opt = relative_l2_error(u_pred_opt[:, :5], u_true[:, :5])
            opt_errors.extend(errors_opt.cpu().numpy())

            u_pred_slice_rand = decoder(z_pred_rand, slice_coords)
            u_pred_rand = u_pred_slice_rand.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
            _, u_pred_rand, _ = compute_U_statistics(
                "inference",
                scaler.inverse(Slice_data),
                scaler.inverse(u_pred_rand),
                scaler
            )
            errors_rand = relative_l2_error(u_pred_rand[:, :5], u_true[:, :5])
            rand_errors.extend(errors_rand.cpu().numpy())

            ch_errors_opt = relative_l2_error_per_channel(u_pred_opt, u_true)
            ch_errors_rand = relative_l2_error_per_channel(u_pred_rand, u_true)
            opt_ch_errprs.extend(ch_errors_opt.cpu().numpy())
            rand_ch_errprs.extend(ch_errors_rand.cpu().numpy())

            if batch_idx == vis_sample_idx:
                efficiency.pause()

                h, w = slice_shape
                u_slice = u_true.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_slice_opt = u_pred_opt.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_slice_rand = u_pred_rand.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)

                visualize_results(
                    noise_level = noise_level,
                    u_true = u_slice,
                    u_pred = u_slice_opt,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/opt_inference_result.png",
                    vis_idx = vis_sample_idx
                )

                visualize_vector_field(
                    u_true = u_slice,
                    u_pred = u_slice_opt,
                    save_path = f"{save_vis_dir}/opt_vector_field.png",
                    vis_idx = vis_sample_idx
                )

                visualize_results(
                    noise_level = noise_level,
                    u_true = u_slice,
                    u_pred = u_slice_rand,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/rand_inference_result.png",
                    vis_idx = vis_sample_idx
                )

                visualize_vector_field(
                    u_true = u_slice,
                    u_pred = u_slice_rand,
                    save_path = f"{save_vis_dir}/rand_vector_field.png",
                    vis_idx = vis_sample_idx
                )
                efficiency.resume()
    save_efficiency(
        os.path.join(save_vis_dir, "computer_efficiency.csv"),
        efficiency.finish(),
    )

    promotion = (np.mean(rand_errors) - np.mean(opt_errors)) / np.mean(rand_errors) * 100
    print(f"优化传感器误差 = {np.mean(opt_errors):.4f}")
    print(f"随机传感器误差 = {np.mean(rand_errors):.4f}")
    print(f"相对提升 = {promotion:.2f}%")

    for name, opt_err, rand_err in zip(['Ux', 'Uy', 'Uz', 'p','nut','U'],
                         np.mean(opt_ch_errprs, axis=0),
                         np.mean(rand_ch_errprs, axis=0)):
        print(f"{name:>8s}: {opt_err:.4f} ; {rand_err:.4f}")

    np.savetxt(os.path.join(save_vis_dir, 'error_summary.csv'),
           np.array(list(zip(['overall','Ux', 'Uy', 'Uz', 'p','nut','U'],
                             [np.mean(opt_errors)] + list(np.mean(opt_ch_errprs, axis=0)),
                             [np.mean(rand_errors)] + list(np.mean(rand_ch_errprs, axis=0))
                             )), dtype=object),
           delimiter=',', fmt=['%s', '%.6f', '%.6f'])

def inference_uncertainty(config, device, num_samples = NUM_SAMPLES):

    sensor_numer = config.sensor.sensor_numer
    noise_level = config.sensor.noise_level
    num_ode_steps = config.sensor.num_ode_steps

    encoder, decoder, dit = get_models(config, sensor_numer, device)

    inference_dir = os.path.join(PROJECT_DIR, "inference")
    slice_dim = 'z'

    job_name = f"sensor_{sensor_numer}_noise_{noise_level}_ode{num_ode_steps}_uncertainty-{num_samples}"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    opt_errors, opt_ch_errprs = [], []
    rand_errors, rand_ch_errprs = [], []
    vis_sample_idx = 0

    opt_sample_stacks = []

    channel_names = ['Ux', 'Uy', 'Uz', 'p', 'nut', 'U']

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            slice_coords, slice_outputs, batch_inputs, slice_shape = batch_parser.query_slice(
                data, slice_dim=slice_dim, slice_idx=None)

            z_u = encoder(batch_inputs)

            opt_z_sens = dit.z_sens[:sensor_numer].to(device)
            opt_x_sens = dit.x_sens[:sensor_numer].to(device)
            opt_y_sens = dit.y_sens[:sensor_numer].to(device)
            sensor_pos = torch.stack([opt_z_sens, opt_y_sens, opt_x_sens], dim=-1).expand(B, -1, -1).to(device)

            sensor_value = sample_sensor_value(
                data, opt_z_sens, opt_y_sens, opt_x_sens
            )

            noise = torch.randn_like(sensor_value, device=device)
            sensor_value = sensor_value + noise * noise_level

            sensor_pos_rand, sensor_value_rand,_,_,_ = get_rand_sensor(
                data, sensor_numer, batch_parser, device, config.seed
                )
            noise_rand = torch.randn_like(sensor_value_rand, device=device)
            sensor_value_rand = sensor_value_rand + noise_rand * noise_level

            opt_sample_collection, rand_sample_collection = [],[]
            for _ in range(num_samples):

                z0 = torch.randn(z_u.shape, device=device)

                z_pred_opt, _ = sample_ode(
                    dit = dit,
                    z0 = z0,
                    sensor_value = sensor_value,
                    sensor_pos = sensor_pos,
                    num_steps = num_ode_steps,
                    use_conditioning = True
                )

                z_pred_rand, _ = sample_ode(
                    dit = dit,
                    z0 = z0,
                    sensor_value = sensor_value_rand,
                    sensor_pos = sensor_pos_rand,
                    num_steps = num_ode_steps,
                    use_conditioning = True
                )

                u_pred_slice = decoder(z_pred_opt, slice_coords)
                u_pred = u_pred_slice.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
                opt_sample_collection.append(u_pred)

                u_pred_slice_rand = decoder(z_pred_rand, slice_coords)
                u_pred_rand = u_pred_slice_rand.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
                rand_sample_collection.append(u_pred_rand)

            stacked_opt = torch.stack(opt_sample_collection)
            opt_sample_stacks.append(stacked_opt)

            Slice_data = slice_outputs.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
            u_true, u_mean_opt, u_var_opt = compute_U_statistics(
                "uncertainty",
                Slice_data,
                stacked_opt,
                scaler
            )

            stacked_rand = torch.stack(rand_sample_collection)
            _, u_mean_rand, u_var_rand = compute_U_statistics(
                "uncertainty",
                Slice_data,
                stacked_rand,
                scaler
            )

            errors_opt = relative_l2_error(u_mean_opt[:, :5], u_true[:, :5])
            opt_errors.extend(errors_opt.cpu().numpy())

            errors_rand = relative_l2_error(u_mean_rand[:, :5], u_true[:, :5])
            rand_errors.extend(errors_rand.cpu().numpy())

            ch_errors_opt = relative_l2_error_per_channel(u_mean_opt, u_true)
            ch_errors_rand = relative_l2_error_per_channel(u_mean_rand, u_true)
            opt_ch_errprs.extend(ch_errors_opt.cpu().numpy())
            rand_ch_errprs.extend(ch_errors_rand.cpu().numpy())

            if batch_idx == vis_sample_idx:

                h, w = slice_shape
                u_slice = u_true.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_slice_opt = u_mean_opt.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_var_slice_opt = u_var_opt.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_slice_rand = u_mean_rand.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)
                u_var_slice_rand = u_var_rand.squeeze(2).squeeze(2).reshape(B, C + 1, h, w)

                visualize_results(
                    noise_level = noise_level,
                    u_true = u_slice,
                    u_pred = u_slice_opt,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/opt_inference_result.png",
                    vis_idx = vis_sample_idx,
                    u_var = u_var_slice_opt
                )

                plot_U_statistics(
                    u_true = u_slice,
                    u_mean = u_slice_opt,
                    u_var = u_var_slice_opt,
                    save_path = f"{save_vis_dir}/opt_U_statistics.png",
                    vis_idx = vis_sample_idx
                )

                visualize_results(
                    noise_level = noise_level,
                    u_true = u_slice,
                    u_pred = u_slice_rand,
                    channel_names = channel_names,
                    save_path = f"{save_vis_dir}/rand_inference_result.png",
                    vis_idx = vis_sample_idx,
                    u_var = u_var_slice_rand
                )

                plot_U_statistics(
                    u_true = u_slice,
                    u_mean = u_slice_rand,
                    u_var = u_var_slice_rand,
                    save_path = f"{save_vis_dir}/rand_U_statistics.png",
                    vis_idx = vis_sample_idx
                )

    promotion = (np.mean(rand_errors) - np.mean(opt_errors)) / np.mean(rand_errors) * 100
    print(f"优化传感器误差 = {np.mean(opt_errors):.4f}")
    print(f"随机传感器误差 = {np.mean(rand_errors):.4f}")
    print(f"相对提升 = {promotion:.2f}%")

    opt_ch_mean = np.mean(opt_ch_errprs, axis=0)
    opt_ch_std = np.std(opt_ch_errprs, axis=0, ddof=0)
    rand_ch_mean = np.mean(rand_ch_errprs, axis=0)

    for name, opt_err, rand_err in zip(['Ux', 'Uy', 'Uz', 'p','nut','U'],
                         opt_ch_mean,
                         rand_ch_mean):
        print(f"{name:>8s}: {opt_err:.4f} ; {rand_err:.4f}")

    np.savetxt(os.path.join(save_vis_dir, 'error_summary.csv'),
           np.array(list(zip(
                            ['overall','Ux', 'Uy', 'Uz', 'p','nut','U'],
                            [np.mean(opt_errors)] + list(opt_ch_mean),
                            [np.std(opt_errors, ddof=0)] + list(opt_ch_std),
                            [np.mean(rand_errors)] + list(rand_ch_mean),
                             )),
                    dtype=object),
           delimiter=',', fmt=['%s', '%.6f', '%.6f', '%.6f'])

    save_uncertainty(
        save_vis_dir,
        opt_sample_stacks,
        scaler,
        channel_names,
        csv_name='sample_opt_summary.csv',
    )

def inference_flow(config, device):

    sensor_numer = config.sensor.sensor_numer
    noise_level = config.sensor.noise_level
    num_ode_steps = config.sensor.num_ode_steps

    encoder, decoder, dit = get_models(config, sensor_numer, device)

    inference_dir = os.path.join(PROJECT_DIR, "inference")

    job_name = f"sensor_{sensor_numer}_noise_{noise_level}_ode{num_ode_steps}_flow"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    opt_errors = []
    rand_errors = []
    vis_sample_idx = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            z_u = encoder(data)

            opt_z_sens = dit.z_sens[:sensor_numer].to(device)
            opt_x_sens = dit.x_sens[:sensor_numer].to(device)
            opt_y_sens = dit.y_sens[:sensor_numer].to(device)
            sensor_pos = torch.stack([opt_z_sens, opt_y_sens, opt_x_sens], dim=-1).expand(B, -1, -1).to(device)

            sensor_value = sample_sensor_value(
                data, opt_z_sens, opt_y_sens, opt_x_sens
            )

            noise = torch.randn_like(sensor_value, device=device)
            sensor_value = sensor_value + noise * noise_level

            sensor_pos_rand, sensor_value_rand, wid_rand, row_rand, col_rand  = get_rand_sensor(
                data, sensor_numer, batch_parser, device, config.seed
                )
            noise_rand = torch.randn_like(sensor_value_rand, device=device)
            sensor_value_rand = sensor_value_rand + noise_rand * noise_level

            z0 = torch.randn(z_u.shape, device=device)
            z_pred_opt, _ = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_value,
                sensor_pos = sensor_pos,
                num_steps = num_ode_steps,
                use_conditioning = True
            )

            z_pred_rand, _ = sample_ode(
                dit = dit,
                z0 = z0,
                sensor_value = sensor_value_rand,
                sensor_pos = sensor_pos_rand,
                num_steps = num_ode_steps,
                use_conditioning = True
            )

            result_cache_path = os.path.join(save_vis_dir, "pred3d.pt")

            if os.path.exists(result_cache_path):

                print(f"Loading result cache............")

                rc = torch.load(result_cache_path, map_location='cpu', weights_only=False)

                u_pred_3d_opt = rc['u_pred_3d']
                u_true_3d = rc['u_true_3d']
                u_pred_3d_rand = rc['u_rand_3d']

            else:

                true_slices, pred_slices, rand_slices = [],[],[]
                for zi in range(W):

                    slice_coords, slice_outputs, _, _ = batch_parser.query_slice(
                        data, slice_dim='z', slice_idx=zi)

                    u_slice_opt = decoder(z_pred_opt, slice_coords)
                    u_slice_rand = decoder(z_pred_rand, slice_coords)

                    pred_slices.append(u_slice_opt)
                    true_slices.append(slice_outputs)
                    rand_slices.append(u_slice_rand)

                u_pred = torch.cat(pred_slices, dim=1)
                u_true = torch.cat(true_slices, dim=1)
                u_rand = torch.cat(rand_slices, dim=1)

                u_pred_3d_opt = scaler.inverse(u_pred.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)
                u_true_3d = scaler.inverse(u_true.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)
                u_pred_3d_rand = scaler.inverse(u_rand.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)

                torch.save({'u_pred_3d': u_pred_3d_opt.cpu(),
                            'u_true_3d': u_true_3d.cpu(),
                            'u_rand_3d': u_pred_3d_rand.cpu(),
                            }, result_cache_path)

            errors_opt = relative_l2_error(u_pred_3d_opt, u_true_3d)
            opt_errors.extend(errors_opt.cpu().numpy())

            errors_rand = relative_l2_error(u_pred_3d_rand, u_true_3d)
            rand_errors.extend(errors_rand.cpu().numpy())

            gt_np = u_true_3d.cpu().numpy()

            plot_phy = ['Velocity_Magnitude', 'Ux', 'Uy', 'Uz', 'p', 'nut']
            for variable in plot_phy:

                vmin, vmax = _scalar_range(gt_np, vis_sample_idx, variable)
                clim = (vmin, vmax)

                u_err_3d_opt = make_error_dataset(u_true_3d, u_pred_3d_opt, variable)
                u_err_3d_rand = make_error_dataset(u_true_3d, u_pred_3d_rand, variable)

                for label, dataset_3d in [('Opt_Ground_Truth', u_true_3d),
                                          ('Opt_Reconstruction', u_pred_3d_opt),
                                          ('Abs_opt_Error', u_err_3d_opt)]:

                    plot_3d_flow_simple(
                        dataset = dataset_3d.cpu().numpy(),
                        time_idx = vis_sample_idx,
                        variable = variable,
                        plot_type = 'vortex',
                        clim =  None if label=='Abs_opt_Error' else clim ,
                        title_label = label,
                        save_path = f"{save_vis_dir}/{label}_{variable}.png",
                        show_interactive = False
                    )

                for label, dataset_3d in [('Rand_Ground_Truth', u_true_3d),
                                          ('Rand_Reconstruction', u_pred_3d_rand),
                                          ('Abs_rand_Error', u_err_3d_rand)]:

                    plot_3d_flow_simple(
                        dataset = dataset_3d.cpu().numpy(),
                        time_idx = vis_sample_idx,
                        variable = variable,
                        plot_type = 'vortex',
                        clim =  None if label=='Abs_rand_Error' else clim ,
                        title_label = label,
                        save_path = f"{save_vis_dir}/{label}_{variable}.png",
                        show_interactive = False
                    )

            plot_3d_flow_simple(
                dataset = u_true_3d.cpu().numpy(),
                time_idx = vis_sample_idx,
                variable = 'Velocity_Magnitude',
                plot_type = 'sensor_only',
                title_label = 'Sensor Comparison (Blue=Random, Red=Optimized)',
                rand_z = wid_rand, rand_y = row_rand, rand_x = col_rand,
                opt_z = opt_z_sens, opt_y = opt_y_sens, opt_x = opt_x_sens,
                save_path = f"{save_vis_dir}/Sensor_Comparison.png",
                show_interactive = False
            )

            plot_sensor_move_demo(
                dataset = u_true_3d.cpu().numpy(),
                time_idx = vis_sample_idx,
                rand_z = wid_rand, rand_y = row_rand, rand_x = col_rand,
                opt_z = opt_z_sens, opt_y = opt_y_sens, opt_x = opt_x_sens,
                save_path = f"{save_vis_dir}/Sensor_Move_Demo.png",
                show_interactive = False
            )

            break

def inference_flow_uncertainty(config, device, num_samples = NUM_SAMPLES):

    sensor_numer = config.sensor.sensor_numer
    noise_level = config.sensor.noise_level
    num_ode_steps = config.sensor.num_ode_steps

    encoder, decoder, dit = get_models(config, sensor_numer, device)

    inference_dir = os.path.join(PROJECT_DIR, "inference")

    job_name = f"sensor_{sensor_numer}_noise_{noise_level}_ode{num_ode_steps}_flow_uncertainty"
    save_vis_dir = os.path.join(inference_dir, job_name)
    os.makedirs(save_vis_dir, exist_ok=True)

    train_loader, dev_loader, test_loader, scaler = create_dataloader(config, config.dataset.batch_size)

    sample_batch = next(iter(test_loader))
    B, C, W, H, L = sample_batch.shape
    batch_parser = BatchParser(config.fae, H, W, L, device)

    opt_errors = []
    rand_errors = []
    vis_sample_idx = 0

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inferencing',leave=False)
        for batch_idx, data in progress_bar:

            data = data.to(device)
            B, C, W, H, L = data.shape

            z_u = encoder(data)

            opt_z_sens = dit.z_sens[:sensor_numer].to(device)
            opt_x_sens = dit.x_sens[:sensor_numer].to(device)
            opt_y_sens = dit.y_sens[:sensor_numer].to(device)
            sensor_pos = torch.stack([opt_z_sens, opt_y_sens, opt_x_sens], dim=-1).expand(B, -1, -1).to(device)

            sensor_value = sample_sensor_value(
                data, opt_z_sens, opt_y_sens, opt_x_sens
            )

            noise = torch.randn_like(sensor_value, device=device)
            sensor_value = sensor_value + noise * noise_level

            sensor_pos_rand, sensor_value_rand, wid_rand, row_rand, col_rand  = get_rand_sensor(
                data, sensor_numer, batch_parser, device, config.seed
                )
            noise_rand = torch.randn_like(sensor_value_rand, device=device)
            sensor_value_rand = sensor_value_rand + noise_rand * noise_level

            result_cache_path = os.path.join(save_vis_dir, "pred3d_opt_uncertainty.pt")
            if os.path.exists(result_cache_path):

                print(f"Loading result cache: .........")

                rc = torch.load(result_cache_path, map_location='cpu', weights_only=False)

                u_pred_mean_opt = rc['u_pred_mean_opt']
                u_true_3d = rc['u_true_3d']
                u_pred_mean_rand = rc['u_pred_mean_rand']

            else:

                opt_3d_collection, rand_3d_collection = [],[]

                for _ in range(num_samples):

                    z0 = torch.randn(z_u.shape, device=device)

                    z_pred_opt, _ = sample_ode(
                        dit = dit,
                        z0 = z0,
                        sensor_value = sensor_value,
                        sensor_pos = sensor_pos,
                        num_steps = num_ode_steps,
                        use_conditioning = True
                    )

                    z_pred_rand, _ = sample_ode(
                        dit = dit,
                        z0 = z0,
                        sensor_value = sensor_value_rand,
                        sensor_pos = sensor_pos_rand,
                        num_steps = num_ode_steps,
                        use_conditioning = True
                    )

                    pred_slices, rand_slices, true_slices = [],[],[]
                    for zi in range(W):

                        slice_coords, slice_outputs, _, _ = batch_parser.query_slice(
                            data, slice_dim='z', slice_idx=zi
                        )

                        u_slice_opt = decoder(z_pred_opt, slice_coords)
                        u_slice_rand = decoder(z_pred_rand, slice_coords)

                        pred_slices.append(u_slice_opt)
                        rand_slices.append(u_slice_rand)
                        true_slices.append(slice_outputs)

                    u_pred_opt = torch.cat(pred_slices, dim=1)
                    u_pred_rand = torch.cat(rand_slices, dim=1)
                    u_true = torch.cat(true_slices, dim=1)

                    u_true_3d = scaler.inverse(u_true.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)
                    u_pred_3d_opt = scaler.inverse(u_pred_opt.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)
                    u_pred_3d_rand = scaler.inverse(u_pred_rand.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(3).reshape(B, C, W, H, L)

                    opt_3d_collection.append(u_pred_3d_opt)
                    rand_3d_collection.append(u_pred_3d_rand)

                stacked_opt = torch.stack(opt_3d_collection)
                stacked_rand = torch.stack(rand_3d_collection)

                u_pred_mean_opt = stacked_opt.mean(dim=0)
                u_pred_mean_rand = stacked_rand.mean(dim=0)

                torch.save({'u_pred_mean_opt': u_pred_mean_opt.cpu(),
                            'u_true_3d': u_true_3d.cpu(),
                            'u_pred_mean_rand': u_pred_mean_rand.cpu()
                            }, result_cache_path)

            errors_opt = relative_l2_error(u_pred_mean_opt, u_true_3d)
            opt_errors.extend(errors_opt.cpu().numpy())

            errors_rand = relative_l2_error(u_pred_mean_rand, u_true_3d)
            rand_errors.extend(errors_rand.cpu().numpy())

            gt_np = u_true_3d.cpu().numpy()

            plot_phy = ['Velocity_Magnitude', 'Ux', 'Uy', 'Uz', 'p', 'nut']
            for variable in plot_phy:

                vmin, vmax = _scalar_range(gt_np, vis_sample_idx, variable)
                clim = (vmin, vmax)

                u_err_3d_opt = make_error_dataset(u_true_3d, u_pred_3d_opt, variable)
                u_err_3d_rand = make_error_dataset(u_true_3d, u_pred_3d_rand, variable)

                for label, dataset_3d in [('Opt_Ground_Truth', u_true_3d),
                                          ('Opt_Reconstruction', u_pred_mean_opt),
                                          ('Abs_opt_Error', u_err_3d_opt)]:

                    plot_3d_flow_simple(
                        dataset = dataset_3d.cpu().numpy(),
                        time_idx = vis_sample_idx,
                        variable = variable,
                        plot_type = 'vortex',
                        clim =  None if label=='Abs_opt_Error' else clim ,
                        title_label = label,
                        save_path = f"{save_vis_dir}/{label}_{variable}.png",
                        show_interactive = False
                    )

                for label, dataset_3d in [('Rand_Ground_Truth', u_true_3d),
                                          ('Rand_Reconstruction', u_pred_mean_rand),
                                          ('Abs_rand_Error', u_err_3d_rand)]:

                    plot_3d_flow_simple(
                        dataset = dataset_3d.cpu().numpy(),
                        time_idx = vis_sample_idx,
                        variable = variable,
                        plot_type = 'vortex',
                        clim =  None if label=='Abs_rand_Error' else clim ,
                        title_label = label,
                        save_path = f"{save_vis_dir}/{label}_{variable}.png",
                        show_interactive = False
                    )

            plot_3d_flow_simple(
                dataset = u_true_3d.cpu().numpy(),
                time_idx = vis_sample_idx,
                variable = 'Velocity_Magnitude',
                plot_type = 'sensor_only',
                title_label = 'Sensor Comparison (Blue=Random, Red=Optimized)',
                rand_z = wid_rand, rand_y = row_rand, rand_x = col_rand,
                opt_z = opt_z_sens, opt_y = opt_y_sens, opt_x = opt_x_sens,
                save_path = f"{save_vis_dir}/Sensor_Comparison.png",
                show_interactive = False
            )

            break
