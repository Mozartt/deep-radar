import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
from model.phase_net import PhaseOnlyNet
from model.delay_net import DelayNet
import torch
from data_loaders.my_dataloader import RadarMatDataset
from torch.utils.data import DataLoader
import numpy as np

def compute_tau_stats(dataset):
    """Compute per-feature mean and std of tau over the full training set."""
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_tau = []
    for signal, heatmap, coord, tau, phi in loader:
        all_tau.append(tau.float())
    all_tau = torch.cat(all_tau, dim=0)  # [N, M]
    tau_mean = all_tau.mean(dim=0)
    tau_std = all_tau.std(dim=0)
    # Avoid exploding normalized error for near-constant receivers.
    std_floor = torch.clamp(tau_std.mean() * 0.1, min=1e-6)
    tau_std = tau_std.clamp(min=std_floor)
    return tau_mean, tau_std

activation = {}

def get_activation(name):
    def hook(model, input, output):
        # .detach() prevents tracking history for gradients
        # .cpu() ensures it can be converted to a NumPy array for plotting
        activation[name] = output.detach().cpu()
        activation["input"] = input[0].detach().cpu()
    return hook

if __name__ == '__main__':
    dataset = RadarMatDataset(root_dir="D:\\radar-dataset\\test")
    signal, heatmap, coord, tau, phi = dataset[0]

    ckpt = torch.load("delay_net.pt", map_location="cpu", weights_only=True)

    model = DelayNet(M=40).to("cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    #model.encoder.register_forward_hook(get_activation("encoder"))


    with torch.no_grad():
        pred = model(signal.unsqueeze(0).float())

    #feature_map = activation["encoder"]  # [C, M, N'] C=128, M=40, N'=128
    #input = activation["input"]  # [2, M, N] = [2, 40, 1024]

    # plot first channel first reciever M=1 
    #import matplotlib.pyplot as plt

    # plt.figure()
    # plt.imshow(feature_map[0,0, :, :].numpy().reshape(40,128), aspect='auto')

    #plt.figure()
    #plt.plot(input[0,0, 0, :].numpy().reshape(1,512).squeeze())
    tau_pred_norm = model(signal.unsqueeze(0).float()) # unsqeeze to add batch dimension → [1, M]
    tau_mean, tau_std = compute_tau_stats(dataset)
    tau_pred = tau_pred_norm * tau_std + tau_mean  # denormalize to

    # exp( 1i * 2 * pi * a * tau' * Ts * n)
    a = 1e13 # chirp slope 
    Fs = 5e7
    Ts = 1 / Fs
    complex_signal = torch.complex(signal[0], signal[1])  # [M, N]

    # 1000 samples of pure beat n T2
    beta_exp = torch.exp(1j * 2 * torch.pi * a * tau_pred[:, :, None] * Ts * torch.arange(1000))  # [1, M, N]
    beta_exp = beta_exp.squeeze(0)  # [M, N]
    # remvoe beat freq from input signal
    
    phase_only = complex_signal * beta_exp

    # keep only non-zero sample for phase_only
    first_nz_idx = torch.zeros(phase_only.shape[0])
    for m in range(phase_only.shape[0]):
        non_zero_idx = torch.nonzero(phase_only[m], as_tuple=True)[0]
        first_nz_idx[m] = non_zero_idx[0]
    max_nz_idx = int(first_nz_idx.max().item())
    phase_only_nz = phase_only[:, max_nz_idx:]  # [M, N'] N' <= 1000

    theta = torch.angle(phase_only_nz)  # [B, M, N]
    theta_np = theta.detach().cpu().numpy()
    theta_unwrapped_np = np.unwrap(theta_np, axis=-1)

    theta_unwrapped = torch.tensor(
        theta_unwrapped_np, dtype=torch.float32, device=phase_only.device
    )  # [B, M, N]

    w = torch.ones_like(theta_unwrapped)
    eps = 1e-12
    N=1000
    w_sum = w.sum(dim=-1, keepdim=True) + eps
    n = torch.arange(N)
    n_view = n.view(1, 1, N)
    n_bar = (n_view).sum(dim=-1, keepdim=True) / w_sum
    theta_bar = (theta_unwrapped).sum(dim=-1, keepdim=True) / w_sum
    n_centered = n_view - n_bar
    theta_centered = theta_unwrapped - theta_bar
    numerator = (n_centered * theta_centered).sum(dim=-1)
    denominator = (n_centered ** 2).sum(dim=-1) + eps

    omega_hat = numerator / denominator  # [B, M], rad/sample
    delta_tau_hat = omega_hat / (2 * torch.pi * a * Ts)

    tau_refined = tau_pred - delta_tau_hat
    beta_exp = torch.exp(1j * 2 * torch.pi * a * tau_refined[:, None] * Ts * torch.arange(1000))  # [1, M, N]
    beta_exp = beta_exp.squeeze(0)  # [M, N]
    # remvoe beat freq from input signal
    
    phase_only = complex_signal * beta_exp
    # phase estimator
    
    # mean_gamma = torch.sum(phase_only)  # [M] average over N' to get per-receiver phase estimate
    # non_zero_cnt = torch.sum(phase_only > 0, dim=1)  # [M] count of non-zero samples per receiver
    # mean_gamma /= non_zero_cnt  # [M] average over non-zero samples per receiver
    # mean_gamma *= torch.exp(1j * 2 * torch.pi * a * tau_pred * t_ref)  # [M] remove the effect of delay from the phase estimate
    # phi_hat = torch.angle(mean_gamma)  # [M] phase estimate in radians
    # cos_pred = torch.cos(phi_hat)  # [M]
    # sin_pred = torch.sin(phi_hat)  # [M]
    # cos_phi_0, sin_phi_0 = cos_pred[0], sin_pred[0]              # [B]
    # cos_delta = cos_pred * cos_phi_0 + sin_pred * sin_phi_0  # [B, M]
    # sin_delta = sin_pred * cos_phi_0 - cos_pred * sin_phi_0  # [B, M]

    cos_gt = torch.cos(phi)  # [M]
    sin_gt = torch.sin(phi)  # [M]
    cos_phi_0_gt, sin_phi_0_gt = cos_gt[0], sin_gt[0]              # [B]
    cos_delta_gt = cos_gt * cos_phi_0_gt + sin_gt * sin_phi_0_gt  # [B, M]
    sin_delta_gt = sin_gt * cos_phi_0_gt - cos_gt * sin_phi_0_gt  # [B, M]
    a=5