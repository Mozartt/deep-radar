import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
from model.phase_net import PhaseOnlyNet
from model.delay_net import DelayNet
import torch
from data_loaders.my_dataloader import RadarMatDataset
from torch.utils.data import DataLoader

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
    dataset = RadarMatDataset(root_dir="D:\\radar-dataset\\train")
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
    
    mask = (complex_signal.detach() != 0)
    col_indices = torch.arange(complex_signal.shape[1], dtype=torch.float32)
    masked_indices = torch.where(mask, col_indices, torch.nan)
    ref_idx = torch.round(torch.nanmean(masked_indices, dim=1, keepdim=True))
    
    t_ref = ref_idx * Ts # nTs
    # 1000 samples of pure beat n T2
    beta_exp = torch.exp(1j * 2 * torch.pi * a * tau_pred[:, :, None] * (Ts * torch.arange(1000) - t_ref))  # [1, M, N]
    beta_exp = beta_exp.squeeze(0)  # [M, N]
    # remvoe beat freq from input signal
    
    phase_only = complex_signal * beta_exp

    # phase estimator
    
    mean_gamma = torch.sum(phase_only)  # [M] average over N' to get per-receiver phase estimate
    non_zero_cnt = torch.sum(phase_only > 0, dim=1)  # [M] count of non-zero samples per receiver
    mean_gamma /= non_zero_cnt  # [M] average over non-zero samples per receiver
    mean_gamma *= torch.exp(1j * 2 * torch.pi * a * tau_pred * t_ref)  # [M] remove the effect of delay from the phase estimate
    phi_hat = torch.angle(mean_gamma)  # [M] phase estimate in radians
    cos_pred = torch.cos(phi_hat)  # [M]
    sin_pred = torch.sin(phi_hat)  # [M]
    cos_phi_0, sin_phi_0 = cos_pred[0], sin_pred[0]              # [B]
    cos_delta = cos_pred * cos_phi_0 + sin_pred * sin_phi_0  # [B, M]
    sin_delta = sin_pred * cos_phi_0 - cos_pred * sin_phi_0  # [B, M]

    cos_gt = torch.cos(phi)  # [M]
    sin_gt = torch.sin(phi)  # [M]
    cos_phi_0_gt, sin_phi_0_gt = cos_gt[0], sin_gt[0]              # [B]
    cos_delta_gt = cos_gt * cos_phi_0_gt + sin_gt * sin_phi_0_gt  # [B, M]
    sin_delta_gt = sin_gt * cos_phi_0_gt - cos_gt * sin_phi_0_gt  # [B, M]
    a=5