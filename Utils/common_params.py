@staticmethod
def getCommonParams():
    import torch
    common_params = type('', (), {})()  # empty object to hold parameters
    common_params.a = 1e13
    common_params.fs = 5e7
    common_params.M = 40
    common_params.n_fft = 1024
    common_params.rx_radius = 100 # m
    theta = 2 * torch.pi * torch.arange(common_params.M, dtype=torch.float32) / common_params.M
    rx_pos_tensor = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros(common_params.M)], dim=1) * common_params.rx_radius

    common_params.tx_pos = torch.zeros(3, dtype=torch.float32)  # transmitter at origin
    common_params.rx_pos = rx_pos_tensor

    return common_params