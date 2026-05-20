# export model to onnx
import torch
from deep_radar import RadarMultiTaskNet, RadarMultiTaskNetONNX

model = RadarMultiTaskNetONNX(heatmap_size=64)
model.eval()

dummy_input = torch.randn(1, 5, 40, 80)

torch.onnx.export(
    model,
    dummy_input,
    "radar_multitask_net.onnx",
    input_names=["receiver_signal"],
    output_names=["pred_heatmap", "pred_coord"],
    opset_version=17,
)