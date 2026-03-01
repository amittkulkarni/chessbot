import os
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.model.resnet import ChessResNet

def export_and_quantize(pth_path: str, onnx_out_path: str):
    """
    Exports PyTorch model to ONNX and applies INT8 dynamic quantization.
    """
    if os.path.exists(onnx_out_path):
        os.remove(onnx_out_path)

    print(f"Loading PyTorch weights from {pth_path}...")
    model = ChessResNet()

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Weights file not found: {pth_path}")

    state_dict = torch.load(pth_path, map_location='cpu')
    # Handle DataParallel prefix if present
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    dummy_input = torch.randn(1, 18, 8, 8)
    temp_onnx = "temp_fp32.onnx"

    print("Exporting to ONNX (Float32)...")
    torch.onnx.export(
        model, dummy_input, temp_onnx,
        export_params=True, opset_version=11,
        do_constant_folding=False,
        input_names=['input'], output_names=['policy', 'value'],
        dynamic_axes={'input': {0: 'batch'}, 'policy': {0: 'batch'}, 'value': {0: 'batch'}}
    )

    print("Applying INT8 Dynamic Quantization...")
    quantize_dynamic(temp_onnx, onnx_out_path, weight_type=QuantType.QUInt8)

    if os.path.exists(temp_onnx):
        os.remove(temp_onnx)

    print(f"Engine Ready: {onnx_out_path}")