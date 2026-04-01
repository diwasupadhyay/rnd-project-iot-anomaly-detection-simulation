"""
Export CNN-LSTM model from PyTorch (.pth) to ONNX (.onnx).
Run locally once: python models/export_onnx.py
The ONNX model is used for deployment (30MB vs 200MB+ for PyTorch).
"""
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_lstm import CNNLSTM

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved", "best_model.pth")
ONNX_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved", "best_model.onnx")

def export():
    print("Loading PyTorch checkpoint...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    n_feat = ckpt.get("n_features", 38)
    n_cls  = ckpt.get("n_classes", 3)

    model = CNNLSTM(n_features=n_feat, n_classes=n_cls)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 10, n_feat)

    print(f"Exporting to ONNX (features={n_feat}, classes={n_cls})...")
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"✅ Exported: {ONNX_PATH} ({size_mb:.1f} MB)")
    print(f"   PyTorch .pth: {os.path.getsize(MODEL_PATH)/(1024*1024):.1f} MB")

if __name__ == "__main__":
    export()
