# V24｜匯出 ONNX

把 ResNet18 小格特徵抽取器（`torch.onnx.export`）與 YOLO11n-pose（ultralytics `model.export`）匯出成 ONNX（存到 `cache/onnx/`，不進 repo），用同一批輸入比對 PyTorch 與 onnxruntime；`edge_lib.py` 供 V25、V26、V28 import。

    python V24-onnx-export/export_onnx.py
    python V24-onnx-export/pitfalls.py
