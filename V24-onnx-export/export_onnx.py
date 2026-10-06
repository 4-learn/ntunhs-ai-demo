"""V24：把兩個模型匯出成 ONNX，並用同一批輸入比對 PyTorch 與 onnxruntime 的輸出。

1. ResNet18 小格特徵抽取器（layer2＋layer3，與 common.resnet_patch_features 同設定）：torch.onnx.export
2. YOLO11n-pose：ultralytics model.export(format="onnx")

ONNX 檔存在 cache/onnx/（不進 repo）。第一次執行要匯出（YOLO 在共用教學機上要數分鐘），之後讀快取。
差異一律印「相對誤差的量級」（最大絕對差 ÷ 輸出的最大絕對值，再取 10 的次方上界），
因為最後幾位小數會因 CPU 與執行緒數而異。

    python V24-onnx-export/export_onnx.py
"""
import os
import sys

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")
import cv2
import numpy as np
import onnx
import torch

sys.path.insert(0, "V24-onnx-export")
from edge_lib import (OPSET, POSE_ONNX, R18_ONNX, export_pose, export_r18, mb, ort_patches, rel_order,  # noqa: E402
                      ort_session, preprocess, r18_patch_model)

from common import MODELS, PHOTOS, mvtec_paths, resnet_patch_features  # noqa: E402

torch.set_num_threads(4)


def io_desc(path):
    g = onnx.load(str(path)).graph
    dims = lambda v: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
    return ([(v.name, dims(v)) for v in g.input], [(v.name, dims(v)) for v in g.output])


print("== 1. ResNet18 小格特徵抽取器 → ONNX（torch.onnx.export）==")
export_r18()
ins, outs = io_desc(R18_ONNX)
print(f"檔案 r18-l23.onnx：{mb(R18_ONNX):.1f} MB，opset {onnx.load(str(R18_ONNX)).opset_import[0].version}")
print("輸入：", ins)
print("輸出：", outs)

print("\n== 2. 同一批輸入：PyTorch vs onnxruntime（bottle 測試圖前 8 張，224×224）==")
P = mvtec_paths("bottle")
x = preprocess(P["test"][:8])
model = r18_patch_model()
with torch.no_grad():
    y_pt = model(torch.from_numpy(x)).numpy().transpose(0, 2, 3, 1)
sess = ort_session(R18_ONNX)
y_ort = ort_patches(sess, x)
print("PyTorch 輸出", y_pt.shape, "｜onnxruntime 輸出", y_ort.shape)
print("onnxruntime vs PyTorch 相對誤差", rel_order(y_ort, y_pt))
y_cache = resnet_patch_features(P["test"])[:8]
print("common 快取（float16 存檔）vs PyTorch 相對誤差", rel_order(y_cache, y_pt))
print("np.allclose(onnx, pytorch, rtol=1e-3, atol=1e-4)：", np.allclose(y_ort, y_pt, rtol=1e-3, atol=1e-4))

print("\n== 3. 動態維度：同一個檔案吃不同 batch、不同解析度 ==")
for n, size in [(1, 224), (4, 224), (2, 160), (2, 320)]:
    out = ort_patches(sess, preprocess(P["test"][:n], size))
    print(f"batch {n}、{size}×{size} → 輸出 {out.shape}")

print("\n== 4. YOLO11n-pose → ONNX（ultralytics model.export）==")
export_pose()
ins, outs = io_desc(POSE_ONNX)
print(f"檔案 yolo11n-pose.onnx：{mb(POSE_ONNX):.1f} MB（.pt 是 {mb(MODELS / 'yolo11n-pose.pt'):.1f} MB），"
      f"opset {onnx.load(str(POSE_ONNX)).opset_import[0].version}")
print("輸入：", ins)
print("輸出：", outs)

print("\n== 5. 同一張照片：PyTorch vs onnxruntime（care-room.png）==")
from ultralytics import YOLO  # noqa: E402

img = cv2.imread(str(PHOTOS / "care-room.png"))
xp = np.ascontiguousarray(cv2.resize(img, (640, 640))[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
yolo_pt = YOLO(str(MODELS / "yolo11n-pose.pt"))
net = yolo_pt.model.fuse().eval()
with torch.no_grad():
    raw_pt = net(torch.from_numpy(xp))
raw_pt = (raw_pt[0] if isinstance(raw_pt, (list, tuple)) else raw_pt).numpy()
raw_ort = ort_session(POSE_ONNX).run(None, {"images": xp})[0]
print("原始輸出 (batch, 4 框＋1 信心＋17×3 點, 8400 個候選)：", raw_pt.shape, raw_ort.shape)
print("onnxruntime vs PyTorch 相對誤差", rel_order(raw_ort, raw_pt))
r_pt = yolo_pt.predict(str(PHOTOS / "care-room.png"), verbose=False)[0]
r_ort = YOLO(str(POSE_ONNX), task="pose").predict(str(PHOTOS / "care-room.png"), verbose=False)[0]
d = np.abs(r_pt.keypoints.xy.numpy() - r_ort.keypoints.xy.numpy()).max()
print(f"偵測人數：PyTorch {len(r_pt.boxes)}、ONNX {len(r_ort.boxes)}；關鍵點最大差 < 0.01 像素：{d < 0.01}")
