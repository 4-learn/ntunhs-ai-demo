"""V25：同一個模型，PyTorch vs onnxruntime；執行緒 1／2／4；batch 1；解析度 224 vs 160。

先檢查「結果一致」，再量延遲。延遲用 p50（中位數，毫秒），每種設定暖機 3 次、量 15 次。
毫秒數會因機器、同時在跑的程式而變（教學機是多人共用的伺服器），
所以講義只貼一致性與計算量的行；延遲表寫在教師紀錄，學員以自己電腦的結果為準。

    python V25-onnx-runtime/speed.py
"""
import os
import sys

os.environ.setdefault("YOLO_OFFLINE", "1")
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "V24-onnx-export")
from edge_lib import (POSE_ONNX, R18_ONNX, export_pose, export_r18, ort_patches, ort_session,  # noqa: E402
                      p50_ms, preprocess, r18_patch_model, rel_order)

from common import PHOTOS, mvtec_paths  # noqa: E402

THREADS = [1, 2, 4]
P = mvtec_paths("bottle")
x8 = preprocess(P["test"][:8])
model = r18_patch_model()
export_r18()
export_pose()

print("== 1. 先確認結果一致：執行緒數不同，輸出也要一樣 ==")
torch.set_num_threads(4)
with torch.no_grad():
    ref = model(torch.from_numpy(x8)).numpy().transpose(0, 2, 3, 1)
sessions = {t: ort_session(R18_ONNX, threads=t) for t in THREADS}
for t, s in sessions.items():
    out = ort_patches(s, x8)
    print(f"onnxruntime {t} 執行緒 vs PyTorch：相對誤差 {rel_order(out, ref)}，allclose {np.allclose(out, ref, rtol=1e-3, atol=1e-4)}")
a, b = preprocess(P["test"][:1], 224), preprocess(P["test"][:1], 160)
fa, fb = ort_patches(sessions[4], a), ort_patches(sessions[4], b)
print(f"224×224 → {fa.shape[1]}×{fa.shape[2]} 小格；160×160 → {fb.shape[1]}×{fb.shape[2]} 小格；"
      f"像素（≈計算量）比 {160 ** 2 / 224 ** 2:.3f}")

print("\n== 2. 延遲 p50（毫秒，batch 1；數字因機器而異，只看相對大小）==")
rows = []
for size in [224, 160]:
    x1 = preprocess(P["test"][:1], size)
    for t in THREADS:
        torch.set_num_threads(t)
        with torch.no_grad():
            pt = p50_ms(lambda: model(torch.from_numpy(x1)), n=10, warmup=2)
        ort = p50_ms(lambda: sessions[t].run(None, {"image": x1}), n=10, warmup=2)
        rows.append({"解析度": size, "執行緒": t, "PyTorch": round(pt), "onnxruntime": round(ort)})
torch.set_num_threads(4)
lat = pd.DataFrame(rows)
print(lat.to_string(index=False))

print("\n== 3. YOLO11n-pose 一張畫面（onnxruntime 4 執行緒，只算模型本體，不含前後處理）==")
img = cv2.imread(str(PHOTOS / "care-room.png"))
pose_sess = ort_session(POSE_ONNX, threads=4)
yrows = []
for size in [640, 320]:
    xi = np.ascontiguousarray(cv2.resize(img, (size, size))[:, :, ::-1].transpose(2, 0, 1)[None],
                              dtype=np.float32) / 255
    yrows.append({"輸入": size, "p50 毫秒": round(p50_ms(lambda: pose_sess.run(None, {"images": xi}), n=10, warmup=2))})
print(pd.DataFrame(yrows).to_string(index=False))

print("\n== 4. 本次執行的排名（多人共用的機器上，排名也可能換位；要重跑幾次才下結論）==")
lat["較快"] = np.where(lat["onnxruntime"] < lat["PyTorch"], "onnxruntime", "PyTorch")
print(lat[["解析度", "執行緒", "較快"]].to_string(index=False))
print(f"onnxruntime 較快的設定：{int((lat['較快'] == 'onnxruntime').sum())}/{len(lat)}")
