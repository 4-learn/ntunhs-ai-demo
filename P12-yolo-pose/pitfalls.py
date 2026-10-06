"""P12：AI 常寫錯的關鍵點程式，實際跑一次看結果。

    python P12-yolo-pose/pitfalls.py
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")
import numpy as np
import torch
from ultralytics import YOLO

from common import KPTS, MODELS, PHOTOS, keypoints, load_poses

torch.set_num_threads(4)
model = YOLO(str(MODELS / "yolo11n-pose.pt"))

print("== 1. 「[0] 就是那位住民」？第 0 人是框信心最高的人，不是固定的身分 ==")
for name in ["care-room", "care-corridor-fall"]:
    r = model(str(PHOTOS / f"{name}.png"), verbose=False)[0]
    kp = r.keypoints.xy.numpy()
    hip_y = kp[:, [11, 12], 1].mean(1)
    desc = ", ".join(f"第{i}人 框信心 {c:.3f} 髖 y={y:.0f}" for i, (c, y) in enumerate(zip(r.boxes.conf.tolist(), hip_y)))
    print(f"{name}：{desc}")

print("\n== 2. xyn 是 0–1，拿像素門檻去比會全錯 ==")
r = model(str(PHOTOS / "care-corridor-fall.png"), verbose=False)[0]
hip_px = float(r.keypoints.xy[0, [11, 12], 1].mean())
hip_n = float(r.keypoints.xyn[0, [11, 12], 1].mean())
print(f"倒地者髖部 y：xy={hip_px:.1f}（像素）、xyn={hip_n:.3f}（比例）")
print(f"AI 的判斷「髖部 y > 700 像素就是倒地」：用 xy → {hip_px > 700}；誤用 xyn → {hip_n > 700}")

print("\n== 3. 沒看信心：遺失的點座標是 (0, 0) ==")
df = load_poses()
raw = keypoints(df, mask_low=False)
lost = (raw[..., 2] < 0.1) & (raw[..., 0] == 0) & (raw[..., 1] == 0)
print(f"poses.csv 共 {raw.shape[0] * 17} 個點，座標 (0,0) 且信心 < 0.1 的有 {lost.sum()} 個（{lost.mean():.3f}）")
i = int(np.where(lost[:, 0])[0][0])  # 第一個鼻子遺失的格
row = df.iloc[[i]]
k_raw = keypoints(row, mask_low=False)[0]
k_mask = keypoints(row)[0]
print(f"{row.seq_id.item()} 第 {row.frame.item()} 格（{row.posture.item()}），鼻子 = ({k_raw[0, 0]:.0f}, {k_raw[0, 1]:.0f})，信心 {k_raw[0, 2]:.3f}")
h_raw = k_raw[:, 1].max() - k_raw[:, 1].min()
h_mask = np.nanmax(k_mask[:, 1]) - np.nanmin(k_mask[:, 1])
print(f"人的高度（y 最大－最小）：不看信心 {h_raw:.1f} 像素；把低信心點設 NaN 後 {h_mask:.1f} 像素")
