"""P12：用 YOLO-pose 對三張照片推論，拆開 keypoints.xy／.conf，對應到 poses.csv 的欄位格式。

    python P12-yolo-pose/pose_infer.py

標註圖存到 /tmp/ntunhs/P12/。
"""
import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from common import KPTS, MIN_KPT_CONF, MODELS, PHOTOS, load_poses

torch.set_num_threads(4)
OUT = Path("/tmp/ntunhs/P12")
OUT.mkdir(parents=True, exist_ok=True)
PHOTO_NAMES = ["care-room", "care-corridor-fall", "factory-lifting"]

model = YOLO(str(MODELS / "yolo11n-pose.pt"))
print("== 1. 模型 ==")
print(f"task={model.task}；類別={model.names}；關鍵點數 {model.model.kpt_shape}")

print("\n== 2. 三張照片的推論結果 ==")
rows, results = [], {}
for name in PHOTO_NAMES:
    r = model(str(PHOTOS / f"{name}.png"), verbose=False)[0]
    results[name] = r
    r.save(filename=str(OUT / f"{name}.jpg"))
    h, w = r.orig_shape
    rows.append(dict(照片=name, 寬高=f"{w}x{h}", 人數=len(r.boxes),
                     框信心=", ".join(f"{c:.3f}" for c in r.boxes.conf.tolist()),
                     xy_shape=tuple(r.keypoints.xy.shape), conf_shape=tuple(r.keypoints.conf.shape)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 3. 走廊照片：倒地那位（第 0 人）的 17 個點 ==")
r = results["care-corridor-fall"]
xy = r.keypoints.xy[0].numpy().astype(float)
conf = r.keypoints.conf[0].numpy().astype(float)
t = pd.DataFrame({"點": KPTS, "x": xy[:, 0].round(1), "y": xy[:, 1].round(1), "conf": conf.round(3)})
t["低信心"] = np.where(conf < MIN_KPT_CONF, "<0.3", "")
t.index.name = "編號"
print(t.to_string())

print("\n== 4. 每個人有幾個點信心 < 0.3 ==")
for name in PHOTO_NAMES:
    c = results[name].keypoints.conf.numpy()
    for i in range(len(c)):
        low = [KPTS[k] for k in np.where(c[i] < MIN_KPT_CONF)[0]]
        print(f"{name} 第{i}人：{len(low)} 點" + (f"（{', '.join(low)}）" if low else ""))

print("\n== 5. 變成 poses.csv 的格式：一個人一列、17×3＝51 欄 ==")
recs = []
for name in PHOTO_NAMES:
    kp = results[name].keypoints
    for i in range(len(kp.xy)):
        rec = {"photo": name, "person": i}
        for k, n in enumerate(KPTS):
            rec[f"{n}_x"] = round(float(kp.xy[i, k, 0]), 1)
            rec[f"{n}_y"] = round(float(kp.xy[i, k, 1]), 1)
            rec[f"{n}_c"] = round(float(kp.conf[i, k]), 3)
        recs.append(rec)
photo_df = pd.DataFrame(recs)
photo_df.to_csv(OUT / "photo_keypoints.csv", index=False)
poses = load_poses()
kcols = [c for c in poses.columns if c.endswith(("_x", "_y", "_c"))]
print(f"照片表 {photo_df.shape}；關鍵點欄 {len(kcols)} 欄，和 poses.csv 同名同序：{kcols == list(photo_df.columns[2:])}")
print(photo_df[["photo", "person", "nose_x", "nose_y", "nose_c", "left_hip_y", "left_hip_c"]].to_string(index=False))
print(f"poses.csv：{poses.shape[0]} 列（每列＝一格畫面裡的一個人），畫面 640x480")
