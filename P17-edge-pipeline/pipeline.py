"""P17：EZCam（mock）→ YOLO-pose → 特徵 → P16 定版模型 → Event JSON。

    NTUNHS_MOCK=1 python P17-edge-pipeline/pipeline.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("YOLO_OFFLINE", "1")
import json
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from common import PHOTOS, ROOT, frame_features, load_poses, subject_split, window_features

sys.path.insert(0, str(ROOT / "P09-ezcam"))
sys.path.insert(0, str(ROOT / "P17-edge-pipeline"))
from edge_pipeline import TZ, FallDetector, load_fall_model, load_pose_model, replay, yolo_rows  # noqa: E402
from ezcam_client import capture, is_mock  # noqa: E402

model, cfg = load_fall_model()
pose = load_pose_model()

print("== 1. 啟動檢查 ==")
print(f"模式：{'mock（讀 data/photos/ezcam-sample.jpg，不連網路）' if is_mock() else 'EZCam 真機'}")
print(f"跌倒模型 {cfg['model']['name']}@{cfg['model']['sha256_12']}，門檻 {cfg['threshold']['value']}"
      f"（{cfg['threshold']['approved_by']}）")

print("\n== 2. 擷取一張 → YOLO-pose ==")
shot = capture()
rows = yolo_rows(pose, shot.image)
print(f"EZCam 照片 {shot.image.shape[1]}x{shot.image.shape[0]}，偵測到 {len(rows)} 人 → 沒有人就沒有骨架、也沒有事件")
img = cv2.imread(str(PHOTOS / "care-corridor-fall.png"))
rows = yolo_rows(pose, img)
print(f"走廊照片 {img.shape[1]}x{img.shape[0]}，偵測到 {len(rows)} 人；每人一列 {rows.shape[1]} 欄（和 poses.csv 相同）")
del img  # 影像用完就丟：管線往下只傳骨架

print("\n== 3. 單張照片丟進窗口模型：沒有「1 秒前」==")
one = rows.assign(seq_id=[f"photo#{i}" for i in range(len(rows))], frame=0)
W = window_features(one)
p = model.predict_proba(W[cfg["model"]["features"]])[:, 1]
print(pd.DataFrame({"人": ["倒在地上", "照服員走來"], "torso_angle": W.torso_angle.round(1),
                    "d_torso_angle": W.d_torso_angle, "跌倒機率": p.round(3)}).to_string(index=False))

print("\n== 4. 用 10 fps 串流重播測試集的序列（一格一格送進偵測器）==")
df = load_poses()
_, test = subject_split(df)
start = datetime(2026, 1, 12, 2, 15, 0, tzinfo=TZ)
det = FallDetector(model, cfg, camera_id="cam2", prefix="s1cam2")
picks = ["SQ0264", "SQ0308", "SQ0302"]   # 測試集：一段跌倒、一段躺床、一段坐下
for sid in picks:
    seq = test[test.seq_id == sid]
    probs, events = replay(det, seq, start)
    ff = int(seq.fall_frame.iloc[0])
    when = f"第 {events[0][0]} 格發出事件" if events else "沒有事件"
    print(f"{sid} {seq.activity.iloc[0]:9s} fall_frame {ff:3d}  最高機率 {probs.max():.3f}  {when}")

print("\n== 5. 串流算出來的分數＝批次算的分數嗎 ==")
seq = test[test.seq_id == picks[0]].sort_values("frame")
det = FallDetector(model, cfg, camera_id="cam2", prefix="s1cam2")   # 新的一支攝影機：事件編號從 0 開始
probs, events = replay(det, seq, start)
batch = model.predict_proba(window_features(seq)[cfg["model"]["features"]])[:, 1]
print(f"40 格逐格比較，最大差異 {np.abs(probs - batch).max():.6f}")

print("\n== 6. 事件長這樣（簡化版 Event JSON，V27 再補 site 等欄位）==")
frame, ev = events[0]
print(json.dumps(ev, ensure_ascii=False, indent=2))
print(f"事件大小 {len(json.dumps(ev, ensure_ascii=False).encode())} bytes；裡面沒有影像")

print("\n== 7. EZCam 每 3 秒一張：窗口模型還能用嗎 ==")
rows_out = []
for step, name in [(1, "10 fps（每 0.1 秒）"), (10, "1 fps（每 1 秒）"), (30, "EZCam（每 3 秒）")]:
    sub = test[test.frame % step == 0]
    s = pd.Series(model.predict_proba(window_features(sub)[cfg["model"]["features"]])[:, 1],
                  index=sub.index).groupby(sub.seq_id).max()
    y = test.groupby("seq_id").is_fall_seq.first()
    a = s >= cfg["threshold"]["value"]
    rows_out.append({"取樣": name, "每段格數": int(40 / step), "抓到跌倒": f"{int((a & (y == 1)).sum())}/{int(y.sum())}",
                     "誤報": int((a & (y == 0)).sum())})
print(pd.DataFrame(rows_out).to_string(index=False))
