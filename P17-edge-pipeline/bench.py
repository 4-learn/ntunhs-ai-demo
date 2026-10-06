"""P17：量本機每個階段花多少時間（限 4 執行緒）。數字因機器而異，不放進講義的輸出區塊。

    python P17-edge-pipeline/bench.py

Pi 5 未實測：本機沒有 Pi 5，教師開課前在 Pi 5 上執行本腳本補上數字。
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("YOLO_OFFLINE", "1")
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from common import PHOTOS, ROOT, KPTS, load_poses

sys.path.insert(0, str(ROOT / "P17-edge-pipeline"))
from edge_pipeline import TZ, FallDetector, load_fall_model, load_pose_model, yolo_rows  # noqa: E402

model, cfg = load_fall_model()
pose = load_pose_model()
img = cv2.imread(str(PHOTOS / "care-corridor-fall.png"))
small = cv2.resize(img, (640, 480))


def ms(fn, n):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


print("== 本機 CPU（torch 4 執行緒、HGB 1 執行緒），每次平均 ms ==")
print(f"YOLO-pose 1448x1086 照片：{ms(lambda: yolo_rows(pose, img), 5):.0f} ms")
print(f"YOLO-pose 640x480：{ms(lambda: yolo_rows(pose, small), 5):.0f} ms")
seq = load_poses().query("seq_id == 'SQ0264'").sort_values("frame")
cols = [f"{k}_{a}" for k in KPTS for a in "xyc"]
recs = [r[cols].to_dict() for _, r in seq.iterrows()]
det = FallDetector(model, cfg)
t0 = datetime(2026, 1, 12, 2, 15, tzinfo=TZ)


def stream():
    det.reset()
    for i, r in enumerate(recs):
        det.push(r, "SQ0264", i, t0)


print(f"特徵＋跌倒模型，每格：{ms(stream, 3) / len(recs):.1f} ms")
