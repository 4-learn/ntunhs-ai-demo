"""P17：邊緣端管線的零件——擷取 → YOLO-pose → 一列 51 欄 → 窗口緩衝 → 特徵 → 模型 → Event JSON（簡化版）。

    import sys; sys.path.insert(0, "P17-edge-pipeline")
    from edge_pipeline import FallDetector, yolo_rows, load_fall_model

只存骨架、不存畫面：Event JSON 裡沒有影像，只有分數與序列編號（P18）。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("YOLO_OFFLINE", "1")
from collections import deque
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import yaml

from common import FPS, KPTS, MODELS, ROOT, window_features

TZ = timezone(timedelta(hours=8))
CONFIG = ROOT / "config" / "fall.yaml"
BUFFER = 11          # 1 秒窗口（10 格）＋1 格，讓 max_hip_speed 的 10 個差分都算得到


def load_fall_model():
    """讀 P16 的定版模型與設定檔，並確認 hash 一致。"""
    import hashlib

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    path = ROOT / cfg["model"]["path"]
    got = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    if got != cfg["model"]["sha256_12"]:
        raise RuntimeError(f"模型檔 hash {got} 與 config/fall.yaml 的 {cfg['model']['sha256_12']} 不符，拒絕啟動")
    return joblib.load(path), cfg


def load_pose_model():
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    return YOLO(str(MODELS / "yolo11n-pose.pt"))


def yolo_rows(pose_model, image) -> pd.DataFrame:
    """一張影像 → 每個人一列（51 欄，和 poses.csv 相同）。影像用完就丟，不回傳。"""
    r = pose_model(image, verbose=False)[0]
    if r.keypoints is None or r.keypoints.conf is None or len(r.keypoints.xy) == 0:
        return pd.DataFrame(columns=[f"{k}_{a}" for k in KPTS for a in "xyc"])
    xy = r.keypoints.xy.numpy().astype(float)
    conf = r.keypoints.conf.numpy().astype(float)
    recs = []
    for i in range(len(xy)):
        rec = {}
        for k, n in enumerate(KPTS):
            rec[f"{n}_x"], rec[f"{n}_y"], rec[f"{n}_c"] = round(xy[i, k, 0], 1), round(xy[i, k, 1], 1), round(conf[i, k], 3)
        recs.append(rec)
    return pd.DataFrame(recs)


class FallDetector:
    """一支攝影機、一個人的串流判斷：每進來一格，回傳這一格的跌倒機率；超過門檻就產生事件。

    同一段序列（同一次「有人在畫面裡」）內，只發一次事件——冷卻與狀態機留給 V21、V22。
    """

    def __init__(self, model, cfg, camera_id="cam2", prefix="s1cam2"):
        self.model, self.cfg = model, cfg
        self.threshold = cfg["threshold"]["value"]
        self.camera_id, self.prefix = camera_id, prefix
        self.buf: deque = deque(maxlen=BUFFER)
        self.fired = False
        self.n_events = 0

    def reset(self):
        self.buf.clear()
        self.fired = False

    def push(self, row: dict, seq_id: str, frame: int, t: datetime):
        row = dict(row, seq_id=seq_id, frame=frame)
        self.buf.append(row)
        win = pd.DataFrame(list(self.buf))
        X = window_features(win)[self.cfg["model"]["features"]].iloc[[-1]]
        p = float(self.model.predict_proba(X)[0, 1])
        event = None
        if p >= self.threshold and not self.fired:
            self.fired = True
            event = self.make_event(p, seq_id, t)
        return p, event

    def make_event(self, score, seq_id, t):
        ev = {
            "event_id": f"{self.prefix}-{t:%Y%m%d-%H%M%S}-{self.n_events}",
            "source": "pose",
            "camera_id": self.camera_id,
            "event_time": t.isoformat(timespec="seconds"),
            "kind": "fall",
            "score": round(score, 3),
            "threshold": self.threshold,
            "evidence": {"seq_id": seq_id, "keypoints_only": True},
            "model": {k: self.cfg["model"][k] for k in ("name", "version", "sha256_12")},
        }
        self.n_events += 1
        return ev


def frame_time(start: datetime, frame: int, fps: int = FPS) -> datetime:
    return start + timedelta(seconds=frame / fps)


def replay(det: FallDetector, seq: pd.DataFrame, start: datetime):
    """把 poses.csv 的一段序列當成攝影機串流，一格一格送進偵測器。"""
    cols = [f"{k}_{a}" for k in KPTS for a in "xyc"]
    det.reset()
    probs, events = [], []
    for _, r in seq.sort_values("frame").iterrows():
        p, ev = det.push(r[cols].to_dict(), r.seq_id, int(r.frame), frame_time(start, int(r.frame)))
        probs.append(p)
        if ev:
            events.append((int(r.frame), ev))
    return np.array(probs), events
