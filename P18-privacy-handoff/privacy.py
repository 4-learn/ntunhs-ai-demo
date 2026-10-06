"""P18：只存骨架不存畫面——資料量、事件內容，以及「骨架能不能認出是誰」。

    python P18-privacy-handoff/privacy.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
import json
import warnings

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from common import FPS, KPTS, PHOTOS, keypoints, load_poses, subject_split

warnings.simplefilter("ignore", RuntimeWarning)

print("== 1. 同一秒鐘，存影像 vs 存骨架 ==")
img = cv2.imread(str(PHOTOS / "care-corridor-fall.png"))
small = cv2.resize(img, (640, 480))
ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
one_person = 17 * 3 * 4          # 17 點 × (x, y, 信心) × float32
rows = [("640x480 JPEG（品質 80）", len(jpg)), ("一個人的骨架（17 點 float32）", one_person)]
print(pd.DataFrame([{"存什麼": a, "一格 bytes": b, f"一晚 8 小時、{FPS} fps（MB）": round(b * FPS * 8 * 3600 / 1e6, 1)}
                    for a, b in rows]).to_string(index=False))
print(f"骨架約是影像的 1/{len(jpg) // one_person}；而且骨架裡沒有臉、沒有衣服、沒有房間")

print("\n== 2. 一筆事件裡有什麼 ==")
ev = {"event_id": "s1cam2-20260112-021501-0", "camera_id": "cam2", "event_time": "2026-01-12T02:15:01+08:00",
      "kind": "fall", "score": 0.963, "threshold": 0.95, "evidence": {"seq_id": "SQ0264", "keypoints_only": True}}
print(f"欄位：{list(ev)}")
print(f"可以直接認出是誰的欄位：{[k for k in ev if k in ('name', 'resident_id', 'bed_no', 'image', 'face')]}")
print("可以間接推出是誰的欄位：['camera_id', 'event_time']（哪支攝影機＋幾點，對照排班與床位就知道）")

print("\n== 3. 骨架能不能認出是誰：只看站著、走路的格，猜是 18 位受試者的哪一位 ==")
df = load_poses()
train, _ = subject_split(df)
st = train[train.activity.isin(["walk", "stand"])]
k = keypoints(st)
xy = k[..., :2]


def seg(a, b):
    return np.linalg.norm(xy[:, a] - xy[:, b], axis=1)


torso = np.linalg.norm(np.nanmean(xy[:, [5, 6]], 1) - np.nanmean(xy[:, [11, 12]], 1), axis=1)
ratio = pd.DataFrame({"大腿/軀幹": (seg(11, 13) + seg(12, 14)) / 2 / torso,
                      "小腿/軀幹": (seg(13, 15) + seg(14, 16)) / 2 / torso,
                      "上臂/軀幹": (seg(5, 7) + seg(6, 8)) / 2 / torso,
                      "肩寬/軀幹": seg(5, 6) / torso}, index=st.index)
h = pd.Series(np.nanmax(xy[..., 1], 1) - np.nanmin(xy[..., 1], 1), index=st.index)
other = pd.DataFrame({"相對身高": h / h.groupby(st.camera_id).transform("median"),
                      "面向": np.sign(xy[:, 0, 0] - np.nanmean(xy[:, [11, 12], 0], 1))}, index=st.index)
y, g = st.subject_id, st.seq_id
res = []
for name, X in [("身體比例（4 個比值）", ratio), ("相對身高＋面向", other), ("全部", pd.concat([ratio, other], axis=1))]:
    p = cross_val_predict(HistGradientBoostingClassifier(random_state=0, max_iter=50), X, y, cv=GroupKFold(5), groups=g)
    vote = pd.Series(p, index=st.index).groupby(g).agg(lambda s: s.mode()[0])
    res.append({"用什麼猜": name, "每格猜對": (p == y).mean(), "每段（多數決）猜對": (vote == y.groupby(g).first()).mean()})
print(pd.DataFrame(res).round(3).to_string(index=False))
print(f"亂猜：{1 / 18:.3f}（18 選 1）")
