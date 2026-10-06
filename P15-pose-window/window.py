"""P15：單格看不出「怎麼變成這樣」；加上 1 秒窗口的變化量，序列層級判斷 is_fall_seq。

    python P15-pose-window/window.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"  # HGB 單執行緒：可重現，共用機器上也較快
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from common import FRAME_FEATURES, WINDOW, WINDOW_FEATURES, frame_features, load_poses, subject_split, window_features

pd.set_option("display.width", 200)
df = load_poses().sort_values(["seq_id", "frame"])
train, _ = subject_split(df)  # 測試集不碰
fit = train[train.subject_id <= "S14"]
valid = train[train.subject_id >= "S15"]
ORDER = ["walk", "stand", "sit", "sit_down", "bend_pick", "fall", "lie_bed"]

print("== 1. 三段序列的最後一格：姿勢一樣，來的路不一樣 ==")
F = frame_features(train)
last = train.groupby("seq_id").tail(1)
show = []
for act in ["fall", "lie_bed", "sit_down"]:
    s = last[(last.activity == act)]
    show.append(F.loc[s.index].assign(activity=act, posture=s.posture).groupby(["activity", "posture"]).median())
print(pd.concat(show)[["torso_angle", "aspect", "hip_height", "knee_angle"]].round(1).to_string())

print("\n== 2. 單格的姿勢分類器（P14）看最後一格 ==")
pm = HistGradientBoostingClassifier(random_state=0).fit(frame_features(fit), fit.posture)
vlast = valid.groupby("seq_id").tail(1)
pred = pd.Series(pm.predict(frame_features(vlast)), index=vlast.index)
print(pd.crosstab(vlast.activity, pred).reindex(["fall", "lie_bed", "sit_down", "sit"]).to_string())

print("\n== 3. 窗口特徵：過去 1 秒（10 格）變了多少 ==")
W = window_features(train)
print(f"window_features 欄位：{WINDOW_FEATURES[len(FRAME_FEATURES):]}；窗口 {WINDOW} 格")
# 每段序列取「變化最大的那一刻」，再看各活動的中位數
mx = W[["d_torso_angle", "d_hip_y", "max_hip_speed"]].groupby(train.seq_id).max()
mx["activity"] = train.groupby("seq_id").activity.first()
print(mx.groupby("activity").median().reindex(ORDER).round(3).to_string())
print(f"前 {WINDOW - 1} 格沒有完整窗口，d_torso_angle 是 NaN：{W.d_torso_angle.isna().mean():.3f}")


def seq_scores(model, d, X):
    """序列分數＝序列內每格跌倒機率的最大值。"""
    p = pd.Series(model.predict_proba(X)[:, 1], index=d.index)
    return pd.DataFrame({"score": p.groupby(d.seq_id).max(),
                         "y": d.groupby("seq_id").is_fall_seq.first(),
                         "activity": d.groupby("seq_id").activity.first()})


after = lambda d: ((d.is_fall_seq == 1) & (d.frame >= d.fall_frame)).astype(int)  # noqa: E731
onset = lambda d: ((d.is_fall_seq == 1) & (d.frame >= d.fall_frame) & (d.frame < d.fall_frame + 15)).astype(int)  # noqa: E731
setups = [("A 單格特徵、標「倒地之後每一格」", frame_features, after),
          ("B 窗口特徵、標「倒地之後每一格」", window_features, after),
          ("C 窗口特徵、標「跌倒開始 1.5 秒內」", window_features, onset)]

print("\n== 4. 三種做法，序列層級（valid 56 段，跌倒 8 段）==")
res, means = [], {}
for name, feat, target in setups:
    m = HistGradientBoostingClassifier(random_state=0).fit(feat(fit), target(fit))
    s = seq_scores(m, valid, feat(valid))
    alarm = s.score >= 0.5
    res.append({"做法": name, "AP": average_precision_score(s.y, s.score),
                "漏報@0.5": int(((s.y == 1) & ~alarm).sum()), "誤報@0.5": int(((s.y == 0) & alarm).sum())})
    means[name[0]] = s.groupby("activity").score.mean()
print(pd.DataFrame(res).round(3).to_string(index=False))

print("\n== 5. 各活動的平均序列分數（越接近 1 越像跌倒）==")
print(pd.DataFrame(means).reindex(ORDER).round(3).T.to_string())
