"""P14：固定切割 fit＝S01–S14、valid＝S15–S18，HGB 姿勢四類，混淆矩陣與錯誤位置。

    python P14-pose-classify/classify.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"  # HGB 用 1 個執行緒：結果可重現，在忙碌的機器上反而比較快（要在 import sklearn 前設）
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from common import FRAME_FEATURES, frame_features, load_poses, subject_split

df = load_poses()
train, _ = subject_split(df)  # 測試集不碰
fit = train[train.subject_id <= "S14"]
valid = train[train.subject_id >= "S15"]
LABELS = ["standing", "sitting", "bending", "lying"]

print("== 1. fit／valid 依受試者切開 ==")
print(f"fit：S01–S14，{len(fit)} 格；valid：S15–S18，{len(valid)} 格；共用受試者 "
      f"{len(set(fit.subject_id) & set(valid.subject_id))} 人")
print(valid.posture.value_counts().reindex(LABELS).to_string())

Xf, Xv = frame_features(fit), frame_features(valid)
model = HistGradientBoostingClassifier(random_state=0).fit(Xf, fit.posture)
pred = model.predict(Xv)
base = DummyClassifier(strategy="most_frequent").fit(Xf, fit.posture)

print("\n== 2. baseline 與 HGB（valid）==")
print(f"全部猜 standing：accuracy {(valid.posture == 'standing').mean():.3f}、macro F1 "
      f"{f1_score(valid.posture, base.predict(Xv), average='macro'):.3f}")
print(f"HGB：accuracy {(pred == valid.posture).mean():.3f}、macro F1 "
      f"{f1_score(valid.posture, pred, average='macro'):.3f}")

print("\n== 3. 每一類的 precision／recall ==")
print(classification_report(valid.posture, pred, labels=LABELS, digits=3))

print("== 4. 混淆矩陣（列＝真實，欄＝預測）==")
cm = pd.DataFrame(confusion_matrix(valid.posture, pred, labels=LABELS), index=LABELS, columns=LABELS)
print(cm.to_string())

print("\n== 5. 錯在哪裡：離「姿勢轉換」幾格 ==")
v = valid[["seq_id", "frame", "activity", "camera_id", "posture"]].copy()
v["pred"] = pred
# 每一格到最近一次標籤改變的距離
dist = []
for _, g in v.groupby("seq_id", sort=False):
    lab = g.posture.to_numpy()
    edges = np.where(lab[1:] != lab[:-1])[0] + 0.5
    fr = np.arange(len(lab))
    dist.append(pd.Series(np.min(np.abs(fr[:, None] - edges[None, :]), 1) if len(edges) else np.full(len(lab), 99.0),
                          index=g.index))
v["dist"] = pd.concat(dist)
v["區段"] = pd.cut(v["dist"], [0, 1, 3, 100], labels=["轉換前後 1 格內", "2–3 格", "4 格以上／沒轉換"])
err = v.posture != v.pred
t = pd.DataFrame({"格數": v.groupby("區段", observed=True).size(),
                  "錯誤": err.groupby(v["區段"], observed=True).sum()})
t["錯誤率"] = (t["錯誤"] / t["格數"]).round(3)
print(t.to_string())

print("\n== 6. bending 的錯：在哪一種活動裡 ==")
b = v[(v.posture == "bending") | (v.pred == "bending")]
b = b[b.posture != b.pred]
print(b.groupby(["activity", "posture", "pred"]).size().rename("格數").to_string())
