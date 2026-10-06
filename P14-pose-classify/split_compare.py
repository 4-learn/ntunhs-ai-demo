"""P14：同一個模型，隨機切格子 vs 依受試者分組，分數差多少（只用 train：S01–S18）。

    python P14-pose-classify/split_compare.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"  # HGB 用 1 個執行緒：結果可重現，在忙碌的機器上反而比較快（要在 import sklearn 前設）
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors

from common import KPTS, frame_features, keypoints, load_poses, subject_split

warnings.simplefilter("ignore", RuntimeWarning)
df = load_poses()
train, test = subject_split(df)
print("== 1. 先封存測試集 ==")
print(f"train：{train.subject_id.nunique()} 人、{train.seq_id.nunique()} 段、{len(train)} 格；"
      f"test：{test.subject_id.nunique()} 人（{test.subject_id.min()}–{test.subject_id.max()}），P16 才打開")

# 每 2 格取 1 格（相鄰兩格幾乎一樣），讓 5 折×4 組實驗在課堂上跑得完
tr = train[train.frame % 2 == 0]
y = tr["posture"]
F = frame_features(tr)
raw = pd.DataFrame(keypoints(tr, mask_low=False)[..., :2].reshape(len(tr), -1), index=tr.index)

print("\n== 2. 隨機切格子：每一格的「最近鄰」是誰 ==")
Z = (F - F.mean()) / F.std()
Zf = Z.fillna(0).to_numpy()  # 只為了找鄰居
nn = NearestNeighbors(n_neighbors=2).fit(Zf)
idx = nn.kneighbors(Zf, return_distance=False)[:, 1]
same_seq = (tr.seq_id.to_numpy()[idx] == tr.seq_id.to_numpy()).mean()
same_subj = (tr.subject_id.to_numpy()[idx] == tr.subject_id.to_numpy()).mean()
print(f"最近的另一格來自同一段序列：{same_seq:.3f}；來自同一個人：{same_subj:.3f}")

print("\n== 3. 同一個 HGB，兩種切法（5 折，macro F1）==")
cvs = {"隨機切格子": (StratifiedKFold(5, shuffle=True, random_state=0), None),
       "依受試者分組": (StratifiedGroupKFold(5, shuffle=True, random_state=0), tr["subject_id"])}
rows = []
for xname, X in [("7 個姿勢特徵", F), ("34 欄像素座標", raw)]:
    for cname, (cv, g) in cvs.items():
        s = cross_val_score(HistGradientBoostingClassifier(random_state=0), X, y, cv=cv, groups=g,
                            scoring="f1_macro")
        rows.append({"輸入": xname, "切法": cname, "平均": s.mean(), "標準差": s.std(), "最低折": s.min()})
t = pd.DataFrame(rows)
print(t.round(3).to_string(index=False))
r = t.set_index(["輸入", "切法"])["平均"]
print(f"隨機切格子：像素座標比特徵高 {r['34 欄像素座標', '隨機切格子'] - r['7 個姿勢特徵', '隨機切格子']:+.3f}")
print(f"依受試者分組：像素座標比特徵高 {r['34 欄像素座標', '依受試者分組'] - r['7 個姿勢特徵', '依受試者分組']:+.3f}")
