"""P16：AI 常寫的「在測試集上找最佳門檻」，和「每次改完都看一下測試集」，實際跑一次。

    python P16-pose-eval/pitfalls.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

from build_fall_model import GRID, counts, new_model, onset_label, seq_meta
from common import FRAME_FEATURES, WINDOW_FEATURES, load_poses, subject_split, window_features

df = load_poses().sort_values(["seq_id", "frame"])
train, test = subject_split(df)
tm = seq_meta(test)
Xtr, ytr, Xte = window_features(train), onset_label(train), window_features(test)

print("== 1. 在測試集上掃門檻，挑 F1 最高的 ==")
m = new_model().fit(Xtr[WINDOW_FEATURES], ytr)
score = pd.Series(m.predict_proba(Xte[WINDOW_FEATURES])[:, 1], index=test.index).groupby(test.seq_id).max()
rows = [{"門檻": t, **counts(tm.y, score, t)} for t in GRID]
tab = pd.DataFrame(rows)
tab["f1"] = (2 * tab.tp / (2 * tab.tp + tab.fn + tab.fp)).round(3)
best = tab.loc[tab.f1.idxmax()]
print(f"測試集上 F1 最高的門檻：{best.門檻}，F1 {best.f1}；"
      f"同一 F1 的門檻有 {int((tab.f1 == best.f1).sum())} 個（{tab[tab.f1 == best.f1].門檻.min()}–"
      f"{tab[tab.f1 == best.f1].門檻.max()}）")

print("\n== 2. 「每改一次就看一下測試集」：5 種特徵組合，挑測試集最好的 ==")
combos = {"7 個單格特徵": FRAME_FEATURES,
          "只有 3 個窗口特徵": WINDOW_FEATURES[7:],
          "窗口＋軀幹角": WINDOW_FEATURES[7:] + ["torso_angle"],
          "窗口＋外框比": WINDOW_FEATURES[7:] + ["aspect"],
          "全部 10 個": WINDOW_FEATURES}
res = []
for name, cols in combos.items():
    mm = new_model().fit(Xtr[cols], ytr)
    s = pd.Series(mm.predict_proba(Xte[cols])[:, 1], index=test.index).groupby(test.seq_id).max()
    best_cost = min(counts(tm.y, s, t)["cost"] for t in GRID)
    res.append({"特徵": name, "測試集最低成本（偷挑門檻）": best_cost,
                "門檻 0.95 的成本": counts(tm.y, s, 0.95)["cost"]})
print(pd.DataFrame(res).to_string(index=False))
print("測試集只有 11 段跌倒：多漏一段＝成本 +10，多一個誤報＝+1")
