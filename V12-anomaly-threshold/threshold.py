"""V12｜門檻：校正良品最大值 vs 少量瑕疵＋成本（20×漏報＋1×誤報）。方法 C（小格 kNN，記憶庫 10%）。"""
import sys

import numpy as np
import pandas as pd

from common import ROOT, mvtec_paths

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
from anomaly_lib import COST_FN, COST_FP, confusion, cost_threshold, scores_patch, valid_test_split  # noqa: E402

print("== 1. 切出驗證：每類 test 瑕疵固定抽 10 張（seed 0）==")
for cat in ["bottle", "screw"]:
    P = mvtec_paths(cat)
    y = P["y"]
    val, final = valid_test_split(y)
    kinds = pd.Series([P["defect"][i] for i in val]).value_counts().sort_index()
    print(f"{cat}：驗證＝校正良品 {len(scores_patch(cat)['calib'])} 張＋瑕疵 {len(val)} 張"
          f"（{'、'.join(f'{k} {v}' for k, v in kinds.items())}）")
    print(f"{' ' * len(cat)}  最終測試＝良品 {int((y[final] == 0).sum())} 張＋瑕疵 {int(y[final].sum())} 張")

print("\n== 2. screw：在驗證集上掃門檻，成本 = 20×漏報 + 1×誤報 ==")
P = mvtec_paths("screw")
y = P["y"]
s = scores_patch("screw")
val, final = valid_test_split(y)
good, bad = s["calib"], s["test"][val]
rows = []
for t in [1.90, 1.95, 1.977, 2.00, 2.10, 2.30, 2.60, 3.00, round(float(good.max()), 3)]:
    fn, fp = int((bad <= t).sum()), int((good > t).sum())
    rows.append({"門檻": t, "剔除的校正良品(誤報)": fp, "放過的瑕疵(漏報)": fn, "成本": COST_FN * fn + COST_FP * fp})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
t_cost, c = cost_threshold(good, bad)
print(f"成本最低：門檻 {t_cost:.3f}，驗證成本 {c}；校正良品最大值 {good.max():.3f}，驗證成本 {COST_FN * int((bad <= good.max()).sum())}")

print("\n== 3. 兩種門檻在最終測試上的結果 ==")
rows = []
for cat in ["bottle", "screw"]:
    P = mvtec_paths(cat)
    y = P["y"]
    s = scores_patch(cat)
    val, final = valid_test_split(y)
    t_cost, _ = cost_threshold(s["calib"], s["test"][val])
    for name, t in [("校正最大值", s["calib"].max()), ("成本門檻", t_cost)]:
        rows.append({"類別": cat, "門檻規則": name, "門檻": t, **confusion(y[final], s["test"][final], t)})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))

print("\n== 4. 換一批 10 張驗證瑕疵，screw 的成本門檻穩不穩 ==")
P = mvtec_paths("screw")
y = P["y"]
s = scores_patch("screw")
rows = []
for seed in range(5):
    val, final = valid_test_split(y, seed=seed)
    t, _ = cost_threshold(s["calib"], s["test"][val])
    rows.append({"seed": seed, "門檻": t, **confusion(y[final], s["test"][final], t)})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))

print("\n== 5. 搬到產線：每 10,000 支螺絲，瑕疵率 0.1%（10 支）==")
val, final = valid_test_split(y)
good_rate = lambda t: float((s["test"][final][y[final] == 0] > t).mean())
bad_rate = lambda t: float((s["test"][final][y[final] == 1] > t).mean())
rows = []
for name, t in [("校正最大值", s["calib"].max()), ("成本門檻", cost_threshold(s["calib"], s["test"][val])[0])]:
    fp, tp = 9990 * good_rate(t), 10 * bad_rate(t)
    rows.append({"門檻規則": name, "剔除良品(誤報)": round(fp), "抓到瑕疵": round(tp, 1), "漏報": round(10 - tp, 1),
                 "成本": round(COST_FN * (10 - tp) + COST_FP * fp)})
print(pd.DataFrame(rows).to_string(index=False))
print("（用最終測試的良品誤報率、瑕疵 recall 推算；驗證集裡瑕疵佔 10/74，產線只佔 1/1000）")
