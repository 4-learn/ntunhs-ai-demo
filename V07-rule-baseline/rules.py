"""V07 傳統視覺 baseline：用 V05–V06 的量測寫規則，良品定門檻，測試集算 recall／precision／誤報／漏報。

    python V07-rule-baseline/rules.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "V06-hough-measure"))
sys.path.insert(0, str(HERE.parent / "V04-augmentation"))
from measure_lib import bottle_measure, load_gray, nut_measure, NUT_SIZE  # noqa: E402
from chirality import handedness  # noqa: E402
from common import memory_calib_split, mvtec_paths  # noqa: E402

K = 3  # 規則門檻：記憶良品的 平均 ± K×標準差


def table(paths, fn):
    return pd.DataFrame([fn(p) for p in paths])


def fit_rules(mem, feats, k=K):
    """每個特徵一條規則：落在 平均 ± k×標準差 之外就判瑕疵。"""
    return {f: (mem[f].mean() - k * mem[f].std(), mem[f].mean() + k * mem[f].std()) for f in feats}


def apply_rules(df, rules):
    hit = pd.DataFrame({f: (df[f] < lo) | (df[f] > hi) for f, (lo, hi) in rules.items()})
    return hit.any(axis=1).to_numpy(), hit


def scores(y, pred):
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "誤報": fp, "漏報": fn}


def run(cat, fn, feats, extra=None):
    P = mvtec_paths(cat)
    mem_p, cal_p = memory_calib_split(P["train"])
    mem, cal, te = table(mem_p, fn), table(cal_p, fn), table(P["test"], fn)
    if extra:
        for name, f in extra.items():
            mem[name], cal[name], te[name] = (np.array([f(p) for p in ps]) for ps in (mem_p, cal_p, P["test"]))
    rules = fit_rules(mem, feats)
    cal_pred, _ = apply_rules(cal, rules)
    te_pred, hit = apply_rules(te, rules)
    return P, rules, cal_pred, te_pred, hit


print("== 1. bottle：規則門檻（記憶良品 167 張，平均 ± 3×標準差）==")
B_FEATS = ["area", "circularity", "fill"]
P, rules, cal_pred, pred, hit = run("bottle", bottle_measure, B_FEATS)
for f, (lo, hi) in rules.items():
    print(f"{f:<12} 正常範圍 {lo:10.3f} ～ {hi:10.3f}")
print(f"校正良品 {len(cal_pred)} 張，被規則判瑕疵：{int(cal_pred.sum())} 張")
y = P["y"]

print("\n== 2. bottle：測試集 ==")
by = pd.DataFrame({"defect": P["defect"], "判瑕疵": pred})
print(by.groupby("defect")["判瑕疵"].agg(["count", "sum"]).rename(columns={"count": "張數", "sum": "判瑕疵"}).to_string())
print("每條規則各抓到幾張瑕疵：", {f: int(hit[f][y == 1].sum()) for f in B_FEATS})
results = [{"類別": "bottle", "規則": "面積＋圓度＋填滿率", **scores(y, pred)}]

print("\n== 3. metal_nut：只用幾何規則 ==")
N_FEATS = ["area", "perimeter", "circularity", "solidity"]
nut_hand = {"handedness": lambda p: handedness(load_gray(p, NUT_SIZE))}
P, rules, cal_pred, pred, hit = run("metal_nut", nut_measure, N_FEATS, nut_hand)
y = P["y"]
print(f"校正良品 {len(cal_pred)} 張，被規則判瑕疵：{int(cal_pred.sum())} 張")
by = pd.DataFrame({"defect": P["defect"], "判瑕疵": pred})
print(by.groupby("defect")["判瑕疵"].agg(["count", "sum"]).rename(columns={"count": "張數", "sum": "判瑕疵"}).to_string())
results.append({"類別": "metal_nut", "規則": "面積＋周長＋圓度＋凸包比", **scores(y, pred)})

print("\n== 4. metal_nut：每條規則各抓到哪一種瑕疵（含 V04 的爪子方向）==")
_, rules2, cal_pred2, pred2, hit2 = run("metal_nut", nut_measure, N_FEATS + ["handedness"], nut_hand)
m = hit2.assign(defect=P["defect"]).groupby("defect").sum().astype(int)
print(m.to_string())
results.append({"類別": "metal_nut", "規則": "幾何＋爪子方向", **scores(y, pred2)})

print("\n== 5. 傳統視覺 baseline 成績表（V08 起的 AI 方法要贏過它）==")
print(pd.DataFrame(results).set_index(["類別", "規則"]).round(3).to_string())
