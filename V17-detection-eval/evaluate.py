"""V17｜在 test 56 張上自己算 IoU、TP/FP/FN；影像層級（任何框 ≥ 門檻即 ng），用 val 選門檻。

    python V17-detection-eval/evaluate.py
"""
import pandas as pd

from detlib import (FN_COST, FP_COST, box_sweep, detect, image_metrics, image_scores, info, iou, load_labels,
                    match, names, per_10k, pick_threshold, pick_threshold_line)

val, test = detect("val"), detect("test")
EXAMPLES = ["metal_nut-scratch-110", "screw-thread_side-121"]  # 教師挑的兩張：有重複框、有漏掉的標註

print("== 1. 一張圖的配對：預測框 vs 標註框 ==")
for n in EXAMPLES:
    gts = load_labels("test", n)
    print(f"[{n}] 標註 {len(gts)} 個框")
    for c, b in [d for d in test[n] if d[0] >= 0.1][:4]:
        best = max((iou(b, g) for g in gts), default=0.0)
        print(f"  預測 conf {c:.3f}  最大 IoU {best:.3f}  → {'TP' if best >= 0.5 else 'FP'}")
    tp, fp, fn = match(test[n], gts, 0.1)
    print(f"  門檻 0.1、IoU ≥ 0.5：TP {tp}  FP {fp}  FN {fn}")

print("\n== 2. 框層級（test 56 張、56 個標註框，IoU ≥ 0.5）==")
print(pd.DataFrame(box_sweep(test, "test", [0.1, 0.25, 0.5])).to_string(index=False))
r = box_sweep(test, "test", [0.1], iou_min=0.1)[0]
print(f"把 IoU 要求放寬到 0.1（門檻 0.1）：TP {r['TP']}  FP {r['FP']}  FN {r['FN']}")

print(f"\n== 3. 影像層級：任何框信心 ≥ 門檻 → ng；val 選門檻（成本 {FN_COST}×漏報 + {FP_COST}×誤報）==")
s_val, y_val = image_scores(val, "val")
print(f"val {len(y_val)} 張：瑕疵 {y_val.sum()}、良品 {(y_val == 0).sum()}")
rows = [{"門檻": t, **image_metrics(y_val, s_val, t)} for t in (0.01, 0.02, 0.05, 0.1, 0.3, 0.5)]
print(pd.DataFrame(rows).to_string(index=False))
thr = pick_threshold(y_val, s_val)
print(f"val 成本最低的門檻：{thr}")

print("\n== 4. test 只打開一次（影像層級）==")
s_te, y_te = image_scores(test, "test")
rows = [{"model": "全猜良品", **image_metrics(y_te, s_te * 0, 0.5)},
        {"model": "全判ng", **image_metrics(y_te, s_te * 0 + 1, 0.5)},
        {"model": f"YOLO 門檻 {thr}", **image_metrics(y_te, s_te, thr)}]
print(pd.DataFrame(rows).to_string(index=False))
good = y_te == 0
print(f"良品誤剔率：{(s_te[good] >= thr).mean():.3f}（{int((s_te[good] >= thr).sum())}/{good.sum()}）")

print("\n== 5. 搬到產線：每 10,000 件、瑕疵率 0.1%（10 件）==")
thr_line = pick_threshold_line(y_val, s_val)
rows = [{"門檻規則": f"val 成本 {thr}", **per_10k(y_te, s_te >= thr)},
        {"門檻規則": f"產線加權 {thr_line}", **per_10k(y_te, s_te >= thr_line)}]
print(pd.DataFrame(rows).to_string(index=False))
print(f"產線加權門檻 {thr_line} 在 test：", image_metrics(y_te, s_te, thr_line))

print(f"\n== 6. 錯誤地圖：test 依產品與瑕疵類型（門檻 {thr} 與 {thr_line}）==")
df = pd.DataFrame([(*info(n), s, y) for n, s, y in zip(names("test"), s_te, y_te)],
                  columns=["cat", "defect", "score", "y"])
df["判ng"] = (df.score >= thr).astype(int)
df["判ng_產線"] = (df.score >= thr_line).astype(int)
g = df.groupby(["cat", "defect"]).agg(張數=("y", "size"), **{f"≥{thr}": ("判ng", "sum"), f"≥{thr_line}": ("判ng_產線", "sum")},
                                     最高信心中位數=("score", "median"))
print(g.round(3).to_string())
