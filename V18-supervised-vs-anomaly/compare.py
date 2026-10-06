"""V18｜同一批測試影像（V15 的 test 56 張：metal_nut＋screw），比較 YOLO（V16）與 V10 小格 kNN。

    python V18-supervised-vs-anomaly/compare.py

- 小格 kNN：V08 anomaly_lib.scores_patch（記憶庫＝train 記憶良品、小格抽 10%；分數＝最不像的那一格），
  每個產品各自一個記憶庫；門檻用 anomaly_lib.cost_threshold，但驗證資料換成 V15 val（和 YOLO 同一批）。
- YOLO：影像分數＝最高的框信心；兩個產品共用一個模型、一個門檻。
- 兩者都在 V15 的 val 54 張上用同一條規則選門檻（cost_threshold；另列產線瑕疵率 0.1% 加權版）；test 只打開一次。
"""
import sys

import numpy as np
import pandas as pd

from common import ROOT

sys.path.insert(0, str(ROOT / "V17-detection-eval"))
from detlib import SIZE, image_metrics, load_labels, per_10k  # noqa: E402
from vslib import CATS, cost_threshold, index_of, line_threshold, predict, table, thresholds  # noqa: E402

df, knn, det, calib = table()

print("== 1. 同一批影像：V15 的 val／test ==")
print(df.groupby(["split", "cat"]).agg(張數=("y", "size"), 瑕疵=("y", "sum")).to_string())

print("\n== 2. 小格 kNN：每個產品自己的記憶庫（train 記憶良品，小格抽 10%）==")
for cat in CATS:
    print(f"{cat}：校正良品 {len(calib[cat])} 張，最大分數 {calib[cat].max():.3f}")

print("\n== 3. 門檻：兩個方法用同一條規則（anomaly_lib.cost_threshold，驗證＝V15 val）==")
va, te = df[df.split == "val"], df[df.split == "test"]
RULES = {"val成本": cost_threshold, "產線加權": line_threshold}
thr = {(m, r): thresholds(va, col, per_cat=(col == "knn"), rule=f)
       for m, col in (("YOLO", "yolo"), ("kNN", "knn")) for r, f in RULES.items()}
for (m, r), t in thr.items():
    print(f"{m:4s} {r}：" + "、".join(f"{c} {t[c]:.3f}" for c in CATS))
print("（YOLO 門檻 0.000＝只要有任何框（信心 ≥ 0.01）就剔除；分數 > 門檻才剔除）")

print("\n== 4. test 只打開一次：影像層級（val 成本門檻）==")
te = te.assign(YOLO=predict(te, "yolo", thr["YOLO", "val成本"]), kNN=predict(te, "knn", thr["kNN", "val成本"]))
rows = []
for cat in CATS + ["合計"]:
    t = te if cat == "合計" else te[te.cat == cat]
    rows.append({"產品": cat, "方法": "YOLO（有標註）", **image_metrics(t.y, t.YOLO, 0.5)})
    rows.append({"產品": cat, "方法": "小格kNN（只有良品）", **image_metrics(t.y, t.kNN, 0.5)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 5. 搬到產線：每 10,000 件、瑕疵率 0.1%（test 的比率換算）==")
rows = []
for (m, r), t in thr.items():
    for cat in CATS:
        x = te[te.cat == cat]
        rows.append({"方法": m, "門檻": r, "產品": cat, **per_10k(x.y, predict(x, m.lower(), t))})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 6. 依瑕疵類型：test 判 ng 幾張（產線加權門檻）==")
te = te.assign(YOLO產線=predict(te, "yolo", thr["YOLO", "產線加權"]), kNN產線=predict(te, "knn", thr["kNN", "產線加權"]))
print(te.groupby(["cat", "defect"]).agg(張數=("y", "size"), YOLO=("YOLO產線", "sum"), kNN=("kNN產線", "sum")).to_string())

print("\n== 7. 定位：判 ng 的瑕疵圖，指出的位置有沒有落在標註框裡 ==")


def inside(xy, boxes, pad=8):
    return any(b[0] - pad <= xy[0] <= b[2] + pad and b[1] - pad <= xy[1] <= b[3] + pad for b in boxes)


rows = []
for method in ("YOLO", "kNN"):
    hit = n = 0
    for _, r in te[(te.y == 1) & (te[method] == 1)].iterrows():
        gts = load_labels("test", r["name"])
        if method == "YOLO":
            c, b = max(det["test"][r["name"]], key=lambda d: d[0])
            xy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        else:
            m = knn[r["cat"]][index_of(r["name"])]
            iy, ix = np.unravel_index(m.argmax(), m.shape)
            xy = ((ix + 0.5) * SIZE / 28, (iy + 0.5) * SIZE / 28)
        n += 1
        hit += inside(xy, gts)
    rows.append({"方法": method, "判ng的瑕疵圖": n, "位置落在框內": hit, "比例": round(hit / n, 3) if n else 0.0})
print(pd.DataFrame(rows).to_string(index=False))
print("YOLO 用最高信心框的中心；kNN 用熱圖最高的小格中心（28×28 → 320×320），框外 8 像素內也算")
