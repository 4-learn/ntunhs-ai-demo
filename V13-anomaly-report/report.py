"""V13｜六類產品成績表與錯誤分析：方法 A／B／C × 6 類，V12 定版門檻（10 張驗證瑕疵＋20×漏報＋1×誤報）。

這張表被 V18、V28、V33 引用；它的 CSV 存到 /tmp/ntunhs/V13/report.csv。
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from common import MVTEC_CATEGORIES, ROOT, mvtec_paths

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
from anomaly_lib import METHODS, confusion, cost_threshold, valid_test_split  # noqa: E402

OUT = Path("/tmp/ntunhs/V13")
OUT.mkdir(parents=True, exist_ok=True)
KIND = {"bottle": "物件", "screw": "物件", "hazelnut": "物件", "metal_nut": "物件", "carpet": "材質", "wood": "材質"}

rows, per_defect = [], []
for cat in MVTEC_CATEGORIES:
    P = mvtec_paths(cat)
    y = P["y"]
    val, final = valid_test_split(y)
    for name, fn in METHODS.items():
        s = fn(cat)
        t, _ = cost_threshold(s["calib"], s["test"][val])
        r = confusion(y[final], s["test"][final], t)
        n_good = int((y[final] == 0).sum())
        rows.append({"類別": cat, "種類": KIND[cat], "方法": name[0],
                     "AUROC": roc_auc_score(y[final], s["test"][final]), **r, "良品誤剔率": r["誤報"] / n_good})
        if name.startswith("C"):
            hit = s["test"] > t
            for i in final[y[final] == 1]:
                per_defect.append({"類別": cat, "瑕疵類型": P["defect"][i], "抓到": bool(hit[i]),
                                   "分數/門檻": s["test"][i] / t, "圖": "/".join(P["test"][i].split("/")[-2:])})

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "report.csv", index=False)

print("== 1. 成績表：最終測試，V12 定版門檻 ==")
print("（AUROC 也只算最終測試；A 像素差、B 整張 kNN、C 小格 kNN 記憶庫 10%）")
print(tab.set_index(["類別", "方法"]).drop(columns="種類").to_string(float_format="%.3f"))
print(f"已存 {OUT / 'report.csv'}")

print("\n== 2. 一眼看完：AUROC ==")
wide = tab.pivot(index="類別", columns="方法", values="AUROC").loc[MVTEC_CATEGORIES]
wide["種類"] = [KIND[c] for c in wide.index]
print(wide.to_string(float_format="%.3f"))
print("平均：", "、".join(f"{m} {wide[m].mean():.3f}" for m in "ABC"))

print("\n== 3. 物件 vs 材質（方法 C）==")
c = tab[tab["方法"] == "C"]
g = c.groupby("種類").agg(類數=("類別", "size"), 平均AUROC=("AUROC", "mean"), 平均recall=("recall", "mean"),
                         漏報=("漏報", "sum"), 誤報=("誤報", "sum"), 成本=("成本", "sum"))
print(g.to_string(float_format="%.3f"))
for m in "AB":
    sub = tab[tab["方法"] == m].groupby("種類")["AUROC"].mean()
    print(f"方法 {m} 平均 AUROC：物件 {sub['物件']:.3f}、材質 {sub['材質']:.3f}")

print("\n== 4. 方法 C：按瑕疵類型拆 recall（最低的 8 種）==")
pdf = pd.DataFrame(per_defect)
d = pdf.groupby(["類別", "瑕疵類型"])["抓到"].agg(["size", "sum", "mean"])
d.columns = ["張數", "抓到", "recall"]
print(d.sort_values(["recall", "張數"], ascending=[True, False]).head(8).to_string(float_format="%.3f"))
print(f"recall 1.000 的瑕疵類型：{int((d['recall'] == 1).sum())} / {len(d)} 種")

print("\n== 5. 方法 C 漏掉的瑕疵：離門檻多遠 ==")
miss = pdf[~pdf["抓到"]].sort_values("分數/門檻")
print(f"共漏 {len(miss)} 張；分數/門檻 > 0.9（差一點就抓到）的 {int((miss['分數/門檻'] > 0.9).sum())} 張")
print(miss.groupby("類別")["分數/門檻"].agg(["size", "min", "median"]).rename(
    columns={"size": "漏報", "min": "最低", "median": "中位數"}).to_string(float_format="%.3f"))
print("離門檻最遠的 3 張：", "、".join(f"{c} {g}（{v:.2f}）" for c, g, v in zip(miss["類別"][:3], miss["圖"][:3], miss["分數/門檻"][:3])))
miss.to_csv(OUT / "missed.csv", index=False)

print("\n== 6. 搬到產線：方法 C，每 10,000 件、瑕疵率 0.1% ==")
c = c.set_index("類別").loc[MVTEC_CATEGORIES]
prod = pd.DataFrame({"誤剔良品": (9990 * c["良品誤剔率"]).round(0).astype(int),
                     "漏報": (10 * (1 - c["recall"])).round(1)})
prod["成本"] = (20 * prod["漏報"] + prod["誤剔良品"]).round(0).astype(int)
print(prod.to_string())
