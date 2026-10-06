"""V13｜讓 AI 寫，然後挑錯：只報六類平均、把 precision 當上線精確率、按瑕疵類型拆 recall 不寫張數。"""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from common import MVTEC_CATEGORIES, ROOT, mvtec_paths

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
from anomaly_lib import confusion, cost_threshold, scores_patch, valid_test_split  # noqa: E402

data = {}
for cat in MVTEC_CATEGORIES:
    y = mvtec_paths(cat)["y"]
    val, final = valid_test_split(y)
    s = scores_patch(cat)
    data[cat] = (y, val, final, s)

print("== 錯誤 1：只報「六類平均 AUROC」 ==")
aucs = {c: roc_auc_score(y[f], s["test"][f]) for c, (y, v, f, s) in data.items()}
print(f"平均 {np.mean(list(aucs.values())):.3f}；最低 {min(aucs, key=aucs.get)} {min(aucs.values()):.3f}")
print("→ 王課長只做螺絲的話，平均數字對他沒有意義")

print("\n== 錯誤 2：把最終測試的 precision 當成上線後的精確率 ==")
y, val, final, s = data["screw"]
t, _ = cost_threshold(s["calib"], s["test"][val])
r = confusion(y[final], s["test"][final], t)
fp_rate = r["誤報"] / int((y[final] == 0).sum())
for ratio, name in [(int(y[final].sum()) / len(final), "最終測試"), (0.001, "產線 0.1%")]:
    tp, fp = ratio * r["recall"], (1 - ratio) * fp_rate
    print(f"screw，瑕疵比例 {ratio:.3f}（{name}）：precision {tp / (tp + fp):.3f}")

print("\n== 錯誤 3：按瑕疵類型拆 recall，卻不寫張數 ==")
P = mvtec_paths("wood")
y, val, final, s = data["wood"]
t, _ = cost_threshold(s["calib"], s["test"][val])
for kind in ["hole", "scratch"]:
    idx = [i for i in final if P["defect"][i] == kind]
    hit = int((s["test"][idx] > t).sum())
    print(f"wood {kind:<8}{hit}/{len(idx)} = {hit / len(idx):.3f}；多抓 1 張就變 {(hit + 1) / len(idx):.3f}"
          f"（一張＝{1 / len(idx):.1%}）")
print("→ 9 張的 0.778 和 19 張的 0.684 不能直接比「哪個比較差」")
