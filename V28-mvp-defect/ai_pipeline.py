"""V28「讓 AI 寫，然後挑錯」：AI 版的瑕疵管線，三個錯一次跑出來。

    python V28-mvp-defect/ai_pipeline.py

分數直接讀 V08 anomaly_lib 的分數快取（與 mvp_defect.py 同一個記憶庫、同一批原始分數），不重算 kNN。
"""
import sys

import numpy as np

from common import ROOT, mvtec_paths

sys.path.insert(0, str(ROOT / "V28-mvp-defect"))
from defect_model import conveyor_order, evaluate, load_model, valid_index  # noqa: E402
from anomaly_lib import scores_patch  # noqa: E402
from event_api import EXAMPLE_NG, create_app, make_client  # noqa: E402

CAT = "metal_nut"
P = mvtec_paths(CAT)
S = scores_patch(CAT)
_, card = load_model(CAT)
raw = dict(zip(P["test"], S["test"]))
shots = conveyor_order(CAT)
y = np.array([s.truth for s in shots])
r = np.array([raw[s.path] for s in shots])

print("== AI 錯誤 1：原始分數直接當 score 送出 ==")
client = make_client(create_app())
codes = [client.post("/events", json={**EXAMPLE_NG, "event_id": f"line1cam1-20260112-0800{i:02d}-0",
                                      "score": round(float(v), 3), "threshold": 2.5}).status_code
         for i, v in enumerate(r[:5])]
print("前 5 顆的回應：", codes, "→ 全部 422，產線以為沒有瑕疵")

print("\n== AI 錯誤 2：用整批的最小／最大值把分數縮到 0–1 ==")
def minmax(a):
    return (a - a.min()) / (a.max() - a.min())


for n in (10, 40, len(r)):
    s = minmax(r[:n])
    m = evaluate(y[:n], (s >= 0.5).astype(int))
    print(f"一批 {n:3d} 顆：同一顆 P00001 的分數 {s[0]:.3f}  漏報 {m['漏報']:2d}  誤報 {m['誤報']}")
print("一顆工件的分數取決於「同一批還有誰」；而且要等整批拍完才算得出來")

print("\n== AI 錯誤 3：門檻用整個測試集挑（含驗收用的工件）==")
yt, st = P["y"], S["test"]
best = min(((20 * int(((st < t) & (yt == 1)).sum()) + int(((st >= t) & (yt == 0)).sum()), t) for t in np.unique(st)))
vi = valid_index(yt)
print(f"在測試集上挑的門檻 {best[1]:.3f}（成本 {best[0]}）；V12 規則只看校正良品＋{len(vi)} 張驗證瑕疵 → {card['threshold_raw']:.3f}")
for name, t in [("測試集挑的", best[1]), ("V12 規則", card["threshold_raw"])]:
    m = evaluate(y, (r >= t).astype(int))
    print(f"{name}：輸送帶 105 顆 漏報 {m['漏報']}  誤報 {m['誤報']}  成本 {20 * m['漏報'] + m['誤報']}")
print("前者的數字好看，是因為答案已經看過；這個門檻上線後的表現沒有任何資料可以預估")
