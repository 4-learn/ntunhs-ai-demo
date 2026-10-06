"""V12｜讓 AI 寫，然後挑錯：用測試集選門檻（作弊示範）、F1 當標準、驗證瑕疵沒從測試集拿掉。"""
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_curve

from common import ROOT, mvtec_paths

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
from anomaly_lib import confusion, cost_threshold, scores_patch, valid_test_split  # noqa: E402

P = mvtec_paths("screw")
y = P["y"]
s = scores_patch("screw")
val, final = valid_test_split(y)
yf, sf = y[final], s["test"][final]


def show(name, t, yy=yf, ss=sf):
    r = confusion(yy, ss, t)
    print(f"{name}：門檻 {t:.3f}｜recall {r['recall']:.3f}｜precision {r['precision']:.3f}｜"
          f"漏報 {r['漏報']}｜誤報 {r['誤報']}｜成本 {r['成本']}")


print("== 錯誤 1：在測試集上用 roc_curve 找「最佳」門檻（作弊）==")
fpr, tpr, thr = roc_curve(y, s["test"])
t_cheat = thr[np.argmax(tpr - fpr)]
r = confusion(y, s["test"], t_cheat - 1e-9)
print(f"AI 回報（全部 160 張 test）：recall {r['recall']:.3f}、precision {r['precision']:.3f}")
show("同一個門檻在最終測試上", t_cheat - 1e-9)
t_ok, _ = cost_threshold(s["calib"], s["test"][val])
show("正確：驗證集＋成本選的門檻", t_ok)

print("\n== 錯誤 2：在測試集上直接最小化成本（作弊的極致）==")
t_best, c_best = cost_threshold(sf[yf == 0], sf[yf == 1])
show("偷看答案的最低成本", t_best)
print(f"→ 誠實流程成本 {confusion(yf, sf, t_ok)['成本']}，偷看 {c_best}：差距就是「報告會比上線好看多少」")

print("\n== 錯誤 3：用 F1 選門檻（把漏報和誤報看得一樣重）==")
good, bad = s["calib"], s["test"][val]
cands = np.unique(np.concatenate([good, bad]))
yv = np.r_[np.zeros(len(good)), np.ones(len(bad))]
sv = np.r_[good, bad]
t_f1 = max(cands, key=lambda t: f1_score(yv, sv > t))
show("F1 最高的門檻（驗證集上選）", t_f1)

print("\n== 錯誤 4：驗證的 10 張瑕疵沒從測試拿掉 ==")
show("最終測試（正確，109 張瑕疵）", t_ok)
show("全部 test（含驗證 10 張）    ", t_ok, y, s["test"])
