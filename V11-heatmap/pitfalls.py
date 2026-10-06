"""V11｜讓 AI 寫，然後挑錯：每張熱圖各自正規化、argmax 的 x/y 寫反、pixel AUROC 只算瑕疵圖。"""
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from common import ROOT, mvtec_paths

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
from anomaly_lib import scores_patch  # noqa: E402

SIZE = 224


def heatmap(m28):
    return cv2.GaussianBlur(cv2.resize(m28.astype(np.float32), (SIZE, SIZE)), (0, 0), 4)


P = mvtec_paths("bottle")
s = scores_patch("bottle")
heats = np.stack([heatmap(m) for m in s["test_map"]])
masks = np.stack([np.zeros((SIZE, SIZE), bool) if p is None else
                  cv2.resize(cv2.imread(p, 0), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127
                  for p in P["mask"]])
y = P["y"]
good = np.where(y == 0)[0]

print("== 錯誤 1：每張熱圖各自縮放到 0～1 再畫「紅色＝> 0.7」 ==")
per_img = [(h - h.min()) / np.ptp(h) for h in heats[good]]
red_ai = np.mean([(h > 0.7).mean() for h in per_img])
thr = s["calib"].max()
red_ok = np.mean([(h > thr).mean() for h in heats[good]])
print(f"測試良品 {len(good)} 張：AI 版平均有 {red_ai:.1%} 的面積是紅色")
print(f"改用固定尺度（校正良品最大分數 {thr:.3f} 以上才紅）：平均 {red_ok:.1%}")

print("\n== 錯誤 2：x, y = np.unravel_index(h.argmax(), h.shape) ==")
ok = wrong = 0
for k in np.where(y == 1)[0]:
    r, c = np.unravel_index(heats[k].argmax(), heats[k].shape)   # 先列（y）後行（x）
    ok += masks[k][r, c]
    wrong += masks[k][c, r]                                     # 把 x 當列來查
print(f"正確（列＝y、行＝x）：最大點在遮罩內 {ok}/{int(y.sum())}；寫反：{wrong}/{int(y.sum())}")

print("\n== 錯誤 3：pixel AUROC 只拿瑕疵圖來算 ==")
d = y == 1
print(f"只算 {int(d.sum())} 張瑕疵圖：{roc_auc_score(masks[d].reshape(-1), heats[d].reshape(-1)):.3f}")
print(f"全部 {len(y)} 張（含良品）：{roc_auc_score(masks.reshape(-1), heats.reshape(-1)):.3f}")
