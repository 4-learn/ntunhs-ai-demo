"""V08｜讓 AI 寫，然後挑錯：AUROC 餵 0/1 預測、標籤方向反了、uint8 直接相減。"""
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from common import memory_calib_split, mvtec_paths


def gray(path, size=128):
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32) / 255


P = mvtec_paths("bottle")
memory, calib = memory_calib_split(P["train"])
mean_img = np.mean([gray(p) for p in memory], axis=0)
score = lambda ps: np.array([np.abs(gray(p) - mean_img).mean() for p in ps])
s_cal, s_test = score(calib), score(P["test"])
y = P["y"]

print("== 錯誤 1：AUROC 餵的是 0/1 預測，不是分數 ==")
pred = s_test > s_cal.max()
print(f"roc_auc_score(y, pred)  = {roc_auc_score(y, pred):.3f}  ← 只剩一個門檻，看不到排序")
print(f"roc_auc_score(y, score) = {roc_auc_score(y, s_test):.3f}")

print("\n== 錯誤 2：標籤寫成「良品＝1」 ==")
y_ai = np.array([1 if "/good/" in p else 0 for p in P["test"]])
print(f"良品當正類：AUROC {roc_auc_score(y_ai, s_test):.3f}（＝1 − {roc_auc_score(y, s_test):.3f}）")

print("\n== 錯誤 3：uint8 直接相減 ==")
a = np.asarray(Image.open(P["test"][0]).convert("L").resize((128, 128)))
b = np.asarray(Image.open(memory[0]).convert("L").resize((128, 128)))
print(f"a - b（uint8）：平均 {(a - b).mean():.3f}、最大 {(a - b).max()}  ← 負數繞回 255 附近")
d = np.abs(a.astype(np.float32) - b)
print(f"先轉 float 再取絕對值：平均 {d.mean():.3f}、最大 {d.max():.0f}")
