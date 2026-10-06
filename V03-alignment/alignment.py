"""V03 幾何校正：用 PCA 求螺絲主軸角度、旋轉轉正，看「良品和良品」的像素差變多小。

    python V03-alignment/alignment.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from align_lib import SIZE, align, load_gray, normalize_light, part_mask, principal_axis  # noqa: E402
from common import memory_calib_split, mvtec_paths  # noqa: E402

OUT = Path("/tmp/ntunhs")
OUT.mkdir(parents=True, exist_ok=True)
P = mvtec_paths("screw")
train = [load_gray(p) for p in P["train"]]

print("== 1. 前 5 張良品螺絲的主軸角度 ==")
rows = []
for i, g in enumerate(train[:5]):
    m = part_mask(g)
    ang, c, skew = principal_axis(m)
    cnt = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    (_, _), (w, h), ra = cv2.minAreaRect(cnt)
    rows.append({"圖": i, "PCA 角度": ang, "minAreaRect 角度": ra, "外框寬": w, "外框高": h,
                 "要再翻 180 度": "是" if skew > 0 else "否"})
print(pd.DataFrame(rows).set_index("圖").round(1).to_string())

print("\n== 2. 驗證：把第 0 張轉已知角度，看估得準不準 ==")
base = principal_axis(part_mask(train[0]))[0]
for turn in [15, 45, 90, 135]:
    M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), turn, 1.0)
    r = cv2.warpAffine(train[0], M, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
    est = principal_axis(part_mask(r))[0]
    diff = (base - est) % 180  # 主軸沒有方向，只看 0–180
    print(f"轉 {turn:>3} 度 → 估計轉了 {diff:6.1f} 度")

print("\n== 3. 校正前後：良品之間的像素差（320 張 train 良品）==")
variants = {
    "原圖": train,
    "只轉主軸（頭可能朝左或朝右）": [align(g, fix_head=False) for g in train],
    "轉主軸＋頭一律朝右": [align(g) for g in train],
}
rows = []
for name, imgs in variants.items():
    X = np.array(imgs, dtype=float)
    mean = X.mean(0)
    rows.append({"做法": name, "與平均良品的像素差": np.abs(X - mean).mean(),
                 "逐像素標準差（中位數）": np.median(X.std(0))})
    cv2.imwrite(str(OUT / f"V03-mean-{len(rows)}.png"), mean.astype(np.uint8))
print(pd.DataFrame(rows).set_index("做法").round(3).to_string())

print("\n== 4. 預告 V08：同樣的像素差，拿來分良品與瑕疵 ==")
mem, cal = memory_calib_split(P["train"])
test = [load_gray(p) for p in P["test"]]
mem_g, cal_g = [load_gray(p) for p in mem], [load_gray(p) for p in cal]
y = P["y"]


def mean_abs(X, mean, sd):
    return np.abs(X - mean).mean((1, 2))


def local_z(X, mean, sd):
    """每個像素用「良品在這個位置的變化幅度」標準化，模糊後取最大值：只看最不正常的一小塊。"""
    z = np.abs(X - mean) / (sd + 5)
    return np.array([cv2.GaussianBlur(d, (0, 0), 4).max() for d in z])


rows = []
for prep, f in [("原圖", lambda g: g), ("轉正", align), ("亮度校正＋轉正", lambda g: align(normalize_light(g)))]:
    M, C, T = (np.array([f(g) for g in L], dtype=float) for L in (mem_g, cal_g, test))
    mean, sd = M.mean(0), M.std(0)
    for score_name, score in [("整張平均差", mean_abs), ("逐像素 z 最大值", local_z)]:
        s_cal, s_test = score(C, mean, sd), score(T, mean, sd)
        pred = s_test > s_cal.max()  # 門檻＝校正良品最大值（V08 的規則）
        rows.append({"前處理": prep, "分數": score_name, "AUROC": roc_auc_score(y, s_test),
                     "recall": pred[y == 1].mean(), "誤報": int(pred[y == 0].sum())})
print(pd.DataFrame(rows).set_index(["前處理", "分數"]).round(3).to_string())

print("\n== 5. 為什麼「轉正」反而變差：測試良品拍得比較暗 ==")
bg = lambda imgs: np.median([np.median(g[part_mask(g) == 0]) for g in imgs])  # noqa: E731
print(f"背景亮度中位數：train 良品 {bg(train):.0f}、test 良品 {bg([g for g, t in zip(test, y) if t == 0]):.0f}、"
      f"test 瑕疵 {bg([g for g, t in zip(test, y) if t == 1]):.0f}")

print("\n== 6. 挑錯：直接拿 minAreaRect 的角度去轉 ==")


def naive_align(g):
    cnt = max(cv2.findContours(part_mask(g), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    (cx, cy), _, ang = cv2.minAreaRect(cnt)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    return cv2.warpAffine(g, M, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)


def tilt(g):
    a = principal_axis(part_mask(g))[0] % 180
    return min(a, 180 - a)  # 與水平線的夾角，0＝水平


for name, f in [("minAreaRect 角度直接轉", naive_align), ("align()", align)]:
    t = np.array([tilt(f(g)) for g in train])
    print(f"{name:<18} 轉完後仍歪超過 10 度：{(t > 10).mean():.3f}；直立（> 80 度）：{(t > 80).mean():.3f}")
