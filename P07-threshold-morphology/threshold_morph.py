"""P07｜二值化與形態學：固定門檻 vs Otsu vs adaptive（打光不均），侵蝕、膨脹、開、閉運算。"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import PHOTOS

OUT = Path("/tmp/ntunhs/P07")
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)          # P06：先輕微去雜訊
# 教師目視：12 支螺栓的大約中心 (x, y)，用來檢查「每支有沒有被抓到」
BOLTS = [(560, 257), (827, 275), (1142, 250), (225, 250), (183, 543), (539, 582),
         (876, 583), (1191, 563), (180, 851), (520, 859), (878, 861), (1208, 856)]
# 輸送帶範圍（左右兩側金屬護欄不要）
BELT = np.array([[135, 0], [1365, 0], [1390, 1086], [45, 1086]])
belt = np.zeros_like(gray)
cv2.fillPoly(belt, [BELT], 255)


def check(bw):
    """白色占比、白色區塊數、面積 > 2000 的區塊數，以及「單獨成塊」的螺栓數：
    螺栓中心落在一個區塊上、那個區塊只包含這一支、面積在 4000–25000（一支螺栓的大小）。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    areas = stats[:, cv2.CC_STAT_AREA]
    owner = []
    for x, y in BOLTS:
        win = labels[y - 15:y + 16, x - 15:x + 16]
        ids = win[win > 0]
        owner.append(np.bincount(ids).argmax() if ids.size else 0)
    alone = sum(1 for i in owner if i and owner.count(i) == 1 and 4000 <= areas[i] <= 25000)
    found = sum(1 for i in owner if i)
    return dict(白色占比=round(float((bw > 0).mean()), 3), 區塊數=n - 1,
                大於2000=int((areas[1:] > 2000).sum()), 碰到螺栓=found, 單獨成塊=alone)


print("== 1. 打光不均：左右亮度差多少 ==")
w = gray.shape[1]
for name, xs in {"左（x 150–450）": (150, 450), "中（x 550–850）": (550, 850), "右（x 1000–1300）": (1000, 1300)}.items():
    region = blur[:, xs[0]:xs[1]][belt[:, xs[0]:xs[1]] > 0]
    print(f"{name}  輸送帶 50% 分位 {np.percentile(region, 50):5.1f}  90% 分位 {np.percentile(region, 90):5.1f}")

print("\n== 2. 固定門檻：一個數字切全圖 ==")
rows = []
for t in [60, 100, 140]:
    _, bw = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
    rows.append({"方法": f"固定 {t}", **check(cv2.bitwise_and(bw, belt))})
    cv2.imwrite(str(OUT / f"fixed-{t}.png"), bw)

t_otsu, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
rows.append({"方法": f"Otsu（自動={t_otsu:.0f}）", **check(cv2.bitwise_and(bw, belt))})
cv2.imwrite(str(OUT / "otsu.png"), bw)

for bs, c in [(51, -10), (401, -30)]:
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, c)
    rows.append({"方法": f"adaptive {bs}, C={c}", **check(cv2.bitwise_and(bw, belt))})
    cv2.imwrite(str(OUT / f"adaptive-{bs}.png"), bw)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 3. Otsu 選的門檻在左右各代表什麼 ==")
for name, xs in {"左": (150, 450), "右": (1000, 1300)}.items():
    region = blur[:, xs[0]:xs[1]][belt[:, xs[0]:xs[1]] > 0]
    print(f"{name}：輸送帶像素中有 {(region > t_otsu).mean():.3f} 高於 Otsu 門檻 {t_otsu:.0f}")
print("Otsu 只看整張的直方圖，不知道「哪裡」亮")

print("\n== 4. adaptive：每個像素和自己附近比 ==")
y, x = 300, 300
local = blur[y - 200:y + 201, x - 200:x + 201].mean()
print(f"左上 (x={x}, y={y})：附近 401×401 平均 {local:.1f} → 門檻 {local + 30:.1f}")
y, x = 300, 1250
local = blur[y - 200:y + 201, x - 200:x + 201].mean()
print(f"右上 (x={x}, y={y})：附近 401×401 平均 {local:.1f} → 門檻 {local + 30:.1f}")
print("C=-30 的意思：比附近平均亮 30 以上才算白")

print("\n== 5. 形態學：用一個小形狀去「掃」白色區域 ==")
bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, -30)
bw = cv2.bitwise_and(bw, belt)
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k41 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
steps = {
    "adaptive 原始": bw,
    "侵蝕 5×5": cv2.erode(bw, k5),
    "膨脹 5×5": cv2.dilate(bw, k5),
    "開運算 5×5": cv2.morphologyEx(bw, cv2.MORPH_OPEN, k5),
    "閉運算 41×41": cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k41),
}
opened = steps["開運算 5×5"]
steps["先開 5 再閉 41"] = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k41)
steps["先閉 41 再開 5"] = cv2.morphologyEx(steps["閉運算 41×41"], cv2.MORPH_OPEN, k5)
rows = [{"步驟": name, **check(im)} for name, im in steps.items()]
print(pd.DataFrame(rows).to_string(index=False))
for name, im in steps.items():
    cv2.imwrite(str(OUT / f"morph-{name.replace(' ', '_')}.png"), im)

print("\n== 6. 開運算 = 先侵蝕再膨脹；閉運算 = 先膨脹再侵蝕 ==")
print("開運算 == dilate(erode(bw)):", np.array_equal(steps["開運算 5×5"], cv2.dilate(cv2.erode(bw, k5), k5)))
print("閉運算 == erode(dilate(bw)):", np.array_equal(steps["閉運算 41×41"], cv2.erode(cv2.dilate(bw, k41), k41)))
final = steps["先開 5 再閉 41"]
cv2.imwrite(str(OUT / "final-mask.png"), final)
print(f"最終遮罩存到 {OUT / 'final-mask.png'}（P08 從這裡接著數）")
