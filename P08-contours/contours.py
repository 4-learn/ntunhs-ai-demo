"""P08｜輪廓與量測：findContours、面積過濾、minAreaRect 長寬比、中心線偏移找彎曲螺栓。"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import PHOTOS

OUT = Path("/tmp/ntunhs/P08")
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
belt = np.zeros_like(gray)
cv2.fillPoly(belt, [np.array([[135, 0], [1365, 0], [1390, 1086], [45, 1086]])], 255)


def segment(g):
    """P07 的流程：adaptive 401 / C=-30 → 輸送帶遮罩 → 開 5 → 閉 41。"""
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, -30)
    bw = cv2.bitwise_and(bw, belt)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k41 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    return cv2.morphologyEx(cv2.morphologyEx(bw, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, k41)


mask = segment(gray)

print("== 1. findContours：把每塊白色的邊界描出來 ==")
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
all_list, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(f"RETR_EXTERNAL（只要最外圈）：{len(contours)} 個輪廓")
print(f"RETR_LIST（連洞的邊也要）  ：{len(all_list)} 個輪廓（閉運算已經把洞補起來，所以一樣多）")
c0 = max(contours, key=cv2.contourArea)
print(f"一個輪廓就是一串點：shape {c0.shape}，前 3 點 {c0[:3, 0].tolist()}")

print("\n== 2. 面積過濾：太小的不是螺栓 ==")
areas = np.array(sorted((cv2.contourArea(c) for c in contours), reverse=True))
print("面積由大到小：", [int(a) for a in areas])
bolts = [c for c in contours if cv2.contourArea(c) > 2000]
print(f"面積 > 2000 的輪廓：{len(bolts)} 個 ← 這就是「數到的螺栓數」")


def order(c):                                     # 由上到下分 3 列、每列由左到右，方便對照照片
    x, y, w, h = cv2.boundingRect(c)
    return (y + h // 2) // 300, x + w // 2


bolts.sort(key=order)

print("\n== 3. minAreaRect：可以旋轉的最小外框 ==")


def centerline_dev(c, rect):
    """沿外框長軸切 10 段，每段算白色像素的中心；扣掉直線後，中間 6 段的最大偏移（像素）。"""
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    long = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    u = long / np.linalg.norm(long)               # 長軸方向
    v = np.array([-u[1], u[0]])                   # 垂直方向
    m = np.zeros_like(gray)
    cv2.drawContours(m, [c], -1, 255, -1)
    ys, xs = np.nonzero(m)
    p = np.stack([xs, ys], 1) - np.array(rect[0])
    t, d = p @ u, p @ v
    edges = np.linspace(t.min(), t.max(), 11)
    tc = (edges[:-1] + edges[1:]) / 2
    cen = np.array([d[(t >= edges[i]) & (t <= edges[i + 1])].mean() for i in range(10)])
    mid = slice(2, 8)                             # 頭和尾形狀本來就不對稱，只看中間 6 段
    line = np.polyfit(tc[mid], cen[mid], 1)
    return float(np.abs(cen[mid] - np.polyval(line, tc[mid])).max())


rows = []
for i, c in enumerate(bolts, 1):
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), _ = rect
    long, short = max(w, h), min(w, h)
    area = cv2.contourArea(c)
    rows.append({"編號": i, "中心": f"({cx:.0f},{cy:.0f})", "面積": int(area), "長": round(long), "寬": round(short),
                 "長寬比": round(long / short, 2), "填滿率": round(area / (w * h), 2),
                 "中心線偏移": round(centerline_dev(c, rect), 1)})
df = pd.DataFrame(rows)
print(df.drop(columns=["中心線偏移"]).to_string(index=False))

print("\n== 4. 只用長寬比，挑得出彎的那支嗎？ ==")
print(df.sort_values("長寬比")[["編號", "中心", "長寬比"]].to_string(index=False))
print("彎曲螺栓是第 3 支 (828,275)，長寬比排在中間：彎了之後外框變寬、長寬比反而接近正常")

print("\n== 5. 面積：一支螺栓該多大？ ==")
med = df["面積"].median()
df["面積/中位數"] = (df["面積"] / med).round(2)
print(f"面積中位數 {med:.0f}")
print(df[["編號", "中心", "面積", "面積/中位數"]].to_string(index=False))
odd = df[(df["面積/中位數"] > 2) | (df["面積/中位數"] < 0.5)]
print("面積異常（> 2 倍或 < 0.5 倍中位數）：", odd["編號"].tolist(), "← 第 1 支是螺栓＋左上角反光黏在一起")

print("\n== 6. 中心線偏移：直的螺栓中心線是一直線 ==")
print(df[["編號", "中心", "長寬比", "中心線偏移"]].to_string(index=False))
bent = df[df["中心線偏移"] > 15]
print("中心線偏移 > 15 像素：", bent["編號"].tolist())

print("\n== 7. 報告 ==")
ok = df[(df["面積/中位數"] <= 2) & (df["面積/中位數"] >= 0.5)]
print(f"輪廓數 {len(df)}；面積正常 {len(ok)} 個；面積異常 {len(odd)} 個（需人工看）")
print(f"疑似彎曲：{len(bent)} 支，位置 {bent['中心'].tolist()}")
vis = img.copy()
for (_, r), c in zip(df.iterrows(), bolts):
    color = (0, 0, 255) if r["中心線偏移"] > 15 else (0, 200, 255) if r["編號"] in odd["編號"].tolist() else (0, 255, 0)
    cv2.drawContours(vis, [np.int32(cv2.boxPoints(cv2.minAreaRect(c)))], -1, color, 4)
    cx, cy = cv2.minAreaRect(c)[0]
    cv2.putText(vis, str(r["編號"]), (int(cx) - 15, int(cy) + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
cv2.imwrite(str(OUT / "report.png"), vis)
print(f"標註圖：{OUT / 'report.png'}（紅＝疑似彎曲、橘＝面積異常、綠＝正常）")
