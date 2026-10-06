"""P08｜讓 AI 寫，然後挑錯：沒過濾面積就數、w/h 直接相除、用 boundingRect 算長寬比、RETR_LIST 數到洞。"""
import cv2
import numpy as np

from common import PHOTOS

gray = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(str(PHOTOS / "bolts-conveyor.png")), cv2.COLOR_BGR2GRAY), (5, 5), 0)
belt = np.zeros_like(gray)
cv2.fillPoly(belt, [np.array([[135, 0], [1365, 0], [1390, 1086], [45, 1086]])], 255)
bw = cv2.bitwise_and(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, -30), belt)
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k41 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k5)
mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k41)

print("== 錯誤 1：findContours 之後直接 len() ==")
cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
big = [c for c in cs if cv2.contourArea(c) > 2000]
print(f"len(contours) = {len(cs)}；面積 > 2000 才是 {len(big)}")

print("\n== 錯誤 2：沒做閉運算、又用 RETR_LIST ==")
raw_list, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
raw_ext, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"只開不閉 + RETR_LIST    ：{len(raw_list)} 個輪廓，面積 > 2000 的 {sum(cv2.contourArea(c) > 2000 for c in raw_list)} 個")
print(f"只開不閉 + RETR_EXTERNAL：{len(raw_ext)} 個輪廓，面積 > 2000 的 {sum(cv2.contourArea(c) > 2000 for c in raw_ext)} 個")
print("螺紋的暗溝在螺栓裡留下洞，RETR_LIST 把洞的邊也當成輪廓")

print("\n== 錯誤 3：長寬比寫成 w / h ==")
ratios = []
for c in big:
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    ratios.append(w / h)
ratios = np.array(ratios)
print("w / h：", np.round(np.sort(ratios), 2).tolist())
print(f"有 {(ratios < 1).sum()} 支小於 1：minAreaRect 不保證 w 是長邊")
print("max/min：", np.round(np.sort([max(cv2.minAreaRect(c)[1]) / min(cv2.minAreaRect(c)[1]) for c in big]), 2).tolist())

print("\n== 錯誤 4：用 boundingRect（不會轉）算長寬比 ==")
br = []
for c in big:
    x, y, w, h = cv2.boundingRect(c)
    br.append(max(w, h) / min(w, h))
mr = [max(cv2.minAreaRect(c)[1]) / min(cv2.minAreaRect(c)[1]) for c in big]
print("boundingRect：", np.round(br, 2).tolist())
print("minAreaRect ：", np.round(mr, 2).tolist())
print("斜放的螺栓，正立的外框接近正方形；長寬比跟著擺放角度變，不是螺栓本身的形狀")
