"""P07｜讓 AI 寫，然後挑錯：彩色圖直接二值化、門檻方向反了（INV）、開閉運算的核大小放反。"""
import cv2
import numpy as np

from common import PHOTOS

img = cv2.imread(str(PHOTOS / "bolts-conveyor.png"))
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
belt = np.zeros_like(gray)
cv2.fillPoly(belt, [np.array([[135, 0], [1365, 0], [1390, 1086], [45, 1086]])], 255)
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k41 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))


def n_big(bw):
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    return int((stats[1:, cv2.CC_STAT_AREA] > 2000).sum())


print("== 錯誤 1：對彩色圖做 cv2.threshold ==")
_, bw_color = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
print(f"結果 shape {bw_color.shape}：三個通道各自二值化")
colors = len(np.unique(bw_color.reshape(-1, 3), axis=0))
print(f"出現 {colors} 種「顏色」組合，不是黑白兩種")
try:
    cv2.findContours(bw_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
except cv2.error as e:
    print("findContours 直接報錯（cv2.error）：它只吃單通道的黑白圖")

print("\n== 錯誤 2：用 THRESH_BINARY_INV（以為螺栓是暗的）==")
inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 401, -30)
ok = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 401, -30)
inv, ok = cv2.bitwise_and(inv, belt), cv2.bitwise_and(ok, belt)
print(f"INV 白色占比 {(inv > 0).mean():.3f}（白的是輸送帶）；正確 {(ok > 0).mean():.3f}")
a = cv2.morphologyEx(cv2.morphologyEx(inv, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, k41)
b = cv2.morphologyEx(cv2.morphologyEx(ok, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, k41)
print(f"同樣的開閉運算後，面積 > 2000 的區塊：INV {n_big(a)} 個；正確 {n_big(b)} 個")

print("\n== 錯誤 3：開運算用 41、閉運算用 5（大小放反）==")
swap = cv2.morphologyEx(cv2.morphologyEx(ok, cv2.MORPH_OPEN, k41), cv2.MORPH_CLOSE, k5)
print(f"開 41 → 閉 5：面積 > 2000 的區塊 {n_big(swap)} 個，白色占比 {(swap > 0).mean():.3f}")
print(f"開 5 → 閉 41：面積 > 2000 的區塊 {n_big(b)} 個，白色占比 {(b > 0).mean():.3f}")
print("開運算的核比螺桿還粗，整支螺桿被當成雜點削掉")
