"""V06 量測函式：瓶口（bottle）與螺帽（metal_nut）。V07 會 import 這支來寫規則。"""
import cv2
import numpy as np

BOTTLE_SIZE = 450  # 原圖 900×900 縮一半
NUT_SIZE = 350     # 原圖 700×700 縮一半
MOUTH_T = 46       # 瓶口內部很暗（中心亮度約 35–45），比它亮一點就算瓶壁


def load_gray(path, size):
    return cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (size, size), interpolation=cv2.INTER_AREA)


def mouth_mask(g):
    """瓶口內部的暗區：固定門檻二值化＋開運算，取覆蓋畫面中央最多的那一塊。"""
    b = cv2.GaussianBlur(g, (9, 9), 0)
    _, th = cv2.threshold(b, MOUTH_T, 255, cv2.THRESH_BINARY_INV)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, _, _ = cv2.connectedComponentsWithStats(th)
    s, w = g.shape[0] // 2, g.shape[0] * 60 // BOTTLE_SIZE
    ids, cnt = np.unique(lab[s - w:s + w, s - w:s + w], return_counts=True)
    cnt[ids == 0] = 0
    if cnt.max() == 0:
        return np.zeros_like(g)
    return (lab == ids[cnt.argmax()]).astype(np.uint8) * 255


def hough_mouth(g, on_mask=True):
    """霍夫圓：在「瓶口遮罩」上找一個圓（在原圖上找會被一圈圈的瓶身干擾）。回傳 (x, y, r) 或 None。"""
    src = cv2.GaussianBlur(mouth_mask(g), (5, 5), 0) if on_mask else cv2.medianBlur(g, 5)
    k = g.shape[0] / BOTTLE_SIZE  # 半徑範圍跟著影像大小縮放
    c = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, dp=1, minDist=g.shape[0], param1=100, param2=30,
                         minRadius=int(60 * k), maxRadius=int(140 * k))
    return None if c is None else tuple(float(v) for v in c[0, 0])


def shape_stats(mask):
    """最大輪廓的面積、周長、圓度 4πA/P²、填滿率（面積／最小外接圓面積）、凸包比（solidity）。"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return dict(area=0.0, perimeter=0.0, circularity=0.0, fill=0.0, solidity=0.0)
    c = max(cs, key=cv2.contourArea)
    a, p = cv2.contourArea(c), cv2.arcLength(c, True)
    (_, _), r = cv2.minEnclosingCircle(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    return dict(area=a, perimeter=p, circularity=4 * np.pi * a / max(p, 1) ** 2,
                fill=a / (np.pi * r * r), solidity=a / max(hull, 1))


def bottle_measure(path):
    g = load_gray(path, BOTTLE_SIZE)
    s = shape_stats(mouth_mask(g))
    h = hough_mouth(g)
    s["hough_r"] = h[2] if h else np.nan
    return s


def nut_mask(g):
    """螺帽比背景亮：Otsu 二值化，填滿內孔，留最大的一塊。"""
    _, th = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(g)
    cv2.drawContours(m, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    return m


def nut_measure(path):
    return shape_stats(nut_mask(load_gray(path, NUT_SIZE)))
