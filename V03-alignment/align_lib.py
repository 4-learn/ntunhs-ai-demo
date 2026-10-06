"""V03 共用：把 MVTec screw 轉正（V03 講解；V07、V08 之後也可以 import）。"""
import cv2
import numpy as np

SIZE = 256


def load_gray(path, size=SIZE):
    return cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (size, size), interpolation=cv2.INTER_AREA)


def part_mask(g):
    """螺絲比背景暗：Otsu 反向二值化，留最大的一塊。"""
    _, b = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    k = 1 + st[1:, cv2.CC_STAT_AREA].argmax()
    return (lab == k).astype(np.uint8) * 255


def principal_axis(mask):
    """PCA：前景像素座標的共變異數矩陣，最大特徵值的方向＝主軸。回傳 (角度, 重心, 偏態)。"""
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys].astype(float)
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    ax = v[:, 1]  # 最大特徵值對應的向量
    proj = (pts - c) @ ax
    skew = (proj ** 3).mean()  # 沿主軸，哪一頭比較「重」（螺絲頭比較寬）
    return np.degrees(np.arctan2(ax[1], ax[0])), c, skew


def align(g, fix_head=True):
    """旋轉到主軸水平、重心移到畫面中央；fix_head=True 時讓螺絲頭一律朝右。"""
    ang, c, skew = principal_axis(part_mask(g))
    if fix_head and skew > 0:
        ang += 180
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
    M[:, 2] += (SIZE / 2 - c[0], SIZE / 2 - c[1])
    return cv2.warpAffine(g, M, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)


def normalize_light(g, target=190):
    """背景（非螺絲）亮度中位數調到同一個值：V01 的打光問題在這裡又出現。"""
    bg = np.median(g[part_mask(g) == 0])
    return np.clip(g.astype(float) * target / bg, 0, 255).astype(np.uint8)
