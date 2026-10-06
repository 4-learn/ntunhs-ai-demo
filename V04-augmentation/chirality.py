"""metal_nut 的「左右手」：沿外輪廓繞一圈，看半徑是「慢慢升、突然掉」還是「突然升、慢慢掉」。

四個爪子是同一個旋轉方向的鉤狀；翻面（MVTec 的 flip 瑕疵）或影像水平翻轉，方向就反過來。
"""
import cv2
import numpy as np


def handedness(gray, size=350):
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    _, th = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnt = max(cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    m = cv2.moments(th, True)
    d = cnt[:, 0, :].astype(float) - (m["m10"] / m["m00"], m["m01"] / m["m00"])
    ang, r = np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])
    idx = np.clip(((ang + np.pi) / (2 * np.pi) * 360).astype(int), 0, 359)
    prof = np.zeros(360)
    np.maximum.at(prof, idx, r)
    prof[prof == 0] = prof[prof > 0].mean()
    dr = np.diff(np.r_[prof, prof[0]])
    return float((dr ** 3).mean() / (dr ** 2).mean() ** 1.5)  # 半徑變化的偏態：正＝右旋、負＝左旋
