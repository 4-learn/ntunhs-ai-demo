"""V06 霍夫轉換與輪廓量測：bottle 瓶口圓（半徑、圓度）、metal_nut 輪廓面積、像素→公釐。

    python V06-hough-measure/measure.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from measure_lib import (BOTTLE_SIZE, NUT_SIZE, bottle_measure, hough_mouth, load_gray,  # noqa: E402
                         nut_measure)
from common import mvtec_paths  # noqa: E402

B = mvtec_paths("bottle")
N = mvtec_paths("metal_nut")

print("== 1. 霍夫圓：在原圖找 vs 在瓶口遮罩上找（bottle train 前 80 張）==")
rows = []
for name, on_mask in [("原圖（中值模糊）", False), ("瓶口遮罩", True)]:
    r = np.array([hough_mouth(load_gray(p, BOTTLE_SIZE), on_mask) for p in B["train"][:80]], dtype=float)
    rows.append({"在哪裡找": name, "找到": int((~np.isnan(r[:, 2])).sum()), "半徑平均": np.nanmean(r[:, 2]),
                 "半徑標準差": np.nanstd(r[:, 2]), "半徑最小": np.nanmin(r[:, 2]), "半徑最大": np.nanmax(r[:, 2])})
print(pd.DataFrame(rows).set_index("在哪裡找").round(1).to_string())

print("\n== 2. bottle 瓶口量測：良品 vs 各種瑕疵（測試集平均）==")
te = pd.DataFrame([bottle_measure(p) for p in B["test"]])
te["defect"] = B["defect"]
cols = ["hough_r", "area", "circularity", "fill"]
print(te.groupby("defect")[cols].mean().round(3).to_string())
print("各組最小圓度：", te.groupby("defect")["circularity"].min().round(3).to_dict())

print("\n== 3. metal_nut 輪廓：面積、周長、圓度（測試集平均）==")
tn = pd.DataFrame([nut_measure(p) for p in N["test"]])
tn["defect"] = N["defect"]
print(tn.groupby("defect")[["area", "perimeter", "circularity", "solidity"]].mean().round(3).to_string())

print("\n== 4. 像素不是公釐：同一張圖，縮放不同，量到的數字就不同 ==")
p = B["train"][0]
for size in [225, 450, 900]:
    g = load_gray(p, size)
    x, y, r = hough_mouth(g)
    print(f"影像 {size}×{size}：瓶口半徑 {r:6.1f} px")
SPEC_MM = 28.0  # 假設值：王課長說瓶口內徑 28 mm（MVTec 沒有提供實際尺寸）
r450 = te.loc[te.defect == "good", "hough_r"].median()
mm_per_px = SPEC_MM / (2 * r450)
print(f"假設良品內徑＝{SPEC_MM:.0f} mm、450 px 影像量到的中位半徑 {r450:.1f} px → 每像素 {mm_per_px:.4f} mm")
print(f"那 1 px 的量測誤差 ≈ {mm_per_px:.3f} mm；半徑標準差換成公釐 ≈ "
      f"{te.loc[te.defect == 'good', 'hough_r'].std() * mm_per_px:.2f} mm")
