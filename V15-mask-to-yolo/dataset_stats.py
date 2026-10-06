"""V15｜讀教師產生好的 cache/yolo-defects/，做統計：張數、每張框數、框大小分布。

    python V15-mask-to-yolo/dataset_stats.py

不重建資料集；若 cache/yolo-defects/ 不存在，請教師先跑 tools/build_yolo_defects.py。
"""
import json
import sys

import numpy as np
import pandas as pd

from common import CACHE

DS = CACHE / "yolo-defects"
SIZE = 320
if not (DS / "data.yaml").exists():
    sys.exit("找不到 cache/yolo-defects/：請先執行 PYTHONPATH=. python tools/build_yolo_defects.py")


def read_labels(split):
    """一張圖一列（含 0 個框的良品）＋一個框一列。"""
    imgs, boxes = [], []
    for f in sorted((DS / "labels" / split).glob("*.txt")):
        cat, rest = f.stem.split("-", 1)
        defect = rest.rsplit("-", 1)[0]
        lines = [l.split() for l in f.read_text().splitlines() if l.strip()]
        imgs.append(dict(split=split, name=f.stem, cat=cat, defect=defect, n_boxes=len(lines)))
        for c, cx, cy, w, h in lines:
            boxes.append(dict(split=split, name=f.stem, cat=cat, defect=defect, cls=int(c),
                              w_px=float(w) * SIZE, h_px=float(h) * SIZE, area=float(w) * float(h)))
    return imgs, boxes


I, B = [], []
for s in ("train", "val", "test"):
    a, b = read_labels(s)
    I += a
    B += b
I, B = pd.DataFrame(I), pd.DataFrame(B)

print("== 1. data.yaml ==")
print((DS / "data.yaml").read_text().replace(str(DS), "cache/yolo-defects").rstrip())

print("\n== 2. 每個 split 的張數與框數 ==")
t = I.assign(瑕疵圖=I.defect != "good").groupby("split", sort=False).agg(
    張數=("name", "size"), 瑕疵圖=("瑕疵圖", "sum"), 框=("n_boxes", "sum"))
print(t.to_string())
summary = json.loads((DS / "summary.json").read_text())
ok = all(summary[s]["images"] == t.loc[s, "張數"] and summary[s]["boxes"] == t.loc[s, "框"] for s in t.index)
print("與 summary.json 一致：", ok)
print("影像檔數 = label 檔數：",
      all(len(list((DS / "images" / s).glob("*.png"))) == len(list((DS / "labels" / s).glob("*.txt")))
          for s in ("train", "val", "test")))

print("\n== 3. 每張圖幾個框 ==")
c = pd.crosstab(I.cat, I.n_boxes)
c.columns = [f"{k} 框" for k in c.columns]
print(c.to_string())
bad = I[(I.defect != "good") & (I.n_boxes == 0)]
print(f"瑕疵圖卻沒有框：{len(bad)} 張；良品有框：{int(((I.defect == 'good') & (I.n_boxes > 0)).sum())} 張")

print("\n== 4. 框的大小（320×320 影像上的像素；面積占整張的比例）==")
g = B.groupby(["cat", "defect"]).agg(框=("area", "size"), 寬中位數=("w_px", "median"), 高中位數=("h_px", "median"),
                                     面積中位數=("area", "median"), 面積最小=("area", "min"))
print(g.round(3).to_string())

print("\n== 5. 框寬（像素）分布：YOLO 看得到嗎 ==")
side = np.sqrt(B.w_px * B.h_px)
bins = [0, 8, 16, 32, 64, 400]
lab = ["<8", "8–16", "16–32", "32–64", "≥64"]
d = pd.crosstab(B.cat, pd.cut(side, bins, labels=lab, right=False).rename("框邊長"))
print(d.to_string())
print(f"screw 的框邊長中位數 {np.median(side[B.cat == 'screw']):.1f} 像素；"
      f"metal_nut {np.median(side[B.cat == 'metal_nut']):.1f} 像素")
