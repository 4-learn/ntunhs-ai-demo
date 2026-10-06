"""V15｜一張遮罩怎麼變成 YOLO 框：連通塊、MIN_AREA 雜點、YOLO 格式。

    python V15-mask-to-yolo/mask_to_box.py

只示範與核對，不重建資料集（資料集由教師用 tools/build_yolo_defects.py 產生在 cache/yolo-defects/）。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from common import CACHE, ROOT, mvtec_paths

sys.path.insert(0, str(ROOT / "tools"))
from build_yolo_defects import CATS, MIN_AREA, SIZE, boxes_from_mask  # noqa: E402

OUT = Path("/tmp/ntunhs/V15")
OUT.mkdir(parents=True, exist_ok=True)
DS = CACHE / "yolo-defects"

P = mvtec_paths("screw")
i = next(k for k, d in enumerate(P["defect"]) if d == "thread_side")
m = cv2.imread(P["mask"][i], cv2.IMREAD_GRAYSCALE)
print("== 1. 遮罩：一張和原圖一樣大的黑白圖 ==")
print(f"screw/thread_side/{Path(P['test'][i]).name}：遮罩 {m.shape[1]}×{m.shape[0]}，"
      f"值只有 {sorted(np.unique(m).tolist())}，白色像素 {int((m > 127).sum())} 個"
      f"（{(m > 127).mean():.2%}）")

print("\n== 2. 連通塊 → 外接矩形 ==")
n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8))
print(f"connectedComponentsWithStats：{n - 1} 個連通塊（0 號是背景）")
for k in range(1, n):
    x, y, w, h, area = stats[k]
    print(f"  塊 {k}：x={x} y={y} 寬={w} 高={h} 面積={area}")
print("YOLO 格式（類別 中心x 中心y 寬 高，除以影像寬高）：")
for cx, cy, bw, bh in boxes_from_mask(P["mask"][i]):
    print(f"  0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

print("\n== 3. MIN_AREA：多小的塊算雜點 ==")
h, w = m.shape
print(f"MIN_AREA = {MIN_AREA}（影像面積的 {MIN_AREA:.2%}）→ 1024×1024 的遮罩約 {MIN_AREA * 1024 * 1024:.0f} 像素")
tot = small = 0
small_imgs = set()
for cat in CATS:
    Q = mvtec_paths(cat)
    for p in Q["mask"]:
        if p is None:
            continue
        mm = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        n2, _, st, _ = cv2.connectedComponentsWithStats((mm > 127).astype(np.uint8))
        for k in range(1, n2):
            tot += 1
            if st[k, 4] < MIN_AREA * mm.size:
                small += 1
                small_imgs.add(p)
print(f"metal_nut＋screw 全部遮罩：連通塊 {tot} 個，小於 MIN_AREA 被丟掉 {small} 個（在 {len(small_imgs)} 張圖上）")

print("\n== 4. 核對：自己從遮罩算的框 = 資料集裡的 label 檔 ==")
lbl = next((DS / "labels").glob(f"*/screw-thread_side-{i:03d}.txt"))
print(f"{lbl.parent.name}/{lbl.name}：")
print(lbl.read_text().rstrip())
good = sorted((DS / "labels" / "train").glob("screw-good-*.txt"))[0]
print(f"良品 {good.name}：{good.stat().st_size} bytes（空檔＝這張圖沒有瑕疵）")

img = cv2.imread(str(DS / "images" / lbl.parent.name / (lbl.stem + ".png")))
for line in lbl.read_text().splitlines():
    _, cx, cy, bw, bh = map(float, line.split())
    x1, y1 = int((cx - bw / 2) * SIZE), int((cy - bh / 2) * SIZE)
    x2, y2 = int((cx + bw / 2) * SIZE), int((cy + bh / 2) * SIZE)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
cv2.imwrite(str(OUT / "thread_side_box.png"), img)
