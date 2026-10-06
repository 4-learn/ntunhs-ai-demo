"""把 MVTec 的瑕疵遮罩轉成 YOLO 偵測資料集（V15 講解、V16 訓練用）。

    PYTHONPATH=. python tools/build_yolo_defects.py

- 類別：metal_nut、screw 兩類產品的瑕疵合成一類 `defect`（class 0）。
- 只用 MVTec 的 test 資料（train 只有良品、沒有瑕疵）；依影像切 train/val/test（60/20/20，seed 0，分層：每個瑕疵類型各自切）。
- 良品也放進去（沒有框的影像＝空的 label 檔），讓模型學會「沒有瑕疵」。
- 影像縮到 320×320，存到 cache/yolo-defects/（不隨 repo 散佈，因為 MVTec 是 CC BY-NC-SA）。
"""
import json

import cv2
import numpy as np

import common as C

OUT = C.CACHE / "yolo-defects"
SIZE = 320
CATS = ["metal_nut", "screw"]
MIN_AREA = 0.0005  # 小於影像面積 0.05% 的連通塊視為雜點


def boxes_from_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = m.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8))
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA * h * w:
            continue
        out.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return out


def main():
    rng = np.random.default_rng(0)
    summary = {}
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for cat in CATS:
        P = C.mvtec_paths(cat)
        groups = {}
        for i, d in enumerate(P["defect"]):
            groups.setdefault(d, []).append(i)
        for d, idx in groups.items():
            idx = list(rng.permutation(idx))
            n = len(idx)
            a, b = int(round(n * 0.6)), int(round(n * 0.8))
            for split, part in (("train", idx[:a]), ("val", idx[a:b]), ("test", idx[b:])):
                for i in part:
                    name = f"{cat}-{d}-{i:03d}"
                    img = cv2.resize(cv2.imread(P["test"][i]), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(OUT / "images" / split / f"{name}.png"), img)
                    bxs = boxes_from_mask(P["mask"][i]) if P["mask"][i] else []
                    with open(OUT / "labels" / split / f"{name}.txt", "w") as f:
                        for cx, cy, bw, bh in bxs:
                            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    s = summary.setdefault(split, dict(images=0, defect_images=0, boxes=0))
                    s["images"] += 1
                    s["defect_images"] += int(d != "good")
                    s["boxes"] += len(bxs)
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
