"""V11｜熱圖：28×28 小格分數放大回原圖、pixel-level AUROC、熱圖最大點是否落在瑕疵遮罩內。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ROOT, mvtec_paths

sys.path.insert(0, str(ROOT / "V08-anomaly-intro"))
from anomaly_lib import scores_patch  # noqa: E402

OUT = Path("/tmp/ntunhs/V11")
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 224          # 熱圖與遮罩都對到 224×224（特徵就是從 224×224 抽的）


def heatmap(m28, size=SIZE, sigma=4):
    """28×28 小格分數 → size×size：雙線性放大，再高斯平滑（小格邊界不會一格一格的）。"""
    h = cv2.resize(m28.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(h, (0, 0), sigma)


def load_mask(path, size=SIZE):
    if path is None:
        return np.zeros((size, size), bool)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127


def overlay(img_path, h, mask, out):
    img = cv2.resize(cv2.imread(img_path), (SIZE, SIZE))
    norm = np.clip((h - h.min()) / (np.ptp(h) + 1e-9) * 255, 0, 255).astype(np.uint8)
    blend = cv2.addWeighted(img, 0.5, cv2.applyColorMap(norm, cv2.COLORMAP_JET), 0.5, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blend, cnts, -1, (255, 255, 255), 1)
    y, x = np.unravel_index(h.argmax(), h.shape)
    cv2.drawMarker(blend, (int(x), int(y)), (0, 0, 0), cv2.MARKER_CROSS, 12, 2)
    cv2.imwrite(str(out), np.hstack([img, blend]))


P = mvtec_paths("bottle")
s = scores_patch("bottle")
maps = s["test_map"]
print("== 1. 小格分數圖 → 熱圖 ==")
print(f"bottle 測試 {len(maps)} 張，每張小格分數 {maps.shape[1:]}；原圖 900×900，遮罩 900×900")
i = P["defect"].index("broken_small")
h = heatmap(maps[i])
print(f"第 {i} 張（{P['defect'][i]}）放大成 {h.shape}：每一小格約對應原圖 {900 // 28} 像素見方")

print("\n== 2. 最大點在哪裡：換回原圖座標 ==")
y, x = np.unravel_index(h.argmax(), h.shape)
mask = load_mask(P["mask"][i])
print(f"熱圖最大點 (x, y) = ({x}, {y})（224 座標）→ 原圖約 ({round(x * 900 / SIZE)}, {round(y * 900 / SIZE)})")
print(f"遮罩面積佔整張 {mask.mean():.3%}；最大點在遮罩內？{bool(mask[y, x])}")

heats = np.stack([heatmap(m) for m in maps])
masks = np.stack([load_mask(p) for p in P["mask"]])
for k, name in [(i, "broken_small"), (P["defect"].index("contamination"), "contamination"),
                (P["defect"].index("good"), "good")]:
    overlay(P["test"][k], heats[k], masks[k], OUT / f"bottle-{name}-{k}.png")

print("\n== 3. 每張瑕疵圖：熱圖最大點是否落在遮罩內 ==")
y_img = P["y"]
rows = []
for k in np.where(y_img == 1)[0]:
    yy, xx = np.unravel_index(heats[k].argmax(), heats[k].shape)
    near = cv2.dilate(masks[k].astype(np.uint8), np.ones((17, 17), np.uint8)) > 0   # 遮罩外擴 8 像素
    rows.append({"瑕疵類型": P["defect"][k], "最大點在遮罩內": masks[k][yy, xx], "8像素內": near[yy, xx],
                 "遮罩面積比": masks[k].mean()})
hit = pd.DataFrame(rows)
print(hit.groupby("瑕疵類型").agg(張數=("最大點在遮罩內", "size"), 落在遮罩內=("最大點在遮罩內", "mean"),
                                  含8像素內=("8像素內", "mean"), 平均遮罩面積比=("遮罩面積比", "mean"))
      .to_string(float_format="%.3f"))
print(f"全部 {len(hit)} 張：落在遮罩內 {hit['最大點在遮罩內'].mean():.3f}；含 8 像素內 {hit['8像素內'].mean():.3f}")

print("\n== 4. pixel-level AUROC：每個像素都是一題 ==")
px_y = masks.reshape(-1)
px_s = heats.reshape(-1)
print(f"{len(px_y)} 個像素（83 張 × 224 × 224），其中瑕疵像素 {px_y.mean():.3%}")
print(f"pixel AUROC {roc_auc_score(px_y, px_s):.3f}｜image AUROC {roc_auc_score(y_img, s['test']):.3f}")
no_blur = np.stack([cv2.resize(m.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) for m in maps])
print(f"不平滑、直接一格一格放大（最近鄰）：pixel AUROC {roc_auc_score(px_y, no_blur.reshape(-1)):.3f}")

print("\n== 5. screw：image 排序普通，定位呢？ ==")
P = mvtec_paths("screw")
s = scores_patch("screw")
heats = np.stack([heatmap(m) for m in s["test_map"]])
masks = np.stack([load_mask(p) for p in P["mask"]])
inside = [masks[k][np.unravel_index(heats[k].argmax(), heats[k].shape)] for k in np.where(P["y"] == 1)[0]]
print(f"pixel AUROC {roc_auc_score(masks.reshape(-1), heats.reshape(-1)):.3f}｜image AUROC {roc_auc_score(P['y'], s['test']):.3f}")
print(f"瑕疵圖最大點落在遮罩內 {np.mean(inside):.3f}；瑕疵像素只佔 {masks[P['y'] == 1].mean():.3%}")
for name in ["thread_top", "good"]:
    k = P["defect"].index(name)
    overlay(P["test"][k], heats[k], masks[k], OUT / f"screw-{name}-{k}.png")
print("（疊圖存在 /tmp/ntunhs/V11/：左原圖、右熱圖疊色，白線＝遮罩、黑十字＝最大點）")
