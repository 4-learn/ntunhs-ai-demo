"""P13：17 個點 → 7 個姿勢特徵（common.frame_features），檢查「與畫面大小無關」與 NaN 的來源。

    python P13-pose-features/features.py
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")
import numpy as np
import pandas as pd

from common import FRAME_FEATURES, KPTS, MIN_KPT_CONF, frame_features, keypoints, load_poses

pd.set_option("display.width", 200)
df = load_poses()
F = frame_features(df)

print("== 1. 每一格 → 7 個特徵 ==")
print(f"poses.csv {df.shape} → 特徵 {F.shape}")
print(F.groupby(df["posture"]).median().round(2).to_string())

print("\n== 2. 與畫面大小無關：把座標放大、縮小、平移 ==")
row = df[(df.posture == "standing")].iloc[[0]]
base = frame_features(row).iloc[0]
rows = []
for name, s, dx, dy in [("原始 640x480", 1.0, 0, 0), ("縮成一半", 0.5, 0, 0),
                        ("放大 2.26 倍（照片解析度）", 2.2625, 0, 0), ("往右下平移 100", 1.0, 100, 100)]:
    r = row.copy()
    for k in KPTS:
        r[f"{k}_x"] = r[f"{k}_x"] * s + dx
        r[f"{k}_y"] = r[f"{k}_y"] * s + dy
    f = frame_features(r).iloc[0]
    hip_px = float(np.nanmean(keypoints(r)[0, [11, 12], 1]))
    rows.append({"變換": name, "髖部 y（像素）": round(hip_px, 1),
                 "7 特徵最大變化": float(np.nanmax(np.abs(f - base).to_numpy()).round(6))})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 3. 同樣是站著，三支攝影機的像素高度差很多，特徵差很少 ==")
st = df[df.posture == "standing"]
k = keypoints(st)
with np.errstate(all="ignore"):
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    height_px = np.nanmax(k[..., 1], 1) - np.nanmin(k[..., 1], 1)
    hip_px = np.nanmean(k[:, [11, 12], 1], 1)
t = pd.DataFrame({"camera": st.camera_id.to_numpy(), "人高(像素)": height_px, "髖部y(像素)": hip_px,
                  "hip_height": F.loc[st.index, "hip_height"].to_numpy(),
                  "aspect": F.loc[st.index, "aspect"].to_numpy(),
                  "torso_angle": F.loc[st.index, "torso_angle"].to_numpy()})
print(t.groupby("camera").median().round(2).to_string())

print("\n== 4. NaN 從哪裡來 ==")
raw = keypoints(df, mask_low=False)
low = raw[..., 2] < MIN_KPT_CONF
print(f"信心 < {MIN_KPT_CONF} 的點：{low.mean():.3f}（遺失＋被擋住）")
both_lost = pd.DataFrame({
    "特徵缺值比例": F[FRAME_FEATURES].isna().mean().round(3),
})
print(both_lost.T.to_string())
by_cam = F[["hip_height", "knee_angle", "hip_angle"]].isna().groupby(df["camera_id"]).mean().round(3)
print(by_cam.to_string())
cam1 = df.camera_id == "cam1"
lower_low = (raw[:, 13:17, 2] < MIN_KPT_CONF).all(1)
print(f"cam1（寢室）膝、踝 4 點全部低信心的格：{lower_low[cam1].mean():.3f}；其他攝影機：{lower_low[~cam1].mean():.3f}")

print("\n== 5. P12 的照片也算得出同樣 7 個特徵 ==")
from photo_keypoints import photo_table  # noqa: E402  （同資料夾的小工具）

P = photo_table()
PF = frame_features(P).round(2)
PF.insert(0, "人", P["photo"] + "#" + P["person"].astype(str))
PF.insert(1, "照片裡在做什麼", P["what"])
print(PF.to_string(index=False))
