"""P13：AI 常寫的「像素特徵＋fillna(0)」，實際跑一次看哪裡壞掉。

    python P13-pose-features/pitfalls.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"  # HGB 用 1 個執行緒：結果可重現，在忙碌的機器上反而比較快（要在 import sklearn 前設）
import numpy as np
import pandas as pd

from common import KPTS, frame_features, load_poses

df = load_poses()
st = df[df.posture == "standing"]

print("== 1. AI 的特徵：髖部 y 像素、人高像素（沒看信心）==")
ys = [f"{k}_y" for k in KPTS]
ai = pd.DataFrame({
    "hip_y_px": (df.left_hip_y + df.right_hip_y) / 2,
    "height_px": df[ys].max(axis=1) - df[ys].min(axis=1),
})
print(ai.loc[st.index].groupby(st.camera_id).median().round(1).to_string())
lying = df.posture == "lying"
print(f"躺著（只在 cam1）的髖部 y 中位數 {ai.hip_y_px[lying].median():.1f}；"
      f"cam2 站著的髖部 y 中位數 {ai.hip_y_px[st.index][st.camera_id == 'cam2'].median():.1f}")

print("\n== 2. 遺失點 (0, 0) 把「人高」灌大 ==")
has_lost = (df[[f"{k}_c" for k in KPTS]] < 0.1).any(axis=1) & (df[ys] == 0).any(axis=1)
h = ai.height_px[st.index]
print(f"站著的格：有遺失點 {has_lost[st.index].mean():.3f}；"
      f"人高中位數 有遺失點 {h[has_lost[st.index]].median():.1f}、沒有 {h[~has_lost[st.index]].median():.1f}")

print("\n== 3. fillna(0)：缺值變成「膝蓋彎成 0 度」==")
F = frame_features(df)
F0 = F.fillna(0)
cam1_st = (df.camera_id == "cam1") & (df.posture == "standing")
print(f"cam1 站著的格，knee_angle 缺值 {F.knee_angle[cam1_st].isna().mean():.3f}")
print(f"fillna(0) 後 knee_angle 中位數：{F0.knee_angle[cam1_st].median():.1f} 度"
      f"（保留 NaN 時 {F.knee_angle[cam1_st].median():.1f} 度）")
print(f"fillna(0) 後 cam1 站著被當成「膝蓋 < 45 度」的格：{(F0.knee_angle[cam1_st] < 45).mean():.3f}")

print("\n== 4. 換一支攝影機：cam1＋cam3 訓練、cam2 測試（姿勢四類，macro F1，每 2 格取 1 格）==")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

xs = [f"{k}_x" for k in KPTS]
pix = pd.concat([ai, df[xs + ys].add_suffix("_px")], axis=1)
even = (df.frame % 2 == 0).to_numpy()
tr = (df.camera_id != "cam2").to_numpy() & even
te = (df.camera_id == "cam2").to_numpy() & even
for name, X in [("像素座標＋像素高度（36 欄）", pix), ("frame_features（7 欄）", F)]:
    m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], df.posture[tr])
    f_in = f1_score(df.posture[tr], m.predict(X[tr]), average="macro")
    f_out = f1_score(df.posture[te], m.predict(X[te]), average="macro")
    print(f"{name}: 訓練攝影機 {f_in:.3f} → 沒看過的 cam2 {f_out:.3f}")
