"""P15：AI 常寫的窗口特徵——沒分序列、偷看未來、用格子當評估單位。實際跑一次。

    python P15-pose-window/pitfalls.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common import frame_features, load_poses, subject_split, window_features

df = load_poses().sort_values(["seq_id", "frame"])
train, _ = subject_split(df)
F = frame_features(train)

print("== 1. 沒有 groupby(seq_id)：窗口跨到上一段序列 ==")
ai = F.torso_angle - F.torso_angle.shift(9)          # AI 版：整張表一起 shift
ok = window_features(train).d_torso_angle             # common 版：同一段序列內
first = train.frame < 9
print(f"每段前 9 格：AI 版有值 {ai[first].notna().mean():.3f}、common 版有值 {ok[first].notna().mean():.3f}")
big = (ai[first].abs() > 45)
print(f"每段前 9 格裡，AI 版「1 秒內軀幹轉了 45 度以上」：{int(big.sum())} 格"
      f"（其實是上一段的結尾接到這一段的開頭）")
lie_first = first & (train.activity == "lie_bed")
print(f"其中 lie_bed（一直躺著）的格：{int((big & lie_first).sum())}")

print("\n== 2. shift(-9)：用「未來 1 秒」當特徵 ==")
fut = (F.groupby(train.seq_id).torso_angle.shift(-9) - F.torso_angle).rename("future_d_torso")
target = ((train.is_fall_seq == 1) & (train.frame >= train.fall_frame)).astype(int)
fit, valid = train.subject_id <= "S14", train.subject_id >= "S15"
for name, X in [("過去 1 秒（可上線）", window_features(train)), ("過去＋未來 1 秒（偷看）", window_features(train).assign(f=fut))]:
    m = HistGradientBoostingClassifier(random_state=0).fit(X[fit], target[fit])
    p = m.predict_proba(X[valid])[:, 1]
    ff = (train[valid].is_fall_seq == 1) & (train[valid].frame == train[valid].fall_frame)
    print(f"{name}：跌倒「開始那一格」的平均機率 {p[ff.to_numpy()].mean():.3f}")

print("\n== 3. 用「格」當單位：分數好看，但林護理長要的是「一次跌倒叫一次」==")
v = train[valid]
m = HistGradientBoostingClassifier(random_state=0).fit(window_features(train[fit]), target[fit])
pv = pd.Series(m.predict_proba(window_features(v))[:, 1] >= 0.5, index=v.index)
tv = target[valid]
print(f"以格計：{len(v)} 格，誤報 {int((pv & (tv == 0)).sum())} 格（{(pv & (tv == 0)).mean():.3f}）")
seq_alarm = pv.groupby(v.seq_id).sum()
nonfall = v.groupby("seq_id").is_fall_seq.first() == 0
print(f"以序列計：非跌倒 {int(nonfall.sum())} 段，其中 {int((seq_alarm[nonfall] > 0).sum())} 段至少響一次")
print(f"單格告警、不合併：一段 4 秒的跌倒平均響 {seq_alarm[~nonfall].mean():.1f} 次")
