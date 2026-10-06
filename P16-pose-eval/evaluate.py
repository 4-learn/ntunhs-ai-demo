"""P16：讀定版模型與設定檔，重現選門檻、錯誤分析，並看測試集（S19–S24）的結果。

    python P16-pose-eval/build_fall_model.py   # 先產生 models/fall-hgb.joblib、config/fall.yaml
    python P16-pose-eval/evaluate.py
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
import joblib
import pandas as pd
import yaml

from build_fall_model import (CONFIG_PATH, MODEL_PATH, cost_table, counts, oof_scores, seq_meta, seq_score,
                              sha12)
from common import frame_features, keypoints, load_poses, subject_split

pd.set_option("display.width", 200)
cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
T = cfg["threshold"]["value"]

print("== 1. 定版檔案 ==")
ok = sha12(MODEL_PATH) == cfg["model"]["sha256_12"]
print(f"models/fall-hgb.joblib sha256_12 {sha12(MODEL_PATH)}，與 config/fall.yaml 一致：{ok}")
print(f"特徵 {len(cfg['model']['features'])} 個；訓練 {cfg['model']['trained_on']['subjects']}；"
      f"資料 poses.csv sha256_12 {cfg['model']['trained_on']['data_sha256_12']}")

df = load_poses().sort_values(["seq_id", "frame"])
train, test = subject_split(df)
oof = oof_scores(train)

print("\n== 2. 選門檻：train 依受試者 5 折的 out-of-fold 序列分數 ==")
tab = cost_table(oof)
print(tab[tab.門檻.isin([0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95])].to_string(index=False))
print(f"成本最低：門檻 {T}（approved_by：{cfg['threshold']['approved_by']}）")

print("\n== 3. 錯誤分析（out-of-fold，門檻 {}）==".format(T))
oof["alarm"] = oof.score >= T
err = oof[(oof.alarm & (oof.y == 0)) | (~oof.alarm & (oof.y == 1))]
print(err[["subject_id", "camera_id", "activity", "y", "score"]].round(3).to_string())

print("\n== 4. 漏報的那一段：標成 fall，但每一格都在躺 ==")
s = train[train.seq_id == "SQ0084"]
print(f"SQ0084：fall_frame {int(s.fall_frame.iloc[0])}、posture {sorted(s.posture.unique())}、"
      f"軀幹角中位數 {frame_features(s).torso_angle.median():.1f} 度")

print("\n== 5. 跌坐 vs 躺平：最後姿勢不同，分數一樣高 ==")
falls = oof[oof.y == 1].copy()
last = train.groupby("seq_id").posture.last()
falls["結束姿勢"] = last[falls.index].map({"lying": "躺平", "sitting": "跌坐"})
print(falls.groupby("結束姿勢").score.agg(["count", "min", "median"]).round(3).to_string())

print("\n== 6. 寢室（cam1）：下半身被擋住，誤報比較多嗎 ==")
neg = oof[oof.y == 0]
k = keypoints(train)
legs_lost = pd.Series(pd.isna(k[:, 13:17, 0]).all(axis=1), index=train.index).groupby(train.seq_id).mean()
t6 = pd.DataFrame({"非跌倒段數": neg.groupby("camera_id").size(),
                   "下半身看不到的格": legs_lost[neg.index].groupby(neg.camera_id).mean().round(3),
                   "誤報@0.5": (neg.score >= 0.5).groupby(neg.camera_id).sum(),
                   f"誤報@{T}": (neg.score >= T).groupby(neg.camera_id).sum()})
print(t6.to_string())

print(f"\n== 7. 測試集 S19–S24：只打開這一次（門檻 {T}，不再調整）==")
model = joblib.load(MODEL_PATH)
tm = seq_meta(test).assign(score=seq_score(model, test))
res = counts(tm.y, tm.score, T)
print(f"{len(tm)} 段、跌倒 {int(tm.y.sum())} 段 → {res}")
print(f"與 config/fall.yaml 記錄一致：{all(cfg['test'][k] == v for k, v in res.items())}")
bad = tm[((tm.score >= T) & (tm.y == 0)) | ((tm.score < T) & (tm.y == 1))]
print(bad[["subject_id", "camera_id", "activity", "y", "score"]].round(3).to_string())
print(f"跌倒段的最低分數 {tm.score[tm.y == 1].min():.3f}；非跌倒段（扣掉 SQ0306）的最高分數 "
      f"{tm.score[(tm.y == 0) & (tm.index != 'SQ0306')].max():.3f}")
