"""P04｜照護事件表：一列＝一位住民的一個「每 10 分鐘檢查點」；特徵＝過去 10 分鐘心率上升；標籤＝接下來 10 分鐘會不會離床。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import (bed_exit_onsets, checkpoint_table, confirmed_exits, hr_features,  # noqa: E402
                      hr_per_minute, load_bed, load_nurse, load_wearable)

pd.set_option("display.width", 120)
OUT = Path("/tmp/ntunhs/P04")
OUT.mkdir(parents=True, exist_ok=True)

bed, nurse, wear = load_bed(), load_nurse(), load_wearable()

print("== 1. 離床到底是幾點？護理紀錄 vs 床墊 ==")
onset = bed_exit_onsets(bed)
m = confirmed_exits(nurse, onset)
lag = (m["time"] - m["exit_at"]).dt.total_seconds() / 60
print(f"護理紀錄「離床」{len(m)} 筆；床墊偵測到的離床開始 {len(onset)} 次")
print(f"merge_asof（往回找 10 分鐘內最近的床墊離床）：對到 {m['exit_at'].notna().sum()} 筆、對不到 {m['exit_at'].isna().sum()} 筆")
print(f"護理師比床墊晚記錄：中位數 {lag.median():.0f} 分鐘、最多 {lag.max():.0f} 分鐘")
print(m[["time", "name", "note", "exit_at"]].head(3).to_string(index=False))
exits = m.dropna(subset=["exit_at"]).drop_duplicates(["resident_id", "exit_at"])
print(f"同一次離床被記了兩筆的去掉後：{len(exits)} 次確認的離床")

print("\n== 2. 手環：不規則間隔 → 每分鐘一格 ==")
r1 = wear[wear["resident_id"] == "R003"]
print("R003 原始前 4 筆的時間：", [t.strftime("%H:%M:%S") for t in r1["ts"].iloc[:4]])
hr1 = hr_per_minute(wear)
print(f"resample('1min').mean() 之後：{len(hr1)} 格（每人從第一晚 20:00 連續切到最後一晚 07:59，白天也有空格）")
print(f"沒有量測的分鐘（NaN）比例：{hr1.isna().mean():.3f}")

print("\n== 3. 特徵：只能用「檢查點以前」的資料 ==")
feat = hr_features(hr1)
print("hr_10   ＝ [t-10, t) 的平均心率")
print("hr_base ＝ [t-60, t-10) 的平均：這位住民「平常」的心率")
print("hr_rise ＝ hr_10 − hr_base")

print("\n== 4. 一列＝一個檢查點：每 10 分鐘、人在床上 ==")
table = checkpoint_table(feat, bed, exits)
print(f"{len(table)} 列、{table.shape[1]} 欄；接下來 10 分鐘會離床的 {table['exit_next_10'].sum()} 列"
      f"（{table['exit_next_10'].mean():.3f}）")
show = table.round({"hr_10": 2, "hr_base": 2, "hr_rise": 2})
show["t"] = show["t"].dt.strftime("%m-%d %H:%M")
print(show[show["exit_next_10"] == 1].head(2).to_string(index=False))
print(show[show["exit_next_10"] == 0].head(2).to_string(index=False))
table.to_csv(OUT / "care_events.csv", index=False)

print("\n== 5. 特徵有沒有用？依標籤比一比 ==")
print(table.groupby("exit_next_10")[["hr_10", "hr_base", "hr_rise"]].mean().round(3).to_string())
bins = pd.cut(table["hr_rise"], [-np.inf, 0, 2, 4, 6, np.inf], labels=["<=0", "0–2", "2–4", "4–6", ">6"])
print("\nhr_rise 分箱：")
print(table.groupby(bins, observed=True)["exit_next_10"].agg(列數="size", 離床比例="mean").round(3).to_string())
