"""P04｜產線事件表：一列＝一個工件；特徵＝完成前 30 秒的振動、溫度、轉速；標籤＝ng。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, ROOT

sys.path.insert(0, str(ROOT / "P03-pandas-clean"))
from cleaning import TZ, load_plc_a, load_plc_b  # noqa: E402

pd.set_option("display.width", 120)
OUT = Path("/tmp/ntunhs/P04")
OUT.mkdir(parents=True, exist_ok=True)

parts = pd.read_csv(DATA / "line/inspection.csv")
parts["done_at"] = pd.to_datetime(parts["done_at"]).dt.tz_localize(TZ)
a, b = load_plc_a(), load_plc_b()

print("== 1. 三個來源，三種頻率 ==")
gap = parts["done_at"].diff().dt.total_seconds()
print(f"工件 {len(parts)} 個（ng {(parts['result'] == 'ng').sum()} 個），間隔 {gap.min():.0f}–{gap.max():.0f} 秒")
print(f"PLC A 每秒一筆：{len(a)} 筆（應有 14400，掉了 {14400 - len(a)} 秒）")
print(f"PLC B 每 5 秒一筆：{len(b)} 筆")

print("\n== 2. merge_asof：每個工件配上「完成那一刻」最近的 PLC 讀值 ==")
snap = pd.merge_asof(parts.sort_values("done_at"), b.sort_values("time"), left_on="done_at", right_on="time",
                     direction="backward", tolerance=pd.Timedelta("5s"))
print(snap[["part_id", "done_at", "time", "spindle_rpm"]].head(3).to_string(index=False))
print(f"對不到（5 秒內沒有 PLC B 讀值）：{snap['time'].isna().sum()} 個")
print("只看一個瞬間的讀值很吵；工件是 30 多秒加工出來的，要看整段。")

print("\n== 3. resample：PLC A 補成完整的每秒，再算「過去 30 秒」 ==")
sa = a.set_index("time").resample("1s").mean()
print(f"resample('1s') 後 {len(sa)} 秒；缺的 {sa['vib_rms'].isna().sum()} 秒是 NaN（不補值）")
roll = sa.rolling("30s", closed="left").agg(["mean", "max", "count"])   # [t-30s, t)
roll.columns = [f"{c}_{s}" for c, s in roll.columns]
sb = (b.set_index("time")["spindle_rpm"].resample("1s").mean()          # 每 5 秒一筆 → 每秒一格（4/5 是 NaN）
      .rolling("30s", closed="left").mean().rename("rpm_30s"))           # mean 會跳過 NaN
print("rolling('30s', closed='left')：每一秒的值＝它『之前』30 秒的統計，不含自己")

print("\n== 4. 一列＝一個工件 ==")
ev = parts.merge(roll[["vib_rms_mean", "vib_rms_max", "temp_c_mean", "vib_rms_count"]],
                 left_on="done_at", right_index=True, how="left")
ev = ev.merge(sb, left_on="done_at", right_index=True, how="left")
ev = ev.rename(columns={"vib_rms_mean": "vib_30s", "vib_rms_max": "vib_max_30s", "temp_c_mean": "temp_30s",
                        "vib_rms_count": "n_sec"})
ev["temp_slope_5m"] = ev["temp_30s"] - ev["done_at"].map(
    lambda t: sa["temp_c"].loc[t - pd.Timedelta("330s"):t - pd.Timedelta("300s")].mean())
ev["y"] = (ev["result"] == "ng").astype(int)
cols = ["part_id", "done_at", "vib_30s", "vib_max_30s", "temp_30s", "temp_slope_5m", "rpm_30s", "n_sec", "y"]
ev = ev[cols]
print(f"{len(ev)} 列 × {ev.shape[1]} 欄；ng 比例 {ev['y'].mean():.3f}")
show = ev.copy()
show["done_at"] = show["done_at"].dt.strftime("%H:%M:%S")
print(show.round(3).iloc[[0, 1, 2]].to_string(index=False))
print("...")
print(show[show["y"] == 1].round(3).head(2).to_string(index=False))
print(f"n_sec < 30（窗口內掉資料）的工件：{(ev['n_sec'] < 30).sum()} 個；最少 {ev['n_sec'].min():.0f} 秒")
ev.to_csv(OUT / "line_events.csv", index=False)

print("\n== 5. 特徵 vs 標籤 ==")
print(ev.groupby("y")[["vib_30s", "vib_max_30s", "temp_30s", "temp_slope_5m", "rpm_30s"]].mean().round(3).to_string())
bins = pd.cut(ev["vib_30s"], [0, 1.1, 1.3, 1.5, np.inf], labels=["<1.1", "1.1–1.3", "1.3–1.5", ">=1.5"])
print("\nvib_30s 分箱：")
print(ev.groupby(bins, observed=True)["y"].agg(工件="size", ng="sum", ng比例="mean").round(3).to_string())
