"""P04｜讓 AI 寫，然後挑錯：特徵偷看未來（center=True）、用「筆數」當時間窗。照護事件表上實測。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_care_table, load_wearable  # noqa: E402

table, hr1, _ = build_care_table()
g = hr1.groupby(level="resident_id")
base = g.transform(lambda s: s.shift(11).rolling(50, min_periods=15).mean())
n_exit = int(table["exit_next_10"].sum())


def report(label, recent):
    rise = (recent - base).rename("rise").reset_index().rename(columns={"ts": "t"})
    m = table[["resident_id", "t", "exit_next_10"]].merge(rise, on=["resident_id", "t"], how="left")
    flag = m["rise"] > 4
    hit = int(m.loc[flag, "exit_next_10"].sum())
    print(f"{label}：rise>4 的檢查點 {int(flag.sum()):4d} 個，真的離床 {hit:3d} 個"
          f"（{m.loc[flag, 'exit_next_10'].mean():.3f}）；離床抓到 {hit}/{n_exit}")


print("== 錯誤 1：rolling(10, center=True)：一半的資料在未來 ==")
report("過去 10 分鐘 [t-10, t)  ", g.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()))
report("前後各 5 分鐘（center） ", g.transform(lambda s: s.rolling(10, center=True, min_periods=3).mean()))
report("未來 10 分鐘 [t, t+10)  ", g.transform(lambda s: s[::-1].rolling(10, min_periods=3).mean()[::-1]))

print("\n== 錯誤 2：在原始手環資料上 rolling(10)：10 筆不是 10 分鐘 ==")
w = load_wearable()
r = w[(w["resident_id"] == "R003") & (w["ts"] < "2026-01-06 08:00+08:00")]["ts"].reset_index(drop=True)
span = (r - r.shift(9)).dt.total_seconds().dropna() / 60
print(f"R003 第一晚：連續 10 筆量測涵蓋 {span.min():.1f}–{span.max():.1f} 分鐘（間隔 1–3 分鐘不規則）")
