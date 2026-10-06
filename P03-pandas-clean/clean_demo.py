"""P03｜多源資料清洗：時間、單位、重複。先看問題長什麼樣子，再用 cleaning.py 的規則修好。"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cleaning import TZ, load_bed, load_nurse, load_plc_a, load_plc_b, load_wearable, parse_log_time  # noqa: E402
from common import DATA  # noqa: E402

pd.set_option("display.width", 120)

print("== 1. 五個來源，五種時間 ==")
raw = {
    "wearable.jsonl": pd.read_json(DATA / "care/wearable.jsonl", lines=True, dtype={"ts": str}),
    "bed_pressure.csv": pd.read_csv(DATA / "care/bed_pressure.csv"),
    "nurse_log.csv": pd.read_csv(DATA / "care/nurse_log.csv", dtype={"room": str}),
    "plc_a.csv": pd.read_csv(DATA / "line/plc_a.csv"),
    "plc_b.csv": pd.read_csv(DATA / "line/plc_b.csv"),
}
tcol = {"wearable.jsonl": "ts", "bed_pressure.csv": "time", "nurse_log.csv": "time", "plc_a.csv": "time",
        "plc_b.csv": "ts"}
print(pd.DataFrame([{"檔案": k, "列數": len(v), "時間欄": tcol[k], "第一筆": str(v[tcol[k]].iloc[0]),
                     "型別": str(v[tcol[k]].dtype)} for k, v in raw.items()]).to_string(index=False))

print("\n== 2. UTC vs 本地時間：手環差了 8 小時 ==")
w = raw["wearable.jsonl"]
naive = pd.to_datetime(w["ts"].str.rstrip("Z"))                  # 錯：把 Z 拿掉，當成本地時間
right = pd.to_datetime(w["ts"], utc=True).dt.tz_convert(TZ)       # 對：UTC → 台北
print("原始字串       ", w["ts"].iloc[0])
print("丟掉 Z 當本地  ", naive.iloc[0])
print("UTC → 台北     ", right.iloc[0])
night_naive = naive.dt.hour.isin([0, 1, 2, 3, 4, 5]).mean()
night_right = right.dt.hour.isin([0, 1, 2, 3, 4, 5]).mean()
print(f"「凌晨 0–6 點」的筆數比例：錯誤版 {night_naive:.3f}、正確版 {night_right:.3f}")
print("床墊第一筆（本地）", raw["bed_pressure.csv"]["time"].iloc[0], "← 和正確版的手環對得上")

print("\n== 3. 毫秒 epoch：PLC B 的時鐘設錯了 ==")
pb, pa = raw["plc_b.csv"], load_plc_a()
as_utc = pd.to_datetime(pb["ts"], unit="ms", utc=True).dt.tz_convert(TZ)
as_local = pd.to_datetime(pb["ts"], unit="ms").dt.tz_localize(TZ)
print(f"PLC A 本地時間：{pa['time'].min():%H:%M} – {pa['time'].max():%H:%M}")
print(f"PLC B 當 UTC 換台北：{as_utc.min():%H:%M} – {as_utc.max():%H:%M}  ← 下午？產線只開上午")
print(f"PLC B 直接當本地：  {as_local.min():%H:%M} – {as_local.max():%H:%M}")
temp_c = (pb["temp_f"] - 32) * 5 / 9
a_temp = pa.set_index("time")["temp_c"]
for name, t in [("當 UTC", as_utc), ("當本地", as_local)]:
    s = a_temp.reindex(t)
    ok = s.notna().to_numpy()
    if ok.sum() == 0:
        print(f"{name}：和 PLC A 對上 0 筆")
    else:
        diff = np.abs(s.to_numpy()[ok] - temp_c.to_numpy()[ok]).mean()
        print(f"{name}：和 PLC A 對上 {ok.sum()} 筆，兩台溫度平均差 {diff:.3f} °C")
print(f"°F → °C：{pb['temp_f'].iloc[0]} °F = {temp_c.iloc[0]:.3f} °C（PLC A 同一秒 {a_temp.iloc[0]} °C）")

print("\n== 4. 單位：kPa 和 mmHg 混在同一欄 ==")
bed = raw["bed_pressure.csv"]
print(bed.groupby("unit")["pressure"].describe()[["count", "50%", "min", "max"]].round(2).to_string())
LIMIT = 1.0  # 規則：壓力 < 1 kPa 視為不在床上
raw_out = (bed["pressure"] < LIMIT).groupby(bed["unit"]).mean()
clean = load_bed()
fix_out = (clean["pressure_kpa"] < LIMIT).groupby(clean["unit"]).mean()
print(f"「壓力 < {LIMIT}」判離床的比例：")
print(pd.DataFrame({"沒換算": raw_out, "換成 kPa": fix_out}).round(3).to_string())

print("\n== 5. 重複上傳 ==")
dup_all = w.duplicated().sum()
dup_key = w.duplicated(["device", "ts"]).sum()
print(f"手環 {len(w)} 筆；整列完全相同 {dup_all} 筆；(device, ts) 相同 {dup_key} 筆")
cnt_raw = w.groupby("device").size()
cnt_fix = w.drop_duplicates(["device", "ts"]).groupby("device").size()
print(f"每支手環的筆數：去重前 {cnt_raw.min()}–{cnt_raw.max()}，去重後 {cnt_fix.min()}–{cnt_fix.max()}")
print("WB-001 被重複上傳的一筆：")
d = w[w.duplicated(["device", "ts"], keep=False) & (w["device"] == "WB-001")].head(2)
print(d.to_string())

print("\n== 6. hr=0：改成缺值，不是刪掉 ==")
hr0 = (w.drop_duplicates(["device", "ts"])["hr"] == 0).sum()
print(f"去重後 hr=0 有 {hr0} 筆 → 改成 NaN（保留時間點，P04 對齊時需要知道「這段時間有沒有量到」）")

print("\n== 7. 護理紀錄：三種時間格式，有的沒有年 ==")
nl = raw["nurse_log.csv"]
fmt = np.select([nl["time"].str.match(r"^\d{4}-"), nl["time"].str.match(r"^\d{4}/")],
                ["2026-01-06 00:01", "2026/01/06 00:01"], "01/06 00:36（沒有年）")
print(pd.Series(fmt).value_counts().to_string())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")              # pandas 會警告「猜不出格式」，這裡先關掉看結果
    guess = pd.to_datetime(nl["time"])
print("pd.to_datetime 直接轉，年份分布：", guess.dt.year.value_counts().to_dict())
print("沒有年的那筆變成：", guess[nl["time"] == "01/06 00:36"].iloc[0])
fixed = parse_log_time(nl["time"])
print("補上年份後：", fixed[nl["time"] == "01/06 00:36"].iloc[0])

print("\n== 8. 名字多了空白：對不回住民 ==")
res = pd.read_csv(DATA / "care/residents.csv", dtype={"room": str})
naive_join = nl.merge(res[["resident_id", "name", "room"]], on=["name", "room"], how="left")
print(f"直接用 (name, room) 對住民表：{naive_join['resident_id'].isna().sum()} 筆對不到，例如 {nl['name'][naive_join['resident_id'].isna()].iloc[0]!r}")
nurse = load_nurse()
print(f"去掉空白後：{nurse['resident_id'].isna().sum()} 筆對不到")
print("note 歸類：", nurse["kind"].value_counts().to_dict())

print("\n== 9. 清洗紀錄（交給下一節）==")
wear = load_wearable()
report = pd.DataFrame([
    {"來源": "wearable", "原始": len(w), "清洗後": len(wear), "處理": "UTC→台北、去重、hr=0→NaN"},
    {"來源": "bed_pressure", "原始": len(bed), "清洗後": len(clean), "處理": "標上台北時區、mmHg→kPa"},
    {"來源": "nurse_log", "原始": len(nl), "清洗後": len(nurse), "處理": "三種格式→一種、補年、名字去空白"},
    {"來源": "plc_a", "原始": len(raw["plc_a.csv"]), "清洗後": len(pa), "處理": "標上台北時區（掉 300 秒未補）"},
    {"來源": "plc_b", "原始": len(pb), "清洗後": len(load_plc_b()), "處理": "epoch 當本地、°F→°C"},
])
print(report.to_string(index=False))
