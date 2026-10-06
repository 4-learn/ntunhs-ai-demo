"""P03｜讓 AI 寫，然後挑錯：AI 版清洗 vs cleaning.py，用「護理紀錄離床時，床墊是否真的空著」驗證。"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cleaning import load_bed, load_nurse  # noqa: E402
from common import DATA  # noqa: E402

LIMIT = 1.0  # kPa


def check(nurse, bed, label):
    """護理師寫「離床」的那一分鐘，床墊壓力是否 < 1 kPa？"""
    ex = nurse[nurse["kind"] == "exit"].dropna(subset=["resident_id", "time"])
    ex = ex.assign(minute=ex["time"].dt.floor("min"))
    m = ex.merge(bed, left_on=["resident_id", "minute"], right_on=["resident_id", "time"], how="left",
                 suffixes=("", "_bed"))
    hit = (m["p"] < LIMIT).sum()
    print(f"{label}：離床紀錄 {(nurse['kind'] == 'exit').sum()} 筆，對到住民與時間 {len(ex)} 筆，"
          f"床墊確認空床 {hit} 筆（{hit / (nurse['kind'] == 'exit').sum():.3f}）")


print("== AI 版 ==")
nl = pd.read_csv(DATA / "care/nurse_log.csv", dtype={"room": str})
res = pd.read_csv(DATA / "care/residents.csv", dtype={"room": str})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nl["time"] = pd.to_datetime(nl["time"], format="mixed")
nl = nl.merge(res[["resident_id", "name", "room"]], on=["name", "room"], how="left")
nl["kind"] = nl["note"].map({"離床": "exit", "夜間離床": "exit", "跌倒": "fall"})
bed = pd.read_csv(DATA / "care/bed_pressure.csv", parse_dates=["time"]).rename(columns={"pressure": "p"})
print(f"note 沒對到類別：{nl['kind'].isna().sum()} 筆；名字沒對到住民：{nl['resident_id'].isna().sum()} 筆；"
      f"年份不是 2026：{(nl['time'].dt.year != 2026).sum()} 筆")
check(nl, bed, "AI 版")

print("\n== cleaning.py 版 ==")
nurse = load_nurse()
b = load_bed().rename(columns={"pressure_kpa": "p"})[["resident_id", "time", "p"]]
check(nurse, b, "清洗版")
wrong_unit = load_bed().rename(columns={"pressure": "p"})[["resident_id", "time", "p"]]
check(nurse, wrong_unit, "清洗版但忘了換單位")
