"""P03 的清洗規則，寫成函式：P04 會直接 import 這支（不再複製一份）。

每個 load_* 回傳「乾淨的」DataFrame：時間一律是台北時間（tz-aware）、單位統一、重複已移除。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA

TZ = "Asia/Taipei"
MMHG_PER_KPA = 7.50062
LOG_YEAR = 2026  # 護理紀錄沒寫年份的，用「這份紀錄涵蓋的期間」補（2026-01-05～01-12，不跨年）


def load_wearable() -> pd.DataFrame:
    """手環：UTC 字串 → 台北時間；移除重複上傳；hr=0 → NaN。"""
    w = pd.read_json(DATA / "care/wearable.jsonl", lines=True, dtype={"ts": str})
    w["ts"] = pd.to_datetime(w["ts"], utc=True).dt.tz_convert(TZ)
    w = w.drop_duplicates(["device", "ts"]).copy()
    w["hr"] = w["hr"].where(w["hr"] > 0).astype(float)
    w["resident_id"] = "R" + w["device"].str[3:]      # 器材登記：WB-001 配給 R001
    return w.sort_values(["device", "ts"]).reset_index(drop=True)


def load_bed() -> pd.DataFrame:
    """床墊：本地時間字串 → 標上台北時區；mmHg → kPa。"""
    b = pd.read_csv(DATA / "care/bed_pressure.csv")
    b["time"] = pd.to_datetime(b["time"]).dt.tz_localize(TZ)
    b["pressure_kpa"] = np.where(b["unit"] == "mmHg", b["pressure"] / MMHG_PER_KPA, b["pressure"])
    return b.sort_values(["resident_id", "time"]).reset_index(drop=True)


def parse_log_time(s: pd.Series, year: int = LOG_YEAR) -> pd.Series:
    """三種手打格式：2026-01-06 00:01、2026/01/06 00:01、01/06 00:36（沒有年）。"""
    s = s.str.strip().str.replace("/", "-", regex=False)
    no_year = s.str.match(r"^\d{2}-\d{2} ")
    s = s.where(~no_year, f"{year}-" + s)
    return pd.to_datetime(s, format="%Y-%m-%d %H:%M").dt.tz_localize(TZ)


def load_nurse() -> pd.DataFrame:
    """護理紀錄：時間統一、名字去空白、對回住民編號、note 歸成 exit／fall。"""
    n = pd.read_csv(DATA / "care/nurse_log.csv", dtype={"room": str})
    res = pd.read_csv(DATA / "care/residents.csv", dtype={"room": str})
    n["time"] = parse_log_time(n["time"])
    n["name"] = n["name"].str.replace(r"\s+", "", regex=True)
    n = n.merge(res[["resident_id", "name", "room"]], on=["name", "room"], how="left", validate="many_to_one")
    n["kind"] = np.where(n["note"].str.contains("跌"), "fall", "exit")
    return n.sort_values("time").reset_index(drop=True)


def load_plc_a() -> pd.DataFrame:
    a = pd.read_csv(DATA / "line/plc_a.csv")
    a["time"] = pd.to_datetime(a["time"]).dt.tz_localize(TZ)
    return a


def load_plc_b() -> pd.DataFrame:
    """PLC B：毫秒 epoch、華氏。注意：這台 PLC 的時鐘把「本地時間」當成 UTC 寫進 epoch（P03 第 4 小節驗證）。"""
    b = pd.read_csv(DATA / "line/plc_b.csv")
    b["time"] = pd.to_datetime(b["ts"], unit="ms").dt.tz_localize(TZ).dt.as_unit("us")  # 不是 utc=True！精度統一成 us，merge_asof 才不會報錯
    b["temp_c"] = ((b["temp_f"] - 32) * 5 / 9).round(2)
    return b.drop(columns=["ts", "temp_f"])
