"""P04 照護事件表的步驟，寫成函式（events_care.py 逐步印出；pitfalls.py 與 workshop 重用）。"""
from __future__ import annotations

import sys

import pandas as pd

from common import ROOT

sys.path.insert(0, str(ROOT / "P03-pandas-clean"))
from cleaning import load_bed, load_nurse, load_wearable  # noqa: E402,F401


def bed_exit_onsets(bed: pd.DataFrame, limit_kpa: float = 1.0) -> pd.DataFrame:
    """床墊壓力由「在床」變成「< limit」的那一分鐘＝離床開始。"""
    out = bed["pressure_kpa"] < limit_kpa
    prev = out.groupby(bed["resident_id"]).shift(fill_value=False)
    return bed.loc[out & ~prev, ["resident_id", "time"]].rename(columns={"time": "exit_at"})


def confirmed_exits(nurse: pd.DataFrame, onsets: pd.DataFrame, tol="10min") -> pd.DataFrame:
    """護理紀錄的離床，往回找 tol 內最近的床墊離床開始；對到的才算「確認的離床」。"""
    exits = nurse[nurse["kind"] == "exit"].sort_values("time")
    m = pd.merge_asof(exits, onsets.sort_values("exit_at"), left_on="time", right_on="exit_at",
                      by="resident_id", direction="backward", tolerance=pd.Timedelta(tol))
    return m


def hr_per_minute(wear: pd.DataFrame) -> pd.Series:
    """不規則間隔 → 每位住民每分鐘一格（該分鐘內的平均；沒量到為 NaN）。"""
    return wear.set_index("ts").groupby("resident_id")["hr"].resample("1min").mean().rename("hr")


def hr_features(hr1: pd.Series, recent: int = 10, base: int = 50) -> pd.DataFrame:
    """檢查點 t 的特徵只用 t 以前：hr_10＝[t-10, t)，hr_base＝[t-60, t-10)。"""
    g = hr1.groupby(level="resident_id")
    f = pd.DataFrame({
        # 「t 這一格」是 [t, t+1) 的量測，已經在檢查點之後，所以先 shift(1)
        "hr_10": g.transform(lambda s: s.shift(1).rolling(recent, min_periods=3).mean()),
        "hr_base": g.transform(lambda s: s.shift(recent + 1).rolling(base, min_periods=15).mean()),
        "hr_n10": g.transform(lambda s: s.shift(1).rolling(recent, min_periods=1).count()),
    })
    f["hr_rise"] = f["hr_10"] - f["hr_base"]
    return f.reset_index().rename(columns={"ts": "t"})


def checkpoint_table(feat: pd.DataFrame, bed: pd.DataFrame, exits: pd.DataFrame,
                     every="10min", horizon="10min") -> pd.DataFrame:
    """一列＝一位住民、一個檢查點（每 10 分鐘、人在床上）；標籤＝接下來 horizon 內會不會離床。"""
    step = pd.Timedelta(every)
    cp = feat[feat["t"] == feat["t"].dt.floor(step)]
    in_bed = bed.loc[bed["pressure_kpa"] >= 1.0, ["resident_id", "time"]].rename(columns={"time": "t"})
    cp = cp.merge(in_bed, on=["resident_id", "t"], how="inner")
    cp = cp.dropna(subset=["hr_10", "hr_base"]).sort_values("t")
    nxt = pd.merge_asof(cp, exits[["resident_id", "exit_at"]].sort_values("exit_at"), left_on="t",
                        right_on="exit_at", by="resident_id", direction="forward",
                        tolerance=pd.Timedelta(horizon), allow_exact_matches=False)
    cp["exit_next_10"] = nxt["exit_at"].notna().astype(int).to_numpy()
    cols = ["resident_id", "t", "hr_10", "hr_base", "hr_rise", "hr_n10", "exit_next_10"]
    return cp[cols].reset_index(drop=True)


def build_care_table():
    bed, nurse, wear = load_bed(), load_nurse(), load_wearable()
    m = confirmed_exits(nurse, bed_exit_onsets(bed))
    exits = m.dropna(subset=["exit_at"]).drop_duplicates(["resident_id", "exit_at"])
    hr1 = hr_per_minute(wear)
    return checkpoint_table(hr_features(hr1), bed, exits), hr1, exits
