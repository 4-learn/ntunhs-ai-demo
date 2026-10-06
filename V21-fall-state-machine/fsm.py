"""V21：跌倒狀態機（V22、V23 也用）。

standing ──(窗口模型 p_fall ≥ 門檻)──> suspected ──(躺／坐地連續 ≥ 2 秒)──> alert
suspected 超過 3 秒都沒有連續躺／坐地 → 回到 standing（例如彎腰撿東西後站起來）
"""
from __future__ import annotations

from dataclasses import dataclass

from common import FPS

GROUND = ("lying", "sitting")


@dataclass
class FallFSM:
    threshold: float = 0.5
    ground_s: float = 2.0      # 躺／坐地要持續多久才告警
    timeout_s: float = 3.0     # 疑似後多久沒結論就回到站立
    state: str = "standing"
    ground_run: int = 0        # 目前連續幾格躺／坐地（姿勢模型判斷）
    since: int = 0             # 進入 suspected 後過了幾格

    def step(self, p_fall: float, pose: str) -> str:
        """吃一格（窗口模型機率、姿勢模型預測），回傳這一格之後的狀態。"""
        self.ground_run = self.ground_run + 1 if pose in GROUND else 0
        if self.state == "alert":
            return self.state
        if self.state == "standing" and p_fall >= self.threshold:
            self.state, self.since = "suspected", 0
        elif self.state == "suspected":
            self.since += 1
            if self.ground_run >= self.ground_s * FPS:
                self.state = "alert"
            elif self.since >= self.timeout_s * FPS and self.ground_run == 0:
                self.state = "standing"
        return self.state

    def reset(self):
        self.state, self.ground_run, self.since = "standing", 0, 0


def run_seq(d, **kw):
    """一段序列從頭播到尾，回傳每格狀態的 list。d 需有 p_fall、pose_pred 欄。"""
    fsm = FallFSM(**kw)
    return [fsm.step(p, g) for p, g in zip(d["p_fall"], d["pose_pred"])]


def hold_last(df, seconds=3.0):
    """每段序列結束後，把最後一格再重複 seconds 秒（模擬「人還停在原地」）。

    poses.csv 每段只有 4 秒；跌倒常在第 1～2 秒才發生，躺地 2 秒還沒到片段就結束了。
    真實攝影機不會在這裡停，所以評估狀態機時補上「保持最後姿勢」的格。
    """
    import pandas as pd

    df = df.sort_values(["seq_id", "frame"])
    last = df.groupby("seq_id", sort=False).tail(1)
    extra = []
    for k in range(1, int(seconds * FPS) + 1):
        e = last.copy()
        e["frame"] = e["frame"] + k
        e["t"] = e["frame"] / FPS
        extra.append(e)
    return pd.concat([df, *extra]).sort_values(["seq_id", "frame"]).reset_index(drop=True)
