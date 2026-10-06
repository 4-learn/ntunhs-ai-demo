"""V19–V23 共用的小模型：窗口跌倒模型＋單格姿勢模型（只用 S01–S18 訓練，不碰測試集）。

P16 的定版模型（models/fall-hgb.joblib）在 V29 才接上；V19–V23 用這裡自己訓練的小模型，
這樣各節可以獨立執行。

    import sys; sys.path.insert(0, "V19-pose-stream")
    from fallmodel import prepare, train_models, score_frames
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from common import FPS, WINDOW_FEATURES, frame_features, window_features

FIT = [f"S{i:02d}" for i in range(1, 15)]      # S01–S14：訓練
VALID = [f"S{i:02d}" for i in range(15, 19)]   # S15–S18：驗證
FALL_WINDOW = 15                               # 跌倒開始後 1.5 秒內的格＝「正在跌」
GROUND = {"lying", "sitting"}                  # 地上（或椅子上）的姿勢


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """依 seq_id、frame 排好、重設 index，並加上每格的「正在跌」標籤 y_win。

    common.window_features 依輸入順序回傳；仍先依 seq_id、frame 排序，因為窗口特徵需要每段序列內的時間順序。
    """
    df = df.sort_values(["seq_id", "frame"]).reset_index(drop=True)
    ff = pd.to_numeric(df["fall_frame"], errors="coerce").fillna(-1).astype(int)
    fall = pd.to_numeric(df["is_fall_seq"], errors="coerce").fillna(0).astype(int) == 1
    df["y_win"] = (fall & (ff >= 0) & (df["frame"] >= ff) & (df["frame"] < ff + FALL_WINDOW)).astype(int)
    return df


def train_models(train: pd.DataFrame):
    """回傳 (窗口跌倒模型, 姿勢模型)。train 必須先經過 prepare()。"""
    # 小資料用 2 條執行緒就夠；機器忙時，OpenMP 開滿反而慢很多倍
    with threadpool_limits(2):
        win = HistGradientBoostingClassifier(random_state=0).fit(window_features(train), train["y_win"])
        pose = HistGradientBoostingClassifier(random_state=0).fit(frame_features(train), train["posture"])
    return win, pose


def score_frames(df: pd.DataFrame, win, pose) -> pd.DataFrame:
    """每格加上 p_fall（窗口模型機率）與 pose_pred（姿勢模型預測）。df 必須先經過 prepare()。"""
    out = df.copy()
    with threadpool_limits(2):
        out["p_fall"] = win.predict_proba(window_features(df))[:, 1]
        out["pose_pred"] = pose.predict(frame_features(df))
    return out


def seq_table(df: pd.DataFrame) -> pd.DataFrame:
    """一段序列一列：標籤、活動、窗口模型分數（序列內最大值）。"""
    return df.groupby("seq_id", sort=True).agg(
        subject_id=("subject_id", "first"), camera_id=("camera_id", "first"),
        activity=("activity", "first"), y=("is_fall_seq", "first"), score=("p_fall", "max"))


def cost(miss: int, fa: int) -> int:
    """全課同一把尺：10×漏報＋1×誤報。"""
    return 10 * miss + fa


__all__ = ["FIT", "VALID", "FALL_WINDOW", "GROUND", "FPS", "WINDOW_FEATURES",
           "threadpool_limits", "prepare", "train_models", "score_frames", "seq_table", "cost"]
