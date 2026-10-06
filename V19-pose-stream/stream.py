"""V19：把 poses.csv 的一段序列逐格「播放」成串流，邊收邊算 1 秒窗口特徵。

    python V19-pose-stream/stream.py
"""
from collections import deque

import numpy as np
import pandas as pd

from common import FPS, WINDOW, keypoints, load_poses, window_features


def play(df: pd.DataFrame, seq_id: str):
    """一次吐一格（一列），就像攝影機每 0.1 秒送來一副骨架。"""
    seq = df[df["seq_id"] == seq_id].sort_values("frame")
    for _, row in seq.iterrows():
        yield row


def hip_y(rows: pd.DataFrame) -> np.ndarray:
    k = keypoints(rows)
    with np.errstate(all="ignore"):
        return np.nanmean(k[:, [11, 12], 1], 1)


def online_window(df, seq_id, buf_len):
    """串流版：只保留最近 buf_len 格，每來一格就算一次，取最後一列。"""
    buf, out = deque(maxlen=buf_len), []
    for row in play(df, seq_id):
        buf.append(row)
        w = window_features(pd.DataFrame(list(buf)).reset_index(drop=True))
        out.append(w.iloc[-1])
    return pd.DataFrame(out).reset_index(drop=True)


def main():
    df = load_poses()
    print("== 1. 資料：一段序列＝一個人做一件事 4 秒 ==")
    print(f"序列 {df['seq_id'].nunique()} 段、每段 {df.groupby('seq_id').size().iloc[0]} 格、"
          f"{FPS} fps；受試者 {df['subject_id'].nunique()} 位")
    seq = df.groupby("seq_id").first()
    print("跌倒序列", int((seq["is_fall_seq"] == 1).sum()), "段；跌倒開始的格（fall_frame）中位數",
          int(seq.loc[seq["is_fall_seq"] == 1, "fall_frame"].median()))
    last = df.sort_values("frame").groupby("seq_id")["posture"].last()
    for act in ("fall", "sit_down"):
        ids = seq.index[seq["activity"] == act]
        print(f"{act:<8} 最後一格的姿勢：", last[ids].value_counts().to_dict())

    sid = "SQ0012"
    print(f"\n== 2. generator：{sid} 一次吐一格 ==")
    stream = play(df, sid)
    for _ in range(3):
        r = next(stream)
        print(f"frame {r['frame']:2d}  t={r['t']:.1f}s  posture={r['posture']:<8} 左髖 y={r['left_hip_y']:.1f}")
    print("…（還沒播的格不在記憶體裡算，這是串流和整批的差別）")

    print(f"\n== 3. 邊收邊算窗口特徵，和整批算的結果比 ==")
    batch = window_features(df[df["seq_id"] == sid].sort_values("frame").reset_index(drop=True))
    for n in (WINDOW, WINDOW + 1):
        on = online_window(df, sid, n)
        diff = (on[batch.columns] - batch).abs().max()
        bad = [c for c in batch.columns if diff[c] > 1e-9]
        print(f"緩衝 {n} 格：最大差異 {np.nanmax(diff.to_numpy()):.3f}；不一致的特徵 {bad or '無'}")

    print(f"\n== 4. {sid} 的髖部高度（像素，越大越低），三種取樣頻率看到的樣子 ==")
    y = hip_y(df[df["seq_id"] == sid].sort_values("frame"))
    ff = int(seq.loc[sid, "fall_frame"])
    print(f"跌倒從第 {ff} 格開始（t={ff / FPS:.1f}s）")
    for name, step in [("10 fps", 1), ("2 fps", 5), ("每 3 秒", 30)]:
        idx = list(range(0, 40, step))
        print(f"{name:>6}（{len(idx):2d} 格）: " + " ".join(f"{y[i]:.0f}" for i in idx[:14]) + (" …" if len(idx) > 14 else ""))


if __name__ == "__main__":
    main()
