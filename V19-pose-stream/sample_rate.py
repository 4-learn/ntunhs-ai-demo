"""V19：降採樣實驗。10 fps、2 fps、每 3 秒一張時，窗口特徵模型能抓到幾成跌倒？

做法（取樣後保持）：系統仍每 0.1 秒更新一次，但只有「新照片到的那一格」是新的骨架，
其他格沿用最近一張（EZCam 輪詢拍照就是這樣：手上永遠是最近一張）。
每種頻率用不同起點（相位）各播一次，並用「同樣頻率」的資料重新訓練，比較才公平。
訓練 S01–S14、驗證 S15–S18；測試集 S19–S24 不打開。

    python V19-pose-stream/sample_rate.py
"""
import sys

import numpy as np
import pandas as pd

from common import KPTS, ROOT, keypoints, load_poses, subject_split, window_features

sys.path.insert(0, str(ROOT / "V19-pose-stream"))
from fallmodel import FIT, VALID, prepare, train_models  # noqa: E402

KCOLS = [f"{k}_{a}" for k in KPTS for a in ("x", "y", "c")]
RATES = [("10 fps", 1), ("2 fps", 5), ("每 3 秒", 30)]
THRESHOLD = 0.5


def hip_y(d: pd.DataFrame) -> np.ndarray:
    k = keypoints(d)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(k[:, [11, 12], 1], 1)


def sample_and_hold(d: pd.DataFrame, step: int, phase: int) -> pd.DataFrame:
    """每 step 格才有一張新骨架（從第 phase 格開始），中間沿用上一張。d 已依 seq_id、frame 排序。"""
    d = d.reset_index(drop=True)
    back = ((d["frame"] - phase) % step).to_numpy()
    keep = (d["frame"] >= phase).to_numpy()
    out = d.copy()
    out[KCOLS] = d[KCOLS].to_numpy()[np.arange(len(d)) - back]
    out["fresh"] = back == 0
    out = out[keep].copy()
    out["seq_id"] = out["seq_id"] + f"@{phase}"
    return prepare(out)  # 重新排序（"SQ0001@10" 會排在 "SQ0001@2" 前面）


def main():
    df = prepare(load_poses())
    train, _ = subject_split(df)
    fit, val = train[train["subject_id"].isin(FIT)], train[train["subject_id"].isin(VALID)]

    print("== 1. 一次跌倒有多快 ==")
    falls = val[(val["is_fall_seq"] == 1) & (val["fall_frame"].astype(int) >= 0)]
    dur = []
    for _, d in falls.groupby("seq_id"):
        ff = int(d["fall_frame"].iloc[0])
        ground = d.index[(d["frame"] >= ff) & d["posture"].isin(["lying", "sitting"])]
        y = pd.Series(hip_y(d)).ffill().to_numpy()
        start, final = np.median(y[:max(ff, 1)]), np.median(y[-10:])
        settled = np.where(np.abs(y - final) <= 0.1 * abs(final - start))[0]
        settled = settled[settled >= ff][0]
        dur.append(((d.loc[ground[0], "frame"] - ff) / 10, (settled - ff) / 10))
    lab, hip = np.array(dur).T
    print(f"驗證集跌倒 {len(dur)} 段（從 fall_frame 起算）")
    print(f"  姿勢標籤變成躺／坐：{lab.min():.1f}～{lab.max():.1f} 秒，中位數 {np.median(lab):.1f}")
    print(f"  髖部降到最終高度的 90%：{hip.min():.1f}～{hip.max():.1f} 秒，中位數 {np.median(hip):.1f}")

    print("\n== 2. 降採樣：每種頻率拍到什麼、抓到幾成 ==")
    rows = []
    for name, step in RATES:
        phases = list(range(0, step, max(1, step // 10)))
        tr = pd.concat([sample_and_hold(fit, step, p) for p in phases])
        tr = prepare(tr)
        win, _ = train_models(tr)
        va = prepare(pd.concat([sample_and_hold(val, step, p) for p in phases]))
        va["p"] = win.predict_proba(window_features(va))[:, 1]
        ff = va["fall_frame"].astype(int)
        mid = va["fresh"] & (va["is_fall_seq"] == 1) & (va["frame"] > ff) & (va["frame"] < ff + 8)
        va["mid"] = mid
        s = va.groupby("seq_id").agg(y=("is_fall_seq", "first"), p=("p", "max"), mid=("mid", "any"),
                                     n=("fresh", "sum"))
        f, ok = s[s["y"] == 1], s[s["y"] == 0]
        rows.append({"取樣": name, "每段幾張": round(s["n"].mean(), 1), "播放次數": len(s),
                     "拍到倒下途中": round(f["mid"].mean(), 3),
                     "recall": round((f["p"] >= THRESHOLD).mean(), 3),
                     "誤報率": round((ok["p"] >= THRESHOLD).mean(), 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"（門檻 {THRESHOLD}；recall 與誤報率以「一段序列的一次播放」為單位）")


if __name__ == "__main__":
    main()
