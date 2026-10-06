"""V23：二館的處置——只重新校正門檻 vs 收二館資料重訓。

只用二館「有標籤」的 42 段，依住民做留一交叉驗證（T01–T10 輪流當驗證，其餘 9 位用來校正或重訓），
這樣每個數字都是在「沒參與校正／重訓的住民」上量的。

    python V23-site-drift/adapt.py
"""
import sys

import pandas as pd

from common import ROOT, load_poses, subject_split

sys.path.insert(0, str(ROOT / "V19-pose-stream"))
sys.path.insert(0, str(ROOT / "V21-fall-state-machine"))
sys.path.insert(0, str(ROOT / "V23-site-drift"))
from drift import THRESHOLD, load_site2, seq_results  # noqa: E402
from fallmodel import cost, prepare, train_models  # noqa: E402

GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def pick_threshold(d, win, pose):
    """在 d 上掃門檻，成本 10×漏報＋1×誤報 最低者；同分取較高的門檻（誤報較少）。"""
    best = None
    for th in GRID:
        _, s = seq_results(d, win, pose, th)
        c = cost(int(((s["y"] == 1) & ~s["alert"]).sum()), int(((s["y"] == 0) & s["alert"]).sum()))
        if best is None or c <= best[1]:
            best = (th, c)
    return best[0]


def tally(s):
    return int(((s["y"] == 1) & ~s["alert"]).sum()), int(((s["y"] == 0) & s["alert"]).sum())


def main():
    df = prepare(load_poses())
    train, _ = subject_split(df)
    lab = load_site2()
    lab = lab[lab["labeled"]].copy()
    lab["is_fall_seq"] = lab["is_fall_seq"].astype(int)
    lab["fall_frame"] = lab["fall_frame"].astype(int)
    win, pose = train_models(train)

    print("== 1. 二館有標籤 42 段，依住民留一交叉驗證 ==")
    rows, ths = [], []
    for subj in sorted(lab["subject_id"].unique()):
        held, rest = lab[lab["subject_id"] == subj], lab[lab["subject_id"] != subj]
        _, s0 = seq_results(held, win, pose, THRESHOLD)
        th = pick_threshold(rest, win, pose)
        ths.append(th)
        _, s1 = seq_results(held, win, pose, th)
        w2, p2 = train_models(prepare(pd.concat([train, rest])))
        _, s2 = seq_results(held, w2, p2, THRESHOLD)
        for name, s in [("照搬（門檻 0.5）", s0), ("重新校正門檻", s1), ("加二館資料重訓", s2)]:
            m, f = tally(s)
            rows.append({"做法": name, "住民": subj, "跌倒": int((s["y"] == 1).sum()), "漏報": m, "誤報": f})
    r = pd.DataFrame(rows).groupby("做法", sort=False)[["跌倒", "漏報", "誤報"]].sum()
    r["recall"] = ((r["跌倒"] - r["漏報"]) / r["跌倒"]).round(3)
    r["成本"] = 10 * r["漏報"] + r["誤報"]
    print(r.to_string())
    print("每一折重新校正選到的門檻：", " ".join(f"{t:.1f}" for t in ths))


if __name__ == "__main__":
    main()
