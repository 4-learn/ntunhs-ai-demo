"""P16：產生定版跌倒模型與設定檔（V29 讀這兩個檔）。

    python P16-pose-eval/build_fall_model.py

做三件事，全部只用 train（S01–S18），不碰測試集：
1. 依受試者 5 折的 out-of-fold 序列分數 → 以成本 10×漏報＋1×誤報 選門檻
2. 用 S01–S18 全部重新訓練窗口模型 → models/fall-hgb.joblib
3. 寫 config/fall.yaml（模型 hash、資料 hash、特徵、門檻與依據）
4. 最後打開測試集（S19–S24）一次，結果寫進 config/fall.yaml 的 test 欄位（只記錄，不再改門檻）
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"  # HGB 單執行緒：模型檔逐位元組可重現（執行緒數不同，hash 會不同）
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from common import DATA, ROOT, WINDOW_FEATURES, load_poses, subject_split, window_features

MODEL_PATH = ROOT / "models" / "fall-hgb.joblib"
CONFIG_PATH = ROOT / "config" / "fall.yaml"
FALL_WINDOW = 15                      # 標籤：fall_frame 起 15 格（1.5 秒）＝「正在跌」（P15 定版）
GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)
COST_MISS, COST_FA = 10, 1            # 全課同一把尺


def onset_label(d):
    return ((d.is_fall_seq == 1) & (d.frame >= d.fall_frame) & (d.frame < d.fall_frame + FALL_WINDOW)).astype(int)


def new_model():
    return HistGradientBoostingClassifier(random_state=0)


def seq_meta(d):
    return d.groupby("seq_id").agg(y=("is_fall_seq", "first"), subject_id=("subject_id", "first"),
                                   camera_id=("camera_id", "first"), activity=("activity", "first"))


def seq_score(model, d):
    """序列分數＝序列內每格跌倒機率的最大值。"""
    p = pd.Series(model.predict_proba(window_features(d)[WINDOW_FEATURES])[:, 1], index=d.index)
    return p.groupby(d.seq_id).max()


def oof_scores(train):
    """依受試者 5 折：每段序列的分數都來自「沒看過這個人」的模型。"""
    meta = seq_meta(train)
    X, y = window_features(train)[WINDOW_FEATURES], onset_label(train)
    p = pd.Series(np.nan, index=train.index)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for a, b in cv.split(meta, meta.y, groups=meta.subject_id):
        r_fit = train.seq_id.isin(meta.index[a]).to_numpy()
        r_val = train.seq_id.isin(meta.index[b]).to_numpy()
        p[r_val] = new_model().fit(X[r_fit], y[r_fit]).predict_proba(X[r_val])[:, 1]
    return meta.assign(score=p.groupby(train.seq_id).max())


def counts(y, score, t):
    alarm = score >= t
    tp = int(((y == 1) & alarm).sum())
    fn = int(((y == 1) & ~alarm).sum())
    fp = int(((y == 0) & alarm).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "cost": COST_MISS * fn + COST_FA * fp,
            "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}


def cost_table(oof):
    return pd.DataFrame([{"門檻": t, **counts(oof.y, oof.score, t)} for t in GRID])


def choose_threshold(table):
    """成本最低；同成本取較低的門檻（寧可多響，也不要漏）。"""
    best = table.cost.min()
    return float(table[table.cost == best].門檻.min())


def sha12(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main():
    df = load_poses().sort_values(["seq_id", "frame"])
    train, test = subject_split(df)
    oof = oof_scores(train)
    table = cost_table(oof)
    t = choose_threshold(table)
    oof_res = counts(oof.y, oof.score, t)

    model = new_model().fit(window_features(train)[WINDOW_FEATURES], onset_label(train))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    manifest = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))
    cfg = {
        "app": "fall-monitor",
        "release": "2026.01",
        "model": {
            "name": "fall-hgb",
            "version": "2026.01",
            "path": "models/fall-hgb.joblib",
            "sha256_12": sha12(MODEL_PATH),
            "sklearn": sklearn.__version__,
            "estimator": "HistGradientBoostingClassifier(random_state=0)",
            "omp_num_threads": 1,
            "features": list(WINDOW_FEATURES),
            "feature_fn": "common.window_features",
            "label": f"is_fall_seq==1 且 fall_frame <= frame < fall_frame+{FALL_WINDOW}",
            "sequence_score": "序列內每格 predict_proba[:, 1] 的最大值",
            "trained_on": {"subjects": "S01–S18", "sequences": int(len(oof)),
                           "data": "data/pose/poses.csv",
                           "data_sha256_12": manifest["files"]["pose/poses.csv"][:12]},
        },
        "threshold": {
            "value": t,
            "chosen_on": "train S01–S18 依受試者 5 折 out-of-fold 序列分數（StratifiedGroupKFold, random_state=0）",
            "rule": f"成本 {COST_MISS}×漏報＋{COST_FA}×誤報 最低；門檻格點 0.05–0.95、間隔 0.05；同成本取較低門檻",
            "approved_by": "待林護理長核准",
        },
        "validation": {"set": "train out-of-fold（37 段跌倒、215 段非跌倒）", **oof_res},
        "test": {"set": "S19–S24（84 段，跌倒 11 段）", "opened": "2026-10-06，門檻決定之後打開一次",
                 **counts(seq_meta(test).y, seq_score(model, test), t),
                 "note": "唯一的誤報 SQ0306 標成 lie_bed，實際動作是跌倒（標註錯誤）；不因此改門檻或模型"},
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text("# P16 定版：跌倒偵測模型與門檻（V29 讀這份）。由 P16-pose-eval/build_fall_model.py 產生，不要手改。\n"
                           + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"門檻 {t}；OOF：{oof_res}")
    print(f"寫出 models/fall-hgb.joblib（sha256_12 {cfg['model']['sha256_12']}）與 config/fall.yaml")


if __name__ == "__main__":
    main()
