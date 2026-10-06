"""V34 示範：同一個題目（健康組跌倒監測），兩份繳交資料——一份照規矩做、一份「AI 幫忙趕出來的」。
用互審清單逐項檢查，程式能查的 8 項自動跑。

    python V34-demo-review/demo_review.py      # 約 1 分鐘

有問題的那份（flawed）由下面的 make_flawed() 產生，錯誤都是真的做出來的（真的用測試受試者訓練、
真的在測試集上挑門檻），不是只改文字。學生先看 /tmp/ntunhs/V34/flawed/ 自己找，再看程式的結果。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from common import ROOT, TEST_SUBJECTS, WINDOW_FEATURES, load_poses, subject_split, window_features  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "V29-mvp-fall"))
sys.path.insert(0, str(ROOT / "P16-pose-eval"))
from build_fall_model import onset_label  # noqa: E402
from fall_pipeline import run_night  # noqa: E402
from review import CHECKLIST, review  # noqa: E402
from deploy import load_config, sha256_12  # noqa: E402
from event_api import create_app, make_client  # noqa: E402

pd.set_option("display.unicode.east_asian_width", True)
WORK = Path("/tmp/ntunhs/V34")
shutil.rmtree(WORK, ignore_errors=True)
DF = load_poses().sort_values(["seq_id", "frame"])
TRAIN, TEST = subject_split(DF)
TRAIN_S = sorted(TRAIN.subject_id.unique())


def seq_scores(model, d):
    p = pd.Series(model.predict_proba(window_features(d)[WINDOW_FEATURES])[:, 1], index=d.index)
    return pd.DataFrame({"y": d.groupby("seq_id").is_fall_seq.first(), "score": p.groupby(d.seq_id).max()})


def counts(s, t):
    a = s.score >= t
    return {"tp": int((a & (s.y == 1)).sum()), "fn": int((~a & (s.y == 1)).sum()), "fp": int((a & (s.y == 0)).sum())}


def rescore(sub):
    """互審的人自己重算：讀繳交的模型檔，在宣稱的評估集（測試受試者）上用繳交的門檻算。"""
    model = joblib.load(ROOT / sub["model"]["path"])
    return counts(seq_scores(model, DF[DF.subject_id.isin(sub["split"]["test_subjects"])]), sub["threshold"]["value"])


def write(path, sub, log, events, notice):
    path.mkdir(parents=True)
    (path / "submission.yaml").write_text(yaml.safe_dump(sub, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (path / "test_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log), encoding="utf-8")
    (path / "events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    (path / "NOTICE.md").write_text(notice, encoding="utf-8")


def demo_events():
    raw = yaml.safe_load((ROOT / "V32-deploy-checklist/config/fall.yaml").read_text(encoding="utf-8"))
    raw["log_path"] = str(WORK / "fall.jsonl")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "fall.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _, _, events = run_night(TEST, load_config(WORK / "fall.yaml"), make_client(create_app()))
    return [{k: v for k, v in e.items() if k != "ack"} for e in events]


NOTICE_OK = ("# 授權\n- poses.csv：本課合成資料\n- ultralytics／yolo11n-pose：AGPL-3.0（交付前由法務確認，見 P12）\n"
             "- scikit-learn：BSD-3-Clause\n")


def make_honest(events):
    card = yaml.safe_load((ROOT / "config/fall.yaml").read_text(encoding="utf-8"))
    sub = {"group": "健康組 A", "source": "pose",
           "model": {"path": card["model"]["path"], "sha256_12": card["model"]["sha256_12"]},
           "threshold": {"value": card["threshold"]["value"], "chosen_on": card["threshold"]["chosen_on"],
                         "rule": card["threshold"]["rule"], "approved_by": "林護理長"},
           "split": {"train_subjects": TRAIN_S, "threshold_subjects": TRAIN_S, "test_subjects": list(TEST_SUBJECTS)},
           "claims": {k: card["test"][k] for k in ("tp", "fn", "fp")}}
    log = [{"who": "健康組 A", "when": "2026-10-06", "set": "test", "what": "P16 定版後打開一次"}]
    write(WORK / "honest", sub, log, events, NOTICE_OK)


def make_flawed(events):
    """「AI 幫忙趕出來的」版本：每一個錯都真的做出來。"""
    d = WORK / "flawed"
    # 錯 1：資料不夠，乾脆 24 位受試者全部拿來訓練（包含測試受試者）
    X, y = window_features(DF)[WINDOW_FEATURES], onset_label(DF)
    old = HistGradientBoostingClassifier(random_state=0, max_iter=50).fit(X, y)
    model = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    d.mkdir(parents=True)
    joblib.dump(old, d / "fall-v1.joblib")
    stale = sha256_12(d / "fall-v1.joblib")
    (d / "fall-v1.joblib").unlink()
    # 錯 4：重訓後忘了更新 hash（submission 裡寫的是 v1 的 hash）
    # 錯 2、3：在測試集上掃門檻，找成本最低的；每換一次特徵就看一次測試集
    s = seq_scores(model, TEST)
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    t = float(min(grid, key=lambda g: (10 * counts(s, g)["fn"] + counts(s, g)["fp"], -g)))
    claims = counts(s, t)
    sub = {"group": "健康組 B", "source": "pose",
           "model": {"path": str(d / "fall-v2.joblib"), "sha256_12": stale},
           "threshold": {"value": t, "chosen_on": "S19–S24 掃 0.05–0.95", "rule": "成本 10×漏報＋1×誤報",
                         "approved_by": "健康組 B"},                                    # 錯 5：自己核准
           "split": {"train_subjects": TRAIN_S + list(TEST_SUBJECTS), "threshold_subjects": list(TEST_SUBJECTS),
                     "test_subjects": list(TEST_SUBJECTS)},
           "claims": claims}                                   # 第 8 項會通過：數字是真的，只是不能證明什麼
    log = [{"who": "健康組 B", "when": "2026-10-05", "set": "test", "what": f"特徵 v{i}"} for i in (1, 2, 3)]
    ev = [dict(e, evidence={"seq_id": e["evidence"]["seq_id"], "frame": e["evidence"]["frame_no"]}) for e in events[:3]]
    shutil.rmtree(d)                                                                  # write() 會重建資料夾
    write(d, sub, log, ev, "# 授權\n- 本專案程式 MIT\n")                                 # 錯 7：沒寫 AGPL
    joblib.dump(model, d / "fall-v2.joblib")
    (d / "snapshot-cam1.jpg").write_bytes(b"\xff\xd8\xff")                              # 錯 6：留了現場截圖
    return t


events = demo_events()
make_honest(events)
t_bad = make_flawed(events)

print("== 1. 兩份繳交資料 ==")
for name in ("honest", "flawed"):
    files = sorted(p.name for p in (WORK / name).iterdir())
    print(f"{name}：{files}")
print(f"flawed 宣稱：{yaml.safe_load((WORK / 'flawed/submission.yaml').read_text(encoding='utf-8'))['claims']}"
      f"、門檻 {t_bad}")

print(f"\n== 2. 互審清單 {len(CHECKLIST)} 項（{sum(a for _, a in CHECKLIST)} 項程式自動查）==")
res = {n: review(WORK / n, rescore) for n in ("honest", "flawed")}
mark = {True: "通過", False: "不通過", None: "人工"}
tab = pd.DataFrame({"項目": [r[0] for r in res["honest"]],
                    "honest": [mark[r[1]] for r in res["honest"]],
                    "flawed": [mark[r[1]] for r in res["flawed"]]})
tab.index = range(1, len(tab) + 1)
print(tab.to_string())

print("\n== 3. flawed 每一項的說明 ==")
for i, (item, ok, msg) in enumerate(res["flawed"], 1):
    if ok is False:
        print(f"{i}. {msg}")

print("\n== 4. 偷看測試集的代價：同一份測試序列 ==")
honest = counts(seq_scores(joblib.load(ROOT / "models/fall-hgb.joblib"), TEST), 0.95)
bad = counts(seq_scores(joblib.load(WORK / "flawed/fall-v2.joblib"), TEST), t_bad)
print(f"honest（S01–S18 訓練、門檻在 train 上選）：{honest}")
print(f"flawed（S01–S24 訓練、門檻在測試集上選）：{bad}  ← 看起來更好，但已經沒有沒看過的資料可以證明")
