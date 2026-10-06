"""P18：產生交接包 handoff/fall-handoff.json，給影像 AI 課（V19–V23、V29）接手。

    python P18-privacy-handoff/handoff.py

交接包只放「可以重現與檢查」的東西：資料 hash、評估集清單與其 hash、標註規則、已知標註錯誤、
模型 hash、門檻與依據、驗證與測試分數、每段測試序列的分數。不放任何影像。
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
import hashlib
import json
import sys

import joblib
import pandas as pd
import yaml

from common import DATA, MIN_KPT_CONF, ROOT, TEST_SUBJECTS, WINDOW, load_poses, subject_split

sys.path.insert(0, str(ROOT / "P16-pose-eval"))
from build_fall_model import CONFIG_PATH, FALL_WINDOW, MODEL_PATH, counts, seq_meta, seq_score, sha12  # noqa: E402

OUT = ROOT / "P18-privacy-handoff" / "handoff" / "fall-handoff.json"


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
manifest = json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
df = load_poses().sort_values(["seq_id", "frame"])
train, test = subject_split(df)

model = joblib.load(MODEL_PATH)
tm = seq_meta(test).assign(score=seq_score(model, test).round(4))
eval_list = "\n".join(f"{s},{int(y)}" for s, y in tm.y.items())   # 評估集＝「序列編號,標籤」清單

pack = {
    "title": "松年長照中心 跌倒偵測 交接包（機器學習補充 → 影像 AI 課）",
    "version": cfg["release"],
    "privacy": {
        "stored": "只存 17 點骨架（x, y, 信心）與事件 JSON；影像在 YOLO-pose 推論後即丟棄",
        "event_has_image": False,
        "residual_risk": "相對身高、面向等可部分認出受試者（P18 實測）；camera_id＋event_time 可對照排班推出是誰",
        "retention": "待松年長照中心依個資法與機構規定決定（交接時尚未訂定）",
    },
    "data": {
        "file": "data/pose/poses.csv",
        "sha256": manifest["pose/poses.csv"],
        "synthetic": True,
        "fps": 10,
        "split": {"train": "S01–S18", "test": f"{TEST_SUBJECTS[0]}–{TEST_SUBJECTS[-1]}",
                  "validation": "train 內 StratifiedGroupKFold(5, shuffle=True, random_state=0), groups=subject_id"},
    },
    "eval_set": {
        "subjects": list(TEST_SUBJECTS),
        "sequences": int(len(tm)),
        "falls": int(tm.y.sum()),
        "list_format": "每行「seq_id,is_fall_seq」，依 seq_id 排序，以 \\n 連接",
        "list_sha256": sha_text(eval_list),
        "opened": "P16 打開一次（2026-10-06）；V23 可再用一次並須說明",
    },
    "labeling_rules": {
        "unit": "一段序列（40 格、4 秒）＝一個標註單位；is_fall_seq 為序列標籤",
        "activities": ["walk", "stand", "sit", "sit_down", "bend_pick", "fall", "lie_bed"],
        "fall": "從站立或行走中失去平衡、在 2 秒內倒到地面（躺平或坐地都算）",
        "not_fall": "自己躺到床上（lie_bed）、自己坐下（sit_down，再快也不算）、彎腰或蹲下撿東西（bend_pick）",
        "fall_frame": "軀幹開始倒下的第一格；非跌倒序列為 -1",
        "posture": "每格四類 standing/sitting/bending/lying；轉換中以「到目標姿勢一半以上」為界",
        "model_target": f"is_fall_seq==1 且 fall_frame <= frame < fall_frame+{FALL_WINDOW}（跌倒開始 1.5 秒內）",
        "keypoint_conf_min": MIN_KPT_CONF,
        "window_frames": WINDOW,
    },
    "known_label_errors": [
        {"seq_id": "SQ0084", "set": "train", "labeled": "fall", "likely": "lie_bed",
         "why": "fall_frame=-1、40 格皆 lying、沒有倒下的動作"},
        {"seq_id": "SQ0306", "set": "test", "labeled": "lie_bed", "likely": "fall",
         "why": "前 12 格站立，軀幹約 1 秒內由 0 度轉到 80 度以上"},
    ],
    "model": {k: cfg["model"][k] for k in ("name", "version", "path", "sha256_12", "sklearn", "features",
                                             "feature_fn", "label", "sequence_score")},
    "threshold": cfg["threshold"],
    "scores": {
        "validation_oof": {k: cfg["validation"][k] for k in ("tp", "fn", "fp", "precision", "recall")},
        "test": counts(tm.y, tm.score, cfg["threshold"]["value"]),
        "test_per_sequence": [{"seq_id": s, "y": int(r.y), "activity": r.activity, "camera_id": r.camera_id,
                               "score": float(r.score)} for s, r in tm.iterrows()],
    },
    "open_issues": [
        "門檻 0.95 待林護理長核准",
        "Pi 5 速度未實測（P17 只量本機）",
        "EZCam 每 3 秒一張不足以餵窗口模型（P17）",
        "SQ0084、SQ0306 待標註負責人重審",
        "二館（poses-site2.csv）尚未評估，見 V23",
    ],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
text = json.dumps(pack, ensure_ascii=False, indent=1) + "\n"
OUT.write_text(text, encoding="utf-8")

print("== 1. 交接包 ==")
print(f"寫出 P18-privacy-handoff/handoff/fall-handoff.json，{len(text.encode())} bytes；sha256_12 {sha12(OUT)}")
print(f"區塊：{list(pack)}")

print("\n== 2. 接手的人怎麼檢查：每一個 hash 都重算一次 ==")
checks = {
    "poses.csv": hashlib.sha256((DATA / "pose" / "poses.csv").read_bytes()).hexdigest() == pack["data"]["sha256"],
    "評估集清單": sha_text(eval_list) == pack["eval_set"]["list_sha256"],
    "模型檔": sha12(MODEL_PATH) == pack["model"]["sha256_12"],
    "測試分數可重現": counts(tm.y, tm.score, pack["threshold"]["value"]) == pack["scores"]["test"],
}
for k, v in checks.items():
    print(f"{k:8s} {'一致' if v else '不一致'}")

print("\n== 3. 交接包的重點數字 ==")
print(f"評估集：{pack['eval_set']['sequences']} 段、跌倒 {pack['eval_set']['falls']} 段；"
      f"清單 sha256_12 {pack['eval_set']['list_sha256'][:12]}")
print(f"門檻 {pack['threshold']['value']}（{pack['threshold']['approved_by']}）")
print(f"驗證（out-of-fold）：{pack['scores']['validation_oof']}")
print(f"測試：{pack['scores']['test']}")
print(f"已知標註錯誤：{[e['seq_id'] for e in pack['known_label_errors']]}")
print(f"含影像：{pack['privacy']['event_has_image']}；待辦 {len(pack['open_issues'])} 項")
