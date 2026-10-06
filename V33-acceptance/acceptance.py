"""V33 示範：兩組 MVP 各端到端跑一次，產出驗收報告（功能清單、成績、每班／每晚告警、已知限制）。

    python V33-acceptance/acceptance.py        # 約 5 分鐘（製造組兩類各跑一輪 kNN）

驗收標準在 ACCEPT 裡，**跑之前就寫好**；跑完只能填結果，不能回頭改標準。
報告寫到 /tmp/ntunhs/V33/acceptance.md（給王課長、林護理長簽名的那一份）。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_THREADS", "2")
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from common import ROOT, load_poses, subject_split  # noqa: E402

for d in ("V28-mvp-defect", "V29-mvp-fall", "V22-debounce"):
    sys.path.insert(0, str(ROOT / d))
from defect_model import evaluate, model_paths, run_line  # noqa: E402
from fall_pipeline import POSTURE_PATH, run_night  # noqa: E402
from deploy import load_config, self_check, sha256_12  # noqa: E402
from event_api import Event, create_app, make_client  # noqa: E402
import night as V22  # noqa: E402

pd.set_option("display.unicode.east_asian_width", True)
WORK = Path("/tmp/ntunhs/V33")
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)

# ---- 驗收標準（2026-01-09 與王課長、林護理長訂定；跑之前寫好）
ACCEPT = {
    "defect": {"recall": 0.95, "良品誤剔率": 0.10},     # 最終測試
    "fall": {"recall": 1.00, "每晚誤報": 5.0},           # 測試序列；每晚＝模擬夜晚平均
}
V13 = {"metal_nut": {"漏報": 2, "誤報": 2}, "screw": {"漏報": 2, "誤報": 40}}   # V13 成績表（方法 C）


def md_table(df):
    """不靠 tabulate：DataFrame → Markdown 表格。"""
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def all_valid(events):
    return all(Event.model_validate({k: v for k, v in e.items() if k != "ack"}) for e in events)


def has_image(events):
    return any(len(str(v)) > 200 for e in events for v in e["evidence"].values())


rows_func, rows_score = [], []

print("== 1. 製造組：metal_nut、screw 各跑一輪輸送帶 ==")
defect = {}
for cat in ("metal_nut", "screw"):
    npy, _ = model_paths(cat)
    info = {"name": f"patch-knn-{cat}", "version": "2026.01", "sha256_12": sha256_12(npy)}
    client = make_client(create_app())
    df, card = run_line(cat, client, info)
    events = client.get("/events", params={"kind": "ng"}).json()
    ids = {e["evidence"]["part_id"] for e in events}
    m = evaluate(df.truth, df.part_id.isin(ids).astype(int))
    fpr = m["誤報"] / int((df.truth == 0).sum())
    dup = client.post("/events", json={k: v for k, v in events[0].items() if k != "ack"}).status_code
    defect[cat] = dict(m, fpr=fpr, n=len(df), n_ok=int((df.truth == 0).sum()), thr=card["threshold"])
    print(f"{cat}：{len(df)} 顆、事件 {len(events)} 筆、漏報 {m['漏報']}、誤報 {m['誤報']}")
    rows_func += [
        {"組": "製造", "項目": f"{cat} 事件全部被 API 收下（無 422）", "結果": (df.status == "rejected").sum() == 0},
        {"組": "製造", "項目": f"{cat} 事件通過 Event 驗證、不含影像", "結果": all_valid(events) and not has_image(events)},
        {"組": "製造", "項目": f"{cat} 事件的模型 hash＝記憶庫檔", "結果": {e["model"]["sha256_12"] for e in events} == {info["sha256_12"]}},
        {"組": "製造", "項目": f"{cat} 重送同一筆 → 409", "結果": dup == 409},
        {"組": "製造", "項目": f"{cat} 成績與 V13 成績表一致", "結果": (m["漏報"], m["誤報"]) == (V13[cat]["漏報"], V13[cat]["誤報"])},
    ]
    rows_score.append({"組": "製造", "評估集": f"{cat} 最終測試", "件數": len(df), "漏報": m["漏報"], "誤報": m["誤報"],
                       "recall": m["recall"], "precision": m["precision"], "良品誤剔率": fpr,
                       "驗收": m["recall"] >= ACCEPT["defect"]["recall"] and fpr <= ACCEPT["defect"]["良品誤剔率"]})

print("\n== 2. 健康組：S19–S24 排成一晚，3 支攝影機 ==")
raw = yaml.safe_load((ROOT / "V32-deploy-checklist/config/fall.yaml").read_text(encoding="utf-8"))
raw["log_path"] = str(WORK / "fall.jsonl")
(WORK / "fall.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
cfg = load_config(WORK / "fall.yaml")
p16 = yaml.safe_load((ROOT / cfg.model.card).read_text(encoding="utf-8"))
_, test = subject_split(load_poses())
client = make_client(create_app())
seq, pipes, events = run_night(test, cfg, client)
tp = int(((seq.y == 1) & (seq.alarm == 1)).sum())
fn = int(((seq.y == 1) & (seq.alarm == 0)).sum())
fp = int(((seq.y == 0) & (seq.alarm == 1)).sum())
print(f"84 段、事件 {len(events)} 筆、抓到 {tp}/{tp + fn}、誤報 {fp}")
checks = self_check(cfg, ROOT)
rows_func += [
    {"組": "健康", "項目": "啟動自檢（含 P16 定版卡比對）", "結果": all(ok for _, ok, _ in checks)},
    {"組": "健康", "項目": "事件全部被 API 收下（無 422）",
     "結果": all(r["status"] != "rejected" for p in pipes.values() for r in p.log)},
    {"組": "健康", "項目": "事件通過 Event 驗證、evidence 只有骨架編號", "結果": all_valid(events) and not has_image(events)},
    {"組": "健康", "項目": "成績與 P16 測試集一致", "結果": (tp, fn, fp) == (p16["test"]["tp"], p16["test"]["fn"], p16["test"]["fp"])},
]
mfg_ok = [r for r in rows_func if r["組"] == "製造"]

print("\n== 3. 健康組：30 個模擬夜晚（V22 的做法，換成定版模型與門檻 0.95）==")
V22.THRESHOLD = cfg.threshold.value
fall_m, pose_m = joblib.load(ROOT / cfg.model.path), joblib.load(POSTURE_PATH)
eps = V22.scored_episodes(V22.prepare(test), fall_m, pose_m)
nights = []
for s in range(30):
    tl = V22.night(eps, s)
    n = V22.notify(tl, V22.fsm_flags(tl, rearm=True), cooldown_s=cfg.cooldown_s)
    sm = V22.summarize(n, tl)
    got, tot = map(int, sm["有通知到的跌倒"].split("/"))
    nights.append({"通知": sm["通知"], "誤報": sm["誤報"], "漏通知": tot - got})
nt = pd.DataFrame(nights)
no_cd = []
for s in range(30):
    tl = V22.night(eps, s)
    sm = V22.summarize(V22.notify(tl, V22.fsm_flags(tl, rearm=True)), tl)
    got, tot = map(int, sm["有通知到的跌倒"].split("/"))
    no_cd.append(tot - got)
print(f"住民 6 位（S19–S24）× 8 小時；每晚通知 {nt.通知.mean():.1f}（{nt.通知.min()}–{nt.通知.max()}），"
      f"誤報 {nt.誤報.mean():.1f}（{nt.誤報.min()}–{nt.誤報.max()}），漏通知 {nt.漏通知.sum()} 件／30 晚")
print(f"同樣 30 晚、不加冷卻：漏通知 {sum(no_cd)} 件 → 漏掉的都是被 60 秒冷卻吞掉的")
rows_func.append({"組": "健康", "項目": "模擬 30 晚：冷卻沒有吞掉真跌倒", "結果": nt.漏通知.sum() == 0})
rows_score.append({"組": "健康", "評估集": "S19–S24 測試序列", "件數": len(seq), "漏報": fn, "誤報": fp,
                   "recall": tp / (tp + fn), "precision": tp / (tp + fp), "良品誤剔率": fp / int((seq.y == 0).sum()),
                   "驗收": tp / (tp + fn) >= ACCEPT["fall"]["recall"] and nt.誤報.mean() <= ACCEPT["fall"]["每晚誤報"]})

print("\n== 4. 驗收表 A：功能清單 ==")
fa = pd.DataFrame(rows_func)
fa["結果"] = fa["結果"].map({True: "通過", False: "不通過"})
print(fa.to_string(index=False))

print("\n== 5. 驗收表 B：成績（標準：瑕疵 recall ≥ 0.95 且良品誤剔率 ≤ 0.10；跌倒 recall 1.0 且每晚誤報 ≤ 5）==")
sb = pd.DataFrame(rows_score)
sb["驗收"] = sb["驗收"].map({True: "通過", False: "不通過"})
print(sb.to_string(index=False, float_format="{:.3f}".format))

print("\n== 6. 驗收表 C：換成現場的量 ==")
mn = defect["metal_nut"]
print(f"一號線 metal_nut：一班 9600 顆、瑕疵率千分之一 → 誤剔良品約 {9600 * 0.999 * mn['fpr']:.0f} 顆、漏報 {9.6 * (1 - mn['recall']):.2f} 顆")
print(f"本館跌倒監測：每晚通知約 {nt.通知.mean():.0f} 次，其中誤報 {nt.誤報.mean():.1f} 次（模擬夜晚，跌倒比例偏高）")

print("\n== 7. 已知限制（寫進報告，簽名前要讀）==")
LIMITS = [
    f"metal_nut 良品誤剔率 {mn['fpr']:.3f} 只由 {mn['n_ok']} 顆良品估計（{mn['誤報']}/{mn['n_ok']}）；誤差範圍很大",
    "screw 不通過：良品誤剔率 0.976，V13 已判定不能上線；只做「標記給人看」",
    "MVTec 是實驗室影像（CC BY-NC-SA 4.0），打光、背景與精工五金產線不同；上線前要用產線照片重做記憶庫與門檻",
    f"跌倒測試只有 {tp + fn} 段跌倒（6 位受試者、演出來的）；{tp}/{tp + fn} 的 recall 95% 信賴區間下限約 {0.025 ** (1 / (tp + fn)):.2f}",
    "SQ0306 標註錯誤（標 lie_bed、實為跌倒），照 P16 決定不改；若改，誤報為 0",
    "每晚數字來自模擬夜晚：每人每晚 10 次在鏡頭前、跌倒後倒地 60 秒，都是假設",
    f"冷卻 60 秒：模擬 30 晚有 {nt.漏通知.sum()} 件真跌倒，因為同一人 60 秒內剛有一則（多半是誤報的）通知而被吞掉（V22 第 4 小節）；"
    "管線的冷卻又是以攝影機為單位，寢室兩人時更容易發生。健康組只能「有條件通過」",
    "Pi 5 速度未實測；kNN 在教師機每顆約 0.7–0.9 秒，EZCam 每 3 秒一張（P17、V26）",
]
for i, s in enumerate(LIMITS, 1):
    print(f"{i}. {s}")

report = ["# 驗收報告 2026.01（看見科技）", "", "## A. 功能清單", "", md_table(fa), "",
          "## B. 成績", "", md_table(sb), "", "## C. 已知限制", ""]
report += [f"{i}. {s}" for i, s in enumerate(LIMITS, 1)]
report += ["", "簽名：王課長 ＿＿＿＿　林護理長 ＿＿＿＿"]
(WORK / "acceptance.md").write_text("\n".join(report) + "\n", encoding="utf-8")
sb.to_csv(WORK / "scores.csv", index=False)
print(f"\n已寫出 /tmp/ntunhs/V33/acceptance.md（{len(report)} 行）、scores.csv")
