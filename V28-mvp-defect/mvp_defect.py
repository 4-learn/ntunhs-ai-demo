"""V28 示範：製造組 MVP 端到端——取像 → 小格 kNN → Event JSON → 事件 API，最後算漏報與誤報。

    python V28-mvp-defect/mvp_defect.py            # 預設 metal_nut
    python V28-mvp-defect/mvp_defect.py screw

把 MVTec 測試影像（扣掉 V12 用來定門檻的 10 張驗證瑕疵）亂序排成「輸送帶依序拍到的工件」。
NTUNHS_MOCK=1：不連 EZCam、不連網路；事件 API 用 TestClient＋SQLite :memory:。
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TORCH_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from common import ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "V32-deploy-checklist"))
from defect_model import (CachedFeatures, ConveyorSource, PatchKnnDetector, conveyor_order, evaluate,  # noqa: E402
                          load_model, model_paths, online_features)
from deploy import JsonLog, load_config, self_check, sha256_12, start  # noqa: E402
from event_api import Event, create_app, make_client  # noqa: E402
from pipeline import TW, ApiSender, PipeConfig, Pipeline, StepClock  # noqa: E402

pd.set_option("display.unicode.east_asian_width", True)
CAT = sys.argv[1] if len(sys.argv) > 1 else "metal_nut"
WORK = Path(f"/tmp/ntunhs/V28/{CAT}")
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)

print(f"== 1. 定版模型：patch-knn-{CAT} ==")
bank, card = load_model(CAT)
npy, _ = model_paths(CAT)
h12 = sha256_12(npy)
print(f"記憶良品 {card['n_memory']}、校正良品 {card['n_calib']}、記憶庫 {card['bank_shape'][0]} 個小格 × {card['bank_shape'][1]} 維")
print(f"門檻規則：校正良品＋驗證瑕疵 {len(card['valid_index'])} 張，成本 20×漏報＋1×誤報 → 原始分數 {card['threshold_raw']:.3f}")
print(f"  （V08 最大值規則會是 {card['calib_max_raw']:.3f}；驗證集上成本 {card['valid_cost']}：漏報 {card['valid_miss']}、誤報 {card['valid_fa']}）")
print(f"分數換算 0–1：原始分數 ÷ {card['scale']:.3f} → 門檻 {card['threshold']:.3f}")
print(f"記憶庫檔 sha256_12 = {h12}")

print("\n== 2. 設定檔＋啟動自檢（V32）==")
cfg_dict = {
    "app": f"defect-line1-{CAT}", "release": card["version"], "source": "defect", "site": "精工五金",
    "model": {"name": card["name"], "version": card["version"], "path": f"cache/models/{npy.name}", "sha256_12": h12},
    "threshold": {"value": card["threshold"], "approved_by": "王課長", "approved_on": "2026-01-09",
                  "chosen_on": f"V12 驗證集：{CAT} 校正良品＋test 瑕疵固定取 10 張（seed 0）",
                  "rule": "成本 20×漏報＋1×誤報 最低；同成本取較高門檻"},
    "cooldown_s": 0, "cameras": [{"id": "line1cam1", "prefix": "line1cam1", "where": "一號線輸送帶上方"}],
    "api_url": "http://127.0.0.1:8000", "log_path": str(WORK / "app.jsonl")}
(WORK / "defect.yaml").write_text(yaml.safe_dump(cfg_dict, allow_unicode=True, sort_keys=False), encoding="utf-8")
cfg = load_config(WORK / "defect.yaml")
for item, ok, msg in self_check(cfg, base=ROOT):
    print(f"{'通過' if ok else '失敗'}  {item}")
log = JsonLog(cfg.log_path, clock=lambda: datetime(2026, 1, 12, 7, 59, tzinfo=TW))
start(cfg, ROOT, log)
print("→ 允許啟動")

shots = conveyor_order(CAT)
feats = CachedFeatures(CAT)

print("\n== 3. 線上抽特徵 vs 快取特徵（要一致才能用快取跑驗收）==")
import cv2  # noqa: E402
for s in shots[:2]:
    img = cv2.imread(s.path)
    d = np.abs(online_features(img) - feats(img)).max()
    print(f"{Path(s.path).parent.name}/{Path(s.path).name}：最大差異 {d:.3f}")

print("\n== 4. 輸送帶跑一輪（每 3 秒一顆，cooldown 0）==")
client = make_client(create_app())
clock = StepClock(datetime(2026, 1, 12, 8, 0, 0, tzinfo=TW), 3)
det = PatchKnnDetector(bank, card, feats, part_ids=[f"P{i + 1:05d}" for i in range(len(shots))])
pipe = Pipeline(ConveyorSource(shots, clock), det,
                ApiSender(client), PipeConfig(source="defect", site=cfg.site, camera_id="line1cam1",
                                              prefix="line1cam1", threshold=cfg.threshold.value,
                                              model={"name": cfg.model.name, "version": cfg.model.version,
                                                     "sha256_12": cfg.model.sha256_12}, cooldown_s=0))
rows = []
t0 = time.perf_counter()
for s in shots:
    st = pipe.step()
    rows.append({"part_id": det.last["part_id"], "truth": "ng" if s.truth else "ok", "defect": s.defect,
                 "raw": det.last["raw"], "score": det.last["score"], "status": st})
elapsed = time.perf_counter() - t0
df = pd.DataFrame(rows)
print(df.head(8).to_string(index=False, formatters={"raw": "{:.3f}".format, "score": "{:.3f}".format}))
print(f"… 共 {len(df)} 顆；狀態統計 {df['status'].value_counts().to_dict()}")

print("\n== 5. 從事件 API 讀回來算成績 ==")
events = client.get("/events", params={"kind": "ng"}).json()
alarm_ids = {e["evidence"]["part_id"] for e in events}
df["alarm"] = df["part_id"].isin(alarm_ids).astype(int)
m = evaluate((df["truth"] == "ng").astype(int), df["alarm"])
print(f"工件 {len(df)}（瑕疵 {int((df['truth'] == 'ng').sum())}、良品 {int((df['truth'] == 'ok').sum())}）  事件 {len(events)} 筆")
print(f"漏報 {m['漏報']}  誤報 {m['誤報']}  recall {m['recall']:.3f}  precision {m['precision']:.3f}")
print(f"每筆都通過 Event 驗證：{all(Event.model_validate({k: v for k, v in e.items() if k != 'ack'}) for e in events)}")
mx = evaluate((df["truth"] == "ng").astype(int), (df["raw"] >= card["calib_max_raw"]).astype(int))
print(f"同一批分數若用 V08 最大值門檻：漏報 {mx['漏報']}  誤報 {mx['誤報']}  recall {mx['recall']:.3f}")

print("\n漏報的瑕疵類型：")
by = df[df["truth"] == "ng"].groupby("defect").agg(瑕疵=("alarm", "size"), 漏報=("alarm", lambda a: int((a == 0).sum())))
print(by.to_string())

print("\n== 6. 換算成一班（8 小時、每 3 秒一顆、瑕疵率千分之一）==")
n_shift = 8 * 3600 // 3
n_ok = int((df["truth"] == "ok").sum())
fpr = m["誤報"] / n_ok
print(f"測試批瑕疵比例 {(df['truth'] == 'ng').mean():.3f}（MVTec 刻意放很多瑕疵），產線約 0.001")
print(f"一班 {n_shift} 顆：預期瑕疵 {n_shift * 0.001:.1f} 顆、漏報 {n_shift * 0.001 * (1 - m['recall']):.2f} 顆、"
      f"誤報 {n_shift * 0.999 * fpr:.0f} 顆（良品誤報率 {fpr:.3f}，只由 {n_ok} 顆良品估計）")

print("\n== 7. 一筆事件（API 存的就是這個；沒有影像本體）==")
e = events[0]
print(json.dumps({k: e[k] for k in ("event_id", "event_time", "score", "threshold", "evidence", "model")},
                 ensure_ascii=False))

# 速度只供教師紀錄參考（含秒數的行不貼進講義）
print(f"\n（本機 CPU {os.environ['TORCH_THREADS']} 執行緒、特徵走快取：每顆 kNN 約 {elapsed / len(df) * 1000:.0f} ms）")
