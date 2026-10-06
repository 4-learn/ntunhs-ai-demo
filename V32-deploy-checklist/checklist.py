"""V32 示範：設定檔驗證、模型 sha256_12、啟動自檢（hash 不符就拒絕啟動）、JSON Lines 紀錄、版本切換與回滾。

    python V32-deploy-checklist/checklist.py

所有寫入都在 /tmp/ntunhs/V32/（每次執行先清空），不動 demo/ 裡的模型與設定。
"""
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from common import MODELS, ROOT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deploy import TW, JsonLog, Releases, StartupRefused, ValidationError, load_config, self_check, sha256_12, start  # noqa: E402

WORK = Path("/tmp/ntunhs/V32/demo")
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)


class Clock:
    """假時鐘：每寫一行紀錄走 1 秒（講義輸出可重現；上線用真時間）。"""

    def __init__(self):
        self.t = datetime(2026, 1, 19, 8, 0, 0, tzinfo=TW)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


print("== 1. 設定檔：兩組各一份，載入時驗證 ==")
for name in ("fall", "defect"):
    c = load_config(HERE / "config" / f"{name}.yaml")
    print(f"{c.app:13s} {c.site}  門檻 {c.threshold.value:.2f}（{c.threshold.approved_by} {c.threshold.approved_on}）"
          f"  冷卻 {c.cooldown_s:.0f}s  攝影機 {[x.id for x in c.cameras]}")

raw = yaml.safe_load((HERE / "config/fall.yaml").read_text(encoding="utf-8"))
bad = {
    "門檻寫成 35": {**raw, "threshold": {**raw["threshold"], "value": 35}},
    "拼錯 cooldown": {**{k: v for k, v in raw.items() if k != "cooldown_s"}, "cooldwn_s": 60},
    "攝影機重複": {**raw, "cameras": raw["cameras"] + [raw["cameras"][0]]},
    "門檻沒寫依據": {**raw, "threshold": {k: v for k, v in raw["threshold"].items() if k != "chosen_on"}},
}
for name, d in bad.items():
    p = WORK / "bad.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    try:
        load_config(p)
        print(f"[{name}] 竟然通過")
    except ValidationError as e:
        x = e.errors()[0]
        print(f"[{name}] {'.'.join(map(str, x['loc']))}: {x['msg']}")

print("\n== 2. 模型檔的 sha256_12 ==")
for f in sorted(MODELS.glob("fall-hgb.joblib")) + sorted(MODELS.glob("yolo11n*.pt")):
    print(f"{f.name:16s} {f.stat().st_size / 1e6:6.3f} MB  sha256_12 {sha256_12(f)}")
print("V32 設定檔寫的：", load_config(HERE / "config/fall.yaml").model.sha256_12,
      "；P16 定版卡寫的：", yaml.safe_load((ROOT / "config/fall.yaml").read_text(encoding="utf-8"))["model"]["sha256_12"])

print("\n== 3. 啟動自檢 ==")
clock = Clock()
log = JsonLog(WORK / "app.jsonl", clock)
cfg = load_config(HERE / "config/fall.yaml").model_copy(update={"log_path": str(WORK / "app.jsonl")})
for item, ok, msg in self_check(cfg, base=ROOT):
    print(f"{'通過' if ok else '失敗'}  {item}：{msg}")
start(cfg, ROOT, log)
print("→ 允許啟動")

print("\n有人把模型檔換掉（只改了 1 個位元組）：")
fake = WORK / "models" / "fall-hgb.joblib"
fake.parent.mkdir(parents=True)
(WORK / "config").mkdir()
shutil.copyfile(ROOT / "config/fall.yaml", WORK / "config/fall.yaml")      # 定版卡照抄，只動模型檔
b = bytearray((MODELS / "fall-hgb.joblib").read_bytes())
b[-1] ^= 1
fake.write_bytes(bytes(b))
print(f"檔案大小相同：{fake.stat().st_size == (MODELS / 'fall-hgb.joblib').stat().st_size}")
try:
    start(cfg, WORK, log)
    print("→ 允許啟動（不應該！）")
except StartupRefused as e:
    print(f"→ 拒絕啟動：{e}")

print("\n== 4. 結構化紀錄（JSON Lines，一行一筆）==")
log.write("event_sent", camera="cam2", event_id="s1cam2-20260119-080412-0", score=0.98, status=201)
log.write("api_error", level="warning", camera="cam1", status="queued", detail="ConnectionError")
log.write("event_rejected", level="error", camera="cam1", event_id="s1cam1-20260119-080420-0", field="score")
for line in (WORK / "app.jsonl").read_text(encoding="utf-8").splitlines():
    print(line)
recs = log.read()
print("level=error 的紀錄：", [r["event"] for r in recs if r["level"] == "error"])

print("\n== 5. 版本與回滾 ==")
rel = Releases(WORK / "site", log)
v1 = {**raw, "log_path": str(WORK / "app.jsonl")}
v2 = {**v1, "threshold": {**raw["threshold"], "value": 0.97, "approved_on": "2026-01-19",
                          "rule": "V30：夜班誤報多，林護理長同意試行兩週（未重算驗證集）"}}
v3 = {**v1, "model": {**raw["model"], "version": "2026.02", "sha256_12": "0123456789ab"}}
CARD = ROOT / "config/fall.yaml"
rel.add("2026.01", v1, MODELS / "fall-hgb.joblib", CARD)
rel.add("2026.02", v2, MODELS / "fall-hgb.joblib", CARD)
rel.add("2026.03", v3, MODELS / "fall-hgb.joblib", CARD)      # 設定寫了新 hash，模型檔卻沒換
print("切到 2026.01：", rel.activate("2026.01", by="E07", reason="首次上線"), " 目前", rel.current())
print("切到 2026.02：", rel.activate("2026.02", by="E07", reason="門檻 0.95→0.97"), " 目前", rel.current())
print("切到 2026.03：", rel.activate("2026.03", by="E07", reason="模型更新"), " 目前", rel.current(), "（自檢失敗，保留舊版）")
print("夜班反映 0.97 漏掉一次跌坐 → 回滾：", rel.rollback(by="E07", reason="林護理長：漏報"), " 目前", rel.current())
print("切換歷程：", rel.history())
print("\n紀錄裡的版本事件：")
for r in log.read():
    if r["event"].startswith(("activate", "startup_refused")):
        keep = {k: r[k] for k in ("event", "release", "previous", "kept", "reason") if k in r}
        print(json.dumps(keep, ensure_ascii=False))
