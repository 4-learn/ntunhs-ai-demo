"""V34｜互審清單＋自動檢查。一組的「繳交資料夾」長這樣：

    submission.yaml   組別、模型檔與 hash、門檻（值、在哪裡選、規則、核准）、資料切割、宣稱的成績
    test_log.jsonl    每打開一次測試集就寫一行（誰、何時、做什麼）
    events.jsonl      Demo 時送出的事件（V27 格式）
    NOTICE.md         用到的資料與套件授權
    （模型檔放在 submission.yaml 指的路徑；資料夾裡不應該有影像或 MVTec 衍生物）

    from review import CHECKLIST, review
    for item, ok, msg in review(path): ...

CHECKLIST 有 10 項，其中 8 項可以寫成程式（auto=True）；另外 2 項要人看（Demo 時現場問）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from common import ROOT, TEST_SUBJECTS

sys.path.insert(0, str(ROOT / "V27-event-api"))
sys.path.insert(0, str(ROOT / "V32-deploy-checklist"))
from deploy import sha256_12  # noqa: E402
from event_api import Event  # noqa: E402

CHECKLIST = [
    ("資料洩漏：訓練／選門檻用的資料與測試資料不重疊", True),
    ("門檻在驗證資料上選、規則寫得出來", True),
    ("測試集只打開一次", True),
    ("模型檔 hash 與繳交資料相符", True),
    ("門檻由客戶核准（不是工程師自己）", True),
    ("隱私：事件與資料夾都沒有影像（只有骨架／編號）", True),
    ("授權：NOTICE 寫明 MVTec（CC BY-NC-SA）與 ultralytics（AGPL-3.0）", True),
    ("宣稱的成績可以從繳交資料重算出來", True),
    ("標註規則前後一致、已知標註錯誤有列出", False),
    ("Demo 有展示失敗案例，不只挑成功的", False),
]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".mp4"}
DERIVED_EXT = {".npy", ".npz", ".pt"}          # 記憶庫、特徵：MVTec 衍生物（NC 授權，不能跟著交付）
TEST_WORDS = ("test", "測試", *TEST_SUBJECTS)


def _load(path: Path):
    sub = yaml.safe_load((path / "submission.yaml").read_text(encoding="utf-8"))
    log = [json.loads(x) for x in (path / "test_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ev = [json.loads(x) for x in (path / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    notice = (path / "NOTICE.md").read_text(encoding="utf-8") if (path / "NOTICE.md").exists() else ""
    return sub, log, ev, notice


def check_leak(sub):
    s = sub["split"]
    if sub["source"] == "pose":
        a = set(s["train_subjects"]) | set(s["threshold_subjects"])
        both = sorted(a & set(s["test_subjects"]))
        return not both, f"重疊的受試者 {both}" if both else "訓練 S01–S18、測試 S19–S24，不重疊"
    a = set(s["memory_index"]) & set(s["eval_index_train"]) if "eval_index_train" in s else set()
    b = sorted(set(s["threshold_index"]) & set(s["eval_index"]))
    bad = sorted(a) + b
    return not bad, f"選門檻用過、又拿來測試的影像 {len(b)} 張" if bad else "選門檻的 10 張驗證瑕疵不在最終測試裡"


def check_threshold(sub):
    th = sub["threshold"]
    where, rule = str(th.get("chosen_on", "")), str(th.get("rule", ""))
    on_test = any(w in where for w in TEST_WORDS)
    ok = bool(where) and bool(rule) and not on_test
    return ok, f"chosen_on：{where or '（空白）'}"


def check_test_opened(log):
    n = sum(1 for r in log if r.get("set") == "test")
    return n <= 1, f"test_log 記錄 {n} 次"


def check_hash(sub, base):
    p = base / sub["model"]["path"]
    if not p.exists():
        return False, f"找不到 {sub['model']['path']}"
    got = sha256_12(p)
    return got == sub["model"].get("sha256_12"), f"宣稱 {sub['model'].get('sha256_12')}，實際 {got}"


def check_approved(sub):
    who = str(sub["threshold"].get("approved_by", ""))
    ok = bool(who.strip()) and "待" not in who and who not in {sub.get("group", ""), "工程師", "我們"}
    return ok, f"approved_by：{who or '（空白）'}"


def check_privacy(ev, path):
    bad = []
    for e in ev:
        try:
            Event.model_validate(e)
        except Exception as x:                                   # noqa: BLE001
            bad.append(f"{e.get('event_id', '?')}：{str(x).splitlines()[2].strip()[:40]}")
    files = [f.name for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXT]
    ok = not bad and not files
    return ok, "；".join(bad[:1] + ([f"影像檔 {files[:2]}"] if files else [])) or f"事件 {len(ev)} 筆都通過、沒有影像檔"


def check_license(sub, notice, path):
    need = ["CC BY-NC-SA"] if sub["source"] == "defect" else []
    if sub.get("uses_ultralytics", sub["source"] == "pose"):
        need.append("AGPL")
    miss = [n for n in need if n not in notice]
    derived = [f.name for f in path.rglob("*") if f.suffix.lower() in DERIVED_EXT] if sub["source"] == "defect" else []
    ok = not miss and not derived
    msg = []
    if miss:
        msg.append(f"NOTICE 缺 {miss}")
    if derived:
        msg.append(f"資料夾內有 MVTec 衍生物 {derived}")
    return ok, "；".join(msg) or f"NOTICE 有 {need}"


def check_claims(sub, rescore):
    """rescore(sub) → dict(tp, fn, fp)：用繳交的模型與門檻，在宣稱的評估集上重算。"""
    got = rescore(sub)
    claim = {k: sub["claims"][k] for k in ("tp", "fn", "fp")}
    return got == claim, f"宣稱 {claim}，重算 {got}"


def review(path, rescore, base=ROOT):
    path = Path(path)
    sub, log, ev, notice = _load(path)
    res = [check_leak(sub), check_threshold(sub), check_test_opened(log), check_hash(sub, base),
           check_approved(sub), check_privacy(ev, path), check_license(sub, notice, path), check_claims(sub, rescore)]
    out = [(item, ok, msg) for (item, _), (ok, msg) in zip(CHECKLIST, res)]
    out += [(item, None, "人工") for item, auto in CHECKLIST if not auto]
    return out
