"""V35｜交接包：產生與驗證。格式沿用 P18 的 fall-handoff.json（同樣的區塊名稱），兩組各一個資料夾：

    handoff/<組>/handoff.json   P18 的區塊：title、version、privacy、data、eval_set、labeling_rules、
                                known_label_errors、model、threshold、scores、open_issues
                                ＋V35 加的：models（所有要用到的模型檔）、known_model_errors、acceptance
    handoff/<組>/eval-list.csv  評估集清單（P18 只放 hash；這裡把清單本身也交出去，hash 照 P18 的算法）
    handoff/<組>/SHA256SUMS     上面兩個檔的 sha256（`sha256sum -c SHA256SUMS` 可以直接驗）

只放「可以重現與檢查」的東西；不放影像、不放 MVTec 衍生物（記憶庫 .npy 只給 hash，接手方自己重建）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from common import MVTEC_ROOT, ROOT, mvtec_paths

HERE = Path(__file__).resolve().parent
OUT = HERE / "handoff"
P18_PACK = ROOT / "P18-privacy-handoff" / "handoff" / "fall-handoff.json"
P18_BLOCKS = ["title", "version", "privacy", "data", "eval_set", "labeling_rules", "known_label_errors",
              "model", "threshold", "scores", "open_issues"]
for _d in ("V08-anomaly-intro", "V28-mvp-defect", "V32-deploy-checklist", "P16-pose-eval"):
    sys.path.insert(0, str(ROOT / _d))


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(pack) -> str:
    return json.dumps(pack, ensure_ascii=False, indent=1) + "\n"


def write_pack(group, pack, eval_list):
    d = OUT / group
    d.mkdir(parents=True, exist_ok=True)
    (d / "handoff.json").write_text(dump(pack), encoding="utf-8")
    (d / "eval-list.csv").write_text(eval_list, encoding="utf-8")
    sums = "".join(f"{sha_file(d / n)}  {n}\n" for n in ("handoff.json", "eval-list.csv"))
    (d / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return d


# ------------------------------------------------------------------ 製造組

def defect_eval(cat):
    """最終測試（扣掉 V12 的 10 張驗證瑕疵）：清單文字、相對路徑、標籤、類型、V12 快取分數。"""
    from anomaly_lib import scores_patch, valid_test_split

    P = mvtec_paths(cat)
    y = np.asarray(P["y"])
    val, final = valid_test_split(y)
    rel = [str(Path(p).relative_to(MVTEC_ROOT)) for p in P["test"]]
    raw = scores_patch(cat)["test"]
    rows = [(rel[i], int(y[i]), P["defect"][i], round(float(raw[i]), 4)) for i in final]
    rows.sort()
    text = "".join(f"{r},{l}\n" for r, l, _, _ in rows)
    return rows, text, [rel[i] for i in val]


def defect_counts(rows, t_raw):
    """V12 的定義：原始分數 > 門檻＝剔除。"""
    tp = sum(1 for _, y, _, s in rows if y == 1 and s > t_raw)
    fn = sum(1 for _, y, _, s in rows if y == 1 and s <= t_raw)
    fp = sum(1 for _, y, _, s in rows if y == 0 and s > t_raw)
    return {"tp": tp, "fn": fn, "fp": fp, "cost": 20 * fn + fp,
            "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}


def mvtec_data_sha(cat):
    """MVTec 本身不能散佈，只能給 hash：每個檔「相對路徑,sha256」一行，整份清單再算 sha256。"""
    P = mvtec_paths(cat)
    lines = [f"{Path(p).relative_to(MVTEC_ROOT)},{sha_file(p)}\n" for p in sorted(P["train"] + P["test"])]
    return sha_text("".join(lines)), len(lines)


def build_defect(cat="metal_nut"):
    from defect_model import load_model, model_paths
    from deploy import sha256_12

    _, card = load_model(cat)
    npy, _ = model_paths(cat)
    rows, eval_list, val = defect_eval(cat)
    data_sha, n_files = mvtec_data_sha(cat)
    test = defect_counts(rows, card["threshold_raw"])
    miss = [{"item": r, "defect": d, "raw": s, "kind": "漏報"} for r, y, d, s in rows if y == 1 and s <= card["threshold_raw"]]
    fa = [{"item": r, "defect": d, "raw": s, "kind": "誤報"} for r, y, d, s in rows if y == 0 and s > card["threshold_raw"]]
    pack = {
        "title": f"精工五金 一號線 {cat} 瑕疵檢測 交接包（影像 AI 課 → 王課長／維運）",
        "version": card["version"],
        "privacy": {"stored": "事件 JSON（part_id、熱點座標、分數）；工件照片不存", "event_has_image": False,
                    "residual_risk": "無個資；產線照片屬精工五金營業資料，不得帶出廠",
                    "retention": "待精工五金決定（交接時尚未訂定）"},
        "data": {"file": f"MVTec AD {cat}（images/train、images/test）", "sha256": data_sha,
                 "sha256_format": "每個檔「相對 MVTEC_ROOT 的路徑,sha256」一行、依路徑排序，整份再算 sha256",
                 "files": n_files, "license": "CC BY-NC-SA 4.0（非商業；不隨交接包散佈）", "synthetic": False,
                 "split": {"memory": f"train/good 的 80%（{card['n_memory']} 張，common.memory_calib_split）",
                           "calibration": f"train/good 的 20%（{card['n_calib']} 張）",
                           "validation": "test 瑕疵固定抽 10 張（anomaly_lib.valid_test_split, seed 0）",
                           "test": "test 其餘全部（良品＋瑕疵）"}},
        "eval_set": {"category": cat, "items": len(rows), "defects": sum(r[1] for r in rows),
                     "list_format": "每行「相對路徑,is_defect」，依路徑排序，以 \\n 結尾",
                     "list_sha256": sha_text(eval_list), "validation_items": val,
                     "opened": "V13 打開一次（方法 C 定版）；V28、V33 以同一份重跑對帳，不據此調整"},
        "labeling_rules": {"unit": "一張照片＝一顆工件", "source": "沿用 MVTec 的資料夾標籤：good＝0，其餘＝1",
                           "defect_types": sorted({r[2] for r in rows if r[1] == 1}),
                           "caveat": "MVTec 的瑕疵定義不等於精工五金的出貨標準；產線上線前要由王課長重訂"},
        "known_label_errors": [],
        "known_model_errors": miss + fa,
        "model": {"name": card["name"], "version": card["version"], "path": f"cache/models/{npy.name}",
                  "sha256_12": sha256_12(npy), "builder": "V28-mvp-defect/defect_model.py build_model()",
                  "features": "common.resnet_patch_features（ResNet18 layer2+3，224×224）",
                  "bank": f"{card['bank_shape'][0]}×{card['bank_shape'][1]} float16（記憶良品小格抽 10%，rng 0）",
                  "score": f"最不像的小格到最近正常小格的距離；事件分數＝原始分數 ÷ {card['scale']}，上限 1",
                  "redistributable": False},
        "models": [],
        "threshold": {"value": card["threshold"], "raw": card["threshold_raw"], "compare": "事件分數 ≥ value（＝原始分數 > raw）",
                      "chosen_on": "V12 驗證集：校正良品＋10 張驗證瑕疵",
                      "rule": "成本 20×漏報＋1×誤報 最低；同成本取較高門檻（anomaly_lib.cost_threshold）",
                      "approved_by": "王課長", "approved_on": "2026-01-09"},
        "scores": {"validation": {"miss": card["valid_miss"], "fa": card["valid_fa"], "cost": card["valid_cost"]},
                   "test": test,
                   "test_per_item": [{"item": r, "y": y, "defect": d, "raw": s} for r, y, d, s in rows]},
        "acceptance": {"report": "V33", "result": "有條件通過",
                       "shift_estimate": "一班 9600 顆、瑕疵率千分之一：誤剔良品約 872 顆、漏報約 0.23 顆",
                       "screw": "不通過（良品誤剔率 0.976）：只做標記給人看"},
        "open_issues": ["良品誤剔率 0.091 只由 22 顆良品估計；試行期收產線良品重估",
                        "MVTec 是實驗室影像；上線前用精工五金產線照片重建記憶庫、重訂門檻",
                        "記憶庫是 MVTec 衍生物（NC 授權）：商用前要換成產線良品建的記憶庫",
                        "Pi 5 上每顆推論時間未實測（教師機約 0.7–0.9 秒，EZCam 每 3 秒一張）",
                        "screw 不上線；其他類別未評估"]}
    pack["models"] = [{k: pack["model"][k] for k in ("name", "path", "sha256_12")}]
    return pack, eval_list


# ------------------------------------------------------------------ 健康組

def fall_pack():
    import yaml
    from deploy import sha256_12

    sys.path.insert(0, str(ROOT / "V29-mvp-fall"))
    from fall_pipeline import HOLD_S, POSTURE_PATH

    p18 = json.loads(P18_PACK.read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "V32-deploy-checklist/config/fall.yaml").read_text(encoding="utf-8"))
    pack = {k: p18[k] for k in P18_BLOCKS}
    pack["title"] = "松年長照中心 跌倒監測 交接包（影像 AI 課 → 松年／維運）"
    pack["eval_set"] = dict(p18["eval_set"], opened="P16 打開一次（2026-10-06）；V29、V33 以同一份重跑對帳，不據此調整")
    pack["known_model_errors"] = [
        {"item": "SQ0306", "kind": "誤報", "score": next(r["score"] for r in p18["scores"]["test_per_sequence"]
                                                       if r["seq_id"] == "SQ0306"), "why": "標註錯誤（實為跌倒），見 known_label_errors"},
        {"item": "冷卻 60 秒", "kind": "漏通知", "why": "V33 模擬 30 晚有 10 件真跌倒被同一人 60 秒內的前一則通知吞掉"}]
    pack["models"] = [
        {"name": "fall-hgb", "path": p18["model"]["path"], "sha256_12": p18["model"]["sha256_12"], "from": "P16"},
        {"name": "posture-hgb", "path": f"models/{POSTURE_PATH.name}", "sha256_12": sha256_12(POSTURE_PATH),
         "from": "V29 fall_pipeline.build_posture_model()（S01–S18、四類姿勢，只給狀態機用）"},
        {"name": "yolo11n-pose", "path": "models/yolo11n-pose.pt", "sha256_12": sha256_12(ROOT / "models/yolo11n-pose.pt"),
         "from": "Ultralytics（AGPL-3.0）"}]
    th = cfg["threshold"]
    pack["threshold"] = {k: str(th[k]) if k == "approved_on" else th[k] for k in ("value", "chosen_on", "rule", "approved_by", "approved_on")}
    pack["pipeline"] = {"config": "V32-deploy-checklist/config/fall.yaml", "cooldown_s": cfg["cooldown_s"],
                        "state_machine": "V21／V22：狀態機告警一次就重置", "hold_s_in_eval": HOLD_S,
                        "cameras": [c["id"] for c in cfg["cameras"]]}
    pack["acceptance"] = {"report": "V33", "result": "有條件通過",
                          "per_night": "模擬 30 晚（S19–S24、門檻 0.95）：通知 14.4、誤報 3.7、漏通知 10 件／30 晚",
                          "condition": "冷卻改以人為單位（V20 追蹤編號）"}
    pack["open_issues"] = [i for i in p18["open_issues"] if "待林護理長核准" not in i] + [
        "P16 定版卡 approved_by 仍是「待林護理長核准」；核准紀錄目前只在 V32 設定檔",
        "冷卻以攝影機為單位，會吞掉真跌倒（V33）",
        "ultralytics／yolo11n-pose 是 AGPL-3.0：交付松年前由法務確認（P12）"]
    eval_list = "\n".join(f"{r['seq_id']},{r['y']}" for r in p18["scores"]["test_per_sequence"])
    return pack, eval_list


# ------------------------------------------------------------------ 驗證（接手方跑的）

def verify(d, base=ROOT):
    """回傳 [(項目, 是否一致, 說明)]。只用交接包＋接手方自己的資料與模型檔。"""
    from deploy import sha256_12

    d = Path(d)
    out = []
    sums = dict(reversed(l.split("  ")) for l in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    for name, h in sums.items():
        out.append((f"SHA256SUMS：{name}", sha_file(d / name) == h, ""))
    pack = json.loads((d / "handoff.json").read_text(encoding="utf-8"))
    missing = [k for k in P18_BLOCKS if k not in pack]
    out.append(("與 P18 區塊相同", not missing, f"缺 {missing}" if missing else ""))
    eval_list = (d / "eval-list.csv").read_text(encoding="utf-8")
    out.append(("評估集清單 sha256", sha_text(eval_list) == pack["eval_set"]["list_sha256"], ""))
    for m in pack["models"]:
        p = base / m["path"]
        got = sha256_12(p) if p.exists() else "（檔案不在）"
        out.append((f"模型 {m['name']}", got == m["sha256_12"], f"宣稱 {m['sha256_12']}，實際 {got}"))
    if "category" in pack["eval_set"]:                                # 製造組
        rows, _, _ = defect_eval(pack["eval_set"]["category"])
        listed = [tuple(l.split(",")) for l in eval_list.splitlines()]
        out.append(("清單＝最終測試（無驗證瑕疵）", listed == [(r, str(y)) for r, y, _, _ in rows]
                    and not set(pack["eval_set"]["validation_items"]) & {r for r, _ in listed}, ""))
        data_sha, _ = mvtec_data_sha(pack["eval_set"]["category"])
        out.append(("MVTec 資料 sha256", data_sha == pack["data"]["sha256"], ""))
        out.append(("測試分數可重現", defect_counts(rows, pack["threshold"]["raw"]) == pack["scores"]["test"], ""))
        from defect_model import load_model
        card = load_model(pack["eval_set"]["category"])[1]
        out.append(("門檻＝模型卡", (card["threshold"], card["threshold_raw"]) == (pack["threshold"]["value"], pack["threshold"]["raw"]),
                    f"交接包 {pack['threshold']['value']}，模型卡 {card['threshold']}"))
    else:                                                             # 健康組
        from build_fall_model import counts, seq_meta, seq_score
        import joblib
        from common import DATA, load_poses, subject_split

        out.append(("poses.csv sha256", sha_file(DATA / "pose/poses.csv") == pack["data"]["sha256"], ""))
        _, test = subject_split(load_poses().sort_values(["seq_id", "frame"]))
        tm = seq_meta(test).assign(score=seq_score(joblib.load(base / pack["model"]["path"]), test).round(4))
        listed = "\n".join(f"{s},{int(y)}" for s, y in tm.y.items())
        out.append(("清單＝S19–S24 序列", listed == eval_list, ""))
        out.append(("測試分數可重現", counts(tm.y, tm.score, pack["threshold"]["value"]) == pack["scores"]["test"], ""))
        import yaml
        cfg = yaml.safe_load((base / pack["pipeline"]["config"]).read_text(encoding="utf-8"))
        out.append(("門檻＝部署設定檔", cfg["threshold"]["value"] == pack["threshold"]["value"],
                    f"交接包 {pack['threshold']['value']}，設定檔 {cfg['threshold']['value']}"))
    return out
