"""V16｜讀教師訓練好的瑕疵偵測器：訓練紀錄（.json）、loss 曲線（results.csv）、在 val 上 model.val()。

    python V16-yolo-train/read_record.py

訓練本身由教師先跑：PYTHONPATH=. python tools/train_yolo_defects.py（多人共用 CPU 上要數十分鐘以上）。
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")

import pandas as pd
import torch
from ultralytics import YOLO

from common import CACHE, MODELS, ROOT

torch.set_num_threads(2)
DS = CACHE / "yolo-defects"
W = MODELS / "defect-yolo11n.pt"
meta = json.loads((MODELS / "defect-yolo11n.json").read_text(encoding="utf-8"))

print("== 1. 訓練紀錄（models/defect-yolo11n.json）==")
for k in ("base", "epochs", "freeze", "imgsz", "batch", "seed", "deterministic", "torch_threads", "sha256_12"):
    print(f"{k:14s} {meta[k]}")
if "note" in meta:
    print(f"{'note':14s} {meta['note']}")


net = YOLO(str(W))
layers = net.model.model
total = sum(p.numel() for p in layers.parameters())
frozen = sum(p.numel() for p in layers[:meta["freeze"]].parameters())
print(f"\n模型共 {len(layers)} 層、{total:,} 個參數；freeze={meta['freeze']} 凍結第 0–{meta['freeze'] - 1} 層"
      f"（{frozen:,}，{frozen / total:.0%}）")
print(f"第 {meta['freeze']} 層起要訓練：{total - frozen:,} 個參數；類別 {net.names}，授權 {net.ckpt.get('license', '?')}")

csv = ROOT / "runs/defect/results.csv"
if csv.exists():
    r = pd.read_csv(csv)
    r.columns = [c.strip() for c in r.columns]
    print("\n== 2. 每個 epoch 的 loss 與 val 指標（runs/defect/results.csv）==")
    cols = {"epoch": "epoch", "train/box_loss": "box", "train/cls_loss": "cls", "train/dfl_loss": "dfl",
            "val/box_loss": "val_box", "val/cls_loss": "val_cls", "metrics/mAP50(B)": "mAP50"}
    t = r[list(cols)].rename(columns=cols)
    show = t[(t.epoch <= 3) | (t.epoch % 5 == 0) | (t.epoch == t.epoch.max())]
    print(show.round(3).to_string(index=False))
    b = t.loc[t.mAP50.idxmax()]
    k = int(r.loc[r["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    print(f"val mAP50 最高在 epoch {int(b.epoch)}：{b.mAP50:.3f}；mAP50-95 最高在 epoch {k}（best.pt 依這個選）")

print("\n== 3. model.val() 在 val 54 張 ==")
tmp = Path(tempfile.mkdtemp(prefix="yolo-val-"))
for sub in ("images", "labels"):
    shutil.copytree(DS / sub / "val", tmp / sub / "val")  # val() 會寫 labels/*.cache，不動原資料夾
(tmp / "d.yaml").write_text(f"path: {tmp}\ntrain: images/val\nval: images/val\nnames:\n  0: defect\n")
m = YOLO(str(W)).val(data=str(tmp / "d.yaml"), split="val", imgsz=meta["imgsz"], batch=8, conf=0.001, iou=0.7,
                     device="cpu", workers=0, plots=False, verbose=False, project=str(tmp), name="val")
print(f"defect：P {m.box.mp:.3f}  R {m.box.mr:.3f}  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
shutil.rmtree(tmp, ignore_errors=True)
