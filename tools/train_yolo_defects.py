"""教師開課前跑一次：在 cache/yolo-defects 上微調 COCO 預訓練 YOLO11n（V16）。

    PYTHONPATH=. python tools/train_yolo_defects.py

產出 models/defect-yolo11n.pt 與 models/defect-yolo11n.json（訓練設定、耗時、sha256_12）。
"""
import hashlib
import json
import os
import shutil
import time

import torch
from ultralytics import YOLO

import common as C

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
cfg = dict(data=str(C.CACHE / "yolo-defects/data.yaml"), epochs=20, imgsz=320, batch=8, freeze=10, device="cpu",
           workers=0, seed=0, deterministic=True, project=str(C.ROOT / "runs"), name="defect", exist_ok=True,
           plots=False, verbose=False)
t = time.time()
model = YOLO(str(C.MODELS / "yolo11n.pt"))
model.train(**cfg)
sec = time.time() - t
best = C.ROOT / "runs/defect/weights/best.pt"
dst = C.MODELS / "defect-yolo11n.pt"
shutil.copy(best, dst)
sha = hashlib.sha256(dst.read_bytes()).hexdigest()[:12]
meta = dict(base="yolo11n.pt (COCO)", train_seconds=round(sec), torch_threads=torch.get_num_threads(),
            sha256_12=sha, **{k: v for k, v in cfg.items() if k not in ("project", "name", "exist_ok")})
(C.MODELS / "defect-yolo11n.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
