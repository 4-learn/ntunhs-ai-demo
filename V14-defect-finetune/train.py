"""V14｜教師開課前跑一次（或課堂示範）：metal_nut 良品／瑕疵二分類微調，存到 models/。

    python V14-defect-finetune/train.py

產出 models/defect-cls-mnv3s.pt 與 models/defect-cls-mnv3s.json（loss、設定、耗時、sha256_12）。
講義的 output 區塊指向 evaluate.py（讀這兩個檔），不指向這支，因為訓練秒數每台機器不同。
"""
import hashlib
import json
import time

import torch

from defectcls import (CAT, META_FILE, MODEL_FILE, build_model, frozen_front, load_images, n_params, split_table,
                       train)

EPOCHS, BATCH, LR, UNFREEZE = 8, 8, 1e-3, 9

df = split_table()
tr = df[df.split == "train"]
print(f"== 訓練：{CAT} train {len(tr)} 張（瑕疵 {tr.y.sum()}）==")
X = load_images(tr.path)
model = build_model(UNFREEZE)
total, trainable = n_params(model)
print(f"全部參數 {total:,}；要訓練 {trainable:,}")
t = time.time()
F4 = frozen_front(model, X, UNFREEZE)
del X
hist = train(model, F4, tr.y.to_numpy(), epochs=EPOCHS, batch=BATCH, lr=LR, split=UNFREEZE,
             log=lambda ep, loss: print(f"epoch {ep}: loss {loss:.3f}", flush=True))
sec = time.time() - t
model.eval()
torch.save(model.state_dict(), MODEL_FILE)
sha = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()[:12]
meta = dict(base="torchvision mobilenet_v3_small IMAGENET1K_V1", category=CAT, unfreeze_from=UNFREEZE,
            epochs=EPOCHS, batch=BATCH, lr=LR, imgsz=224, augment="rot90", seed=0,
            torch_threads=torch.get_num_threads(), params_total=total, params_trainable=trainable,
            train_images=len(tr), train_defects=int(tr.y.sum()), loss=[round(v, 4) for v in hist],
            train_seconds=round(sec), sha256_12=sha, torch=torch.__version__)
META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
print(f"已存 {MODEL_FILE.name}（sha256_12 {sha}），訓練 {sec:.0f} 秒")
