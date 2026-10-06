"""V24 挑錯：AI 寫的匯出程式常見的三個錯，實際跑出來。

A. 用 == 比對；參考的 PyTorch 模型忘了 .eval()（BatchNorm 用「這一批」的平均），誤以為匯出壞了
B. 沒設 dynamic_axes：batch 被寫死成 1，上線一次送 4 張就報錯
C. 前處理沒對齊：ONNX 檔沒錯，但送進去的是 0–255、沒正規化的像素

暫存檔寫在 /tmp/ntunhs/V24/。

    python V24-onnx-export/pitfalls.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "V24-onnx-export")
from edge_lib import (OPSET, export_r18, ort_patches, ort_session, preprocess, r18_patch_model,  # noqa: E402
                      rel_order)

from common import mvtec_paths  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(0)
TMP = Path("/tmp/ntunhs/V24")
TMP.mkdir(parents=True, exist_ok=True)

P = mvtec_paths("bottle")
x = preprocess(P["test"][:4])
ref_model = r18_patch_model()
with torch.no_grad():
    ref = ref_model(torch.from_numpy(x)).numpy().transpose(0, 2, 3, 1)


def rel(a):
    return float(np.abs(a - ref).max() / np.abs(ref).max())


def export(model, path, axes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (torch.from_numpy(x[:1]),), str(path), dynamo=False, opset_version=OPSET,
                          input_names=["image"], output_names=["patches"], dynamic_axes=axes)


print("== A. 用 == 比對，而且 PyTorch 端忘了 .eval() ==")
sess = ort_session(export_r18())
y_ort = ort_patches(sess, x)
print("np.array_equal(onnx, pytorch)：", np.array_equal(y_ort, ref), "← 浮點數不會逐位元相同")
print("np.allclose(onnx, pytorch, rtol=1e-3, atol=1e-4)：", np.allclose(y_ort, ref, rtol=1e-3, atol=1e-4))
m = r18_patch_model().train()  # 參考模型忘了 .eval()：BatchNorm 改用「這一批」的平均
with torch.no_grad():
    ref_train = m(torch.from_numpy(x)).numpy().transpose(0, 2, 3, 1)
print(f"和「沒 .eval() 的 PyTorch」比：相對誤差 {float(np.abs(y_ort - ref_train).max() / np.abs(ref_train).max()):.2f}"
      "（看起來像匯出壞了，其實是參考答案錯）")

print("\n== B. 沒設 dynamic_axes，batch 被寫死 ==")
export(r18_patch_model(), TMP / "fixed.onnx", None)
s = ort_session(TMP / "fixed.onnx")
print("輸入形狀：", s.get_inputs()[0].shape)
try:
    s.run(None, {"image": x[:4]})
    print("batch 4：成功")
except Exception as e:  # onnxruntime 的 InvalidArgument
    print("batch 4：失敗，", type(e).__name__)

print("\n== C. 前處理沒對齊（送 0–255 原始像素）==")
from PIL import Image  # noqa: E402

raw = np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224)), np.float32).transpose(2, 0, 1)
                for p in P["test"][:4]])
good = sess
print("正確前處理：相對誤差", rel_order(ort_patches(good, x), ref))
print(f"沒正規化：  相對誤差 {rel(ort_patches(good, raw)):.1f}")
