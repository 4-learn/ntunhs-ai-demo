# P10｜從手寫濾波器到卷積層

把 P06 的 Sobel 核放進 torch.nn.Conv2d、驗證與 cv2 相同；多核、多通道；ResNet18 conv1 的 64 個學習核統計；參數量計算；AI 常見錯誤實跑。ResNet18 權重需已在 ~/.cache/torch/hub/checkpoints/。

    python P10-conv-layer/conv_layer.py
    python P10-conv-layer/pitfalls.py
