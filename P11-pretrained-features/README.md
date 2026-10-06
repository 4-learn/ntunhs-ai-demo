# P11｜預訓練 CNN 在看什麼：特徵向量

ResNet18（ImageNet）拿掉最後一層 → 每張圖 512 維向量（features.py，有磁碟快取）；MVTec bottle／screw 的良品與瑕疵距離、最近鄰；AI 常見錯誤實跑。需要 ~/datasets/mvtec-ad 與 ResNet18 權重；第一次執行約 5 分鐘（算特徵），之後讀快取。

    python P11-pretrained-features/pretrained_features.py
    python P11-pretrained-features/pitfalls.py
