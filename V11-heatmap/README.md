# V11｜熱圖：瑕疵在哪裡

28×28 小格分數放大回 224×224 熱圖、最大點座標、用遮罩算 pixel AUROC 與最大點命中率；疊圖存到 /tmp/ntunhs/V11/。讀 V10 的分數快取。

    python V11-heatmap/heatmap.py
    python V11-heatmap/pitfalls.py
