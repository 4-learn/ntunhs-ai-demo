# V23｜換一間機構就不準：漂移

同一個模型搬到二館（`poses-site2.csv`，攝影機更高更暗、只有約 30% 序列有標籤）：特徵分布、無標籤指標、有標籤序列上的 recall，以及「重新校正門檻 vs 收資料重訓」。

```bash
python V23-site-drift/drift.py
python V23-site-drift/adapt.py
```
