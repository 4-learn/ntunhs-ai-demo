# V20｜追蹤：哪一副骨架是同一個人

把兩段不同受試者的骨架序列疊成「一個畫面兩個人」、每格順序打亂，用 IoU／中心距離＋匈牙利法（`scipy.optimize.linear_sum_assignment`，沒有 scipy 時退回貪婪法）追蹤並計算 ID switch。

```bash
python V20-tracking/track.py
```
