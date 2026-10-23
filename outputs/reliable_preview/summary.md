# Reliable score — 2026-10-23 12:43

- Nhãn: bad = ngừng giao hoặc đơn 3 tháng tới < 0.3 × baseline (đã chia hệ số thị trường). Prediction set: đơn đủ 3 tháng, TB ≥ 5 đơn/tháng, thâm niên ≥ 3 tháng.
- Cửa sổ trượt: train 3 cohort → test cohort kế tiếp; chọn biến lại trong mỗi cửa sổ; tham số Optuna tune một lần trên train cửa sổ đầu (0.7931 AUC CV).

| Cửa sổ | Train | Test | Nhãn test | n test | Tỷ lệ bad | AUC | Gini | KS | AUC S0 | PSI score | Số biến |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20251201–20260201 | 20260301 | full | 220,164 | 11.48% | 0.7701 | 0.5403 | 0.3998 | 0.7848 | 0.9971 | 46 |
| 2 | 20260101–20260301 | 20260401 | full | 216,078 | 12.68% | 0.8077 | 0.6155 | 0.4615 | 0.8241 | 0.0143 | 51 |
| 3 | 20260201–20260401 | 20260501 | full | 204,431 | 12.42% | 0.8017 | 0.6035 | 0.4499 | 0.8180 | 0.0250 | 50 |
| 4 | 20260301–20260501 | 20260601 | partial_2m | 221,786 | 13.05% | 0.7931 | 0.5863 | 0.4287 | 0.8131 | 0.0162 | 49 |

- Độ ổn định (3 cửa sổ nhãn đủ): AUC 0.7932 ± 0.0165 (min 0.7701, max 0.8077); Gini 0.5864 ± 0.0330.
- Model cuối train trên [20260301, 20260401, 20260501], 49 biến → `(model bản preview không lưu vào repo)`.
- Thời gian chạy: 12.2 phút.