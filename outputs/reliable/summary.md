# Reliable score — 2026-10-23 14:38

- Nhãn: bad = ngừng giao hoặc đơn 3 tháng tới < 0.3 × baseline (đã chia hệ số thị trường). Prediction set: đơn đủ 3 tháng, TB ≥ 5 đơn/tháng, thâm niên ≥ 3 tháng.
- Cửa sổ trượt: train 3 cohort → test cohort kế tiếp; chọn biến lại trong mỗi cửa sổ; tham số Optuna tune một lần trên train cửa sổ đầu (0.8178 AUC CV).

| Cửa sổ | Train | Test | Nhãn test | n test | Tỷ lệ bad | AUC | Gini | KS | AUC S0 | PSI score | Số biến |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20251201–20260201 | 20260301 | full | 220,164 | 11.48% | 0.7771 | 0.5543 | 0.4125 | 0.7941 | 1.1692 | 46 |
| 2 | 20260101–20260301 | 20260401 | full | 216,078 | 12.68% | 0.8239 | 0.6478 | 0.4928 | 0.8397 | 0.0178 | 50 |
| 3 | 20260201–20260401 | 20260501 | full | 204,431 | 12.42% | 0.8202 | 0.6405 | 0.4837 | 0.8363 | 0.0174 | 50 |
| 4 | 20260301–20260501 | 20260601 | partial_2m | 221,786 | 13.05% | 0.8101 | 0.6203 | 0.4625 | 0.8297 | 0.0075 | 50 |

- Độ ổn định (3 cửa sổ nhãn đủ): AUC 0.8071 ± 0.0212 (min 0.7771, max 0.8239); Gini 0.6142 ± 0.0425.
- Model cuối train trên [20260301, 20260401, 20260501], 50 biến → `models/reliable/model.cbm`.
- Thời gian chạy: 105.6 phút.