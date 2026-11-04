# Leadgen model — 2026-10-25 11:27

- Nhãn: shop trong WL nộp hồ sơ trong tháng snapshot. Train [20260501, 20260601], test [20260701, 20260801] (out-of-time).
- Chọn biến: 210 → stat_filter 95 → vote 61 → cum_importance 44: **44 biến**.
- Optuna 5 trial, AUC CV train tốt nhất 0.8836. Tham số: `{"learning_rate": 0.03920673972242137, "depth": 7, "l2_leaf_reg": 1.9721610970573997, "random_strength": 2.571172192068058, "bagging_temperature": 2.9620728443102124, "border_count": 128, "iterations": 264}`

| Tháng | Tập | n | Nộp hồ sơ | AUC | Gini |
|---|---|---|---|---|---|
| 20260501 | train | 53,427 | 205 | 0.9083 | 0.8167 |
| 20260601 | train | 52,650 | 467 | 0.8662 | 0.7325 |
| 20260701 | test | 51,039 | 340 | 0.8840 | 0.7680 |
| 20260801 | test | 50,010 | 243 | 0.8662 | 0.7324 |
| all_test | test | 101,049 | 583 | 0.8767 | 0.7533 |
| all_train | train | 106,077 | 672 | 0.8801 | 0.7602 |

- Bin 1 (10% điểm cao nhất) trên test: precision 3.68%, lift ×6.38, bắt 63.8% số shop nộp hồ sơ.
- PSI score: 20260701 0.0028, 20260801 0.0053, all_test 0.0037
- Thời gian chạy: 10.1 phút.