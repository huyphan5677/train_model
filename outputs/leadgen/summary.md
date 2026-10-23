# Leadgen model — 2026-10-23 09:55

- Nhãn: shop trong WL nộp hồ sơ trong tháng snapshot. Train [20260501, 20260601], test [20260701, 20260801] (out-of-time).
- Chọn biến: 210 → lọc thống kê 95 → theo group 80 → gộp, cộng dồn 90% importance: **57 biến**.
- Optuna 40 trial, AUC CV train tốt nhất 0.8838. Tham số: `{"learning_rate": 0.05436251374235174, "depth": 7, "l2_leaf_reg": 11.360666215395595, "random_strength": 0.8770160530216139, "bagging_temperature": 0.02023183370342043, "border_count": 254, "iterations": 276}`

| Tháng | Tập | n | Nộp hồ sơ | AUC | Gini |
|---|---|---|---|---|---|
| 20260501 | train | 53,427 | 205 | 0.9144 | 0.8288 |
| 20260601 | train | 52,650 | 467 | 0.8692 | 0.7384 |
| 20260701 | test | 51,039 | 340 | 0.8886 | 0.7771 |
| 20260801 | test | 50,010 | 243 | 0.8721 | 0.7443 |
| all_test | test | 101,049 | 583 | 0.8818 | 0.7636 |
| all_train | train | 106,077 | 672 | 0.8840 | 0.7680 |

- Bin 1 (10% điểm cao nhất) trên test: precision 3.77%, lift ×6.54, bắt 65.4% số shop nộp hồ sơ.
- PSI score: 20260701 0.0020, 20260801 0.0031, all_test 0.0022
- Thời gian chạy: 38.5 phút.