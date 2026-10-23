# train_model — Leadgen model (VTP × TNEX)

Train **một lần** model dự báo shop trong whitelist (WL) có **nộp hồ sơ vay TNEX** khi được mời, lưu model để dùng lại cho prediction, và xuất toàn bộ bảng/biểu đồ cho báo cáo.

## Bài toán

| | |
|---|---|
| Tập dữ liệu | Shop trong WL từng tháng snapshot 05–08/2026 (`merged_customer_application_fpd.xlsx`, sheet `Merged`) |
| Nhãn | 1 = shop nộp hồ sơ trong tháng snapshot (cột `month` có giá trị), 0 = còn lại |
| Feature | Toàn bộ cột feature store trên MinIO, mỗi thư mục S3 (`features/<nhóm>/<cửa sổ>`) là một **group** |
| Chia tập | Out-of-time: train 05–06/2026, test 07–08/2026 (sửa trong config) |
| Model | CatBoost, tune bằng Optuna |

## Pipeline (`train.py`)

1. **Dữ liệu** — đọc nhãn từ Excel, đọc feature từ S3 (lọc theo `date`, `cus_id` ngay khi quét), cache ở `data/`.
2. **Chọn biến** (chỉ dùng tập train, CV 3 fold theo nhóm `cus_id`):
   1. Lọc thống kê: thiếu > 95%, gần hằng số (> 99,5% một giá trị), cặp |Spearman| > 0,95 (giữ biến AUC đơn biến cao hơn).
   2. Theo từng group: CatBoost trên biến của group, giữ biến tới khi cộng dồn importance ≥ 90%.
   3. Gộp các biến còn lại của mọi group, train lại, giữ top biến tới khi cộng dồn importance ≥ 90%.
3. **Optuna** — tối ưu AUC trung bình CV trên train (depth, learning_rate, l2_leaf_reg, random_strength, bagging_temperature, border_count); số vòng lặp lấy từ early stopping.
4. **Model cuối** — train trên toàn bộ train với tham số tốt nhất, lưu `models/leadgen/model.cbm` + `model_meta.json`.
5. **Đánh giá** — train dùng dự báo OOF của CV (không dùng điểm in-sample), test dùng model cuối.

## Kết quả (`outputs/leadgen/`)

| File | Nội dung |
|---|---|
| `summary.md` | Tóm tắt: số biến qua từng bước, tham số, AUC/Gini theo tháng, precision bin 1, PSI score |
| `label_rate_by_month.png/.csv` | Tỷ lệ nhãn theo thời gian |
| `auc_gini_by_month.png/.csv`, `roc_train_test.png` | AUC/Gini theo tháng (train OOF, test), ROC |
| `shap_top10.png/.csv` | SHAP của 10 biến quan trọng nhất theo CatBoost |
| `feature_importance_by_group.png/.csv`, `feature_importance.csv` | Importance theo group và theo biến |
| `precision_by_decile_test.png/.csv` | Precision theo 10 bin điểm (bin 1 = 10% điểm cao nhất), lift, recall cộng dồn; cả test và từng tháng |
| `psi_score_test.csv`, `psi_features_test.png/.csv` | PSI của score và của từng biến: train vs từng tháng test |
| `selection_1_stat_filter.csv`, `selection_2_by_group.csv`, `selection_3_combined.csv` | Nhật ký chọn biến |
| `optuna_trials.csv`, `optuna_history.png`, `best_params.json` | Kết quả Optuna |
| `report_tables.xlsx` | Gom mọi bảng trên vào một file Excel |

## Chạy

```bash
pip install -r requirements.txt
python train.py --config configs/leadgen.yaml            # train một lần
python predict.py --date 20260801                         # chấm toàn bộ WL của snapshot
python predict.py --date 20260801 --cus-file list.csv     # chấm danh sách cus_id
```

`predict.py` ghi `outputs/predictions/leadgen_<date>.parquet/.csv`: `proba`, `score` (0–100, xếp hạng trong lô), `bin` (1 = 10% điểm cao nhất, theo mốc decile của tập test lúc train).

## Cấu hình

`configs/leadgen.yaml`: đường dẫn file nhãn, danh sách nguồn S3, cột loại trừ, ngưỡng lọc, ngưỡng importance cộng dồn, số trial Optuna, tháng train/test. Credentials MinIO **không** nằm trong repo: đọc từ `connect.yaml` của repo vtp_v2 (đường dẫn trong config); endpoint từ host là `127.0.0.1:9990`.
