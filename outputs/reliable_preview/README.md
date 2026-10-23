# Reliable score — bản PREVIEW (cấu hình rút gọn)

Kết quả chạy thử để kiểm tra pipeline, **không phải kết quả chính thức**: Optuna chỉ 2 trial, CatBoost chọn biến 100 vòng, mẫu chọn biến 30 nghìn dòng. Kết quả chính thức sẽ nằm ở `outputs/reliable/` (Optuna 25 trial, mẫu 200 nghìn dòng).

Đã thấy ở bản này:
- 4 cửa sổ: test 03/2026, 04/2026, 05/2026 (nhãn đủ) và 06/2026 (nhãn tạm 2 tháng).
- Cửa sổ test 03/2026 có PSI score ~1,0 và AUC thấp nhất: tháng test này dùng dữ liệu tháng 2 (Tết), đơn giảm mạnh làm model chấm rủi ro cao cho gần như mọi shop. Thứ hạng vẫn giữ (AUC 0,77) nhưng mức điểm bị đẩy lên. Nhãn đã khử Tết bằng hệ số thị trường, feature thì chưa.
