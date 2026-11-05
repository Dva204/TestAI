# Dataset v2: 5 chuyên khoa (Mắt, Nội tổng quát, Ngoại, Răng Hàm Mặt, Tai Mũi Họng)

Phiên bản mở rộng dữ liệu tổng hợp cho bài toán dự đoán bệnh y khoa và định tuyến vào một trong năm chuyên khoa:
1. **Mắt (Nhãn khoa)**
2. **Nội tổng quát**
3. **Ngoại**
4. **Răng Hàm Mặt**
5. **Tai Mũi Họng**

## Quy mô tập dữ liệu
- **160 bệnh lý**: Cân bằng 32 bệnh cho mỗi chuyên khoa.
- **16,000 dòng mẫu**: 100 mẫu tổng hợp cho từng bệnh.
- **849 đặc trưng**:
  - 731 triệu chứng và dấu hiệu lâm sàng (`clinical_symptom_or_sign`).
  - 24 dấu hiệu thăm khám thực thể (`exam`).
  - 23 dấu hiệu phủ định rõ ràng (`negative`, tiền tố `neg_*`).
  - 23 yếu tố nguy cơ (`risk`, tiền tố `risk_*`).
  - 22 dấu hiệu cảnh báo khẩn cấp (`red_flag`, tiền tố `red_*`).
  - 8 đặc trưng thời gian diễn tiến (`duration`).
  - 7 đặc trưng nhân khẩu học & nhóm tuổi (`demographic`).
  - 6 kiểu khởi phát (`onset`).
  - 5 mức độ nghiêm trọng (`severity`).

## Các file dữ liệu
- `training_data_5_specialties_v2.csv`: File dữ liệu huấn luyện gồm 849 cột đặc trưng nhị phân (0 hoặc 1) và cột nhãn mục tiêu `prognosis`.
- `disease_specialty_map_v2.csv`: Bảng ánh xạ 160 mã bệnh (`disease_key`) sang tên tiếng Việt (`disease_name_vi`) và chuyên khoa tương ứng (`specialty`).
- `feature_dictionary_v2.csv`: Bảng tra cứu nhóm và ý nghĩa của từng đặc trưng trong 849 đặc trưng.
- `generate_dataset_v2.py`: Mã nguồn Python tái tạo tập dữ liệu với seed cố định (`20261024`).

## Quy ước dữ liệu thiếu và phủ định
Trong cột triệu chứng thông thường, `0` nghĩa là **không được ghi nhận** (chưa được hỏi hoặc chưa ghi vào hồ sơ), không đồng nghĩa với việc bệnh nhân chắc chắn không có triệu chứng đó. Những triệu chứng đã hỏi và được ghi nhận là âm tính được lưu ở cột `neg_*` riêng biệt (ví dụ `neg_no_fever=1`).

## Hướng dẫn huấn luyện trên Kaggle
- Nhãn mục tiêu: cột `prognosis` (160 lớp bệnh).
- Khi chia tập Train/Validation/Test, sử dụng `stratify=y` theo cột `prognosis` để đảm bảo mỗi lớp phân bố đều.
- Thuật toán khuyến nghị: `RandomForestClassifier`, `LightGBM`, `XGBoost`, hoặc `CatBoost`.
- Đánh giá mô hình: Sử dụng Macro F1, Balanced Accuracy, và Confusion Matrix bên cạnh Accuracy thuần túy.
