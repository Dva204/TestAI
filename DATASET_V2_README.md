# Dataset v2: 5 chuyên khoa

Phiên bản v2 mở rộng dữ liệu tổng hợp cho bài toán dự đoán bệnh, sau đó ánh xạ bệnh sang một trong năm chuyên khoa: Nội tổng quát, Ngoại, Nhi, Răng Hàm Mặt và Tai Mũi Họng.

Phạm vi hiện tại: 100 bệnh (20 bệnh mỗi chuyên khoa), 450 đặc trưng và 100 mẫu tổng hợp cho mỗi bệnh.

## Cấu trúc

- `training_data_5_specialties_v2.csv`: đặc trưng nhị phân và nhãn `prognosis`.
- `disease_specialty_map_v2.csv`: tên bệnh tiếng Việt và chuyên khoa.
- `feature_dictionary_v2.csv`: nhóm và ý nghĩa của từng đặc trưng.
- `generate_dataset_v2.py`: script tái tạo dữ liệu bằng seed cố định.

Các nhóm đặc trưng gồm triệu chứng/dấu hiệu, tuổi, giới, thời gian, kiểu khởi phát, mức độ, dấu hiệu khám, yếu tố nguy cơ, dấu hiệu cảnh báo và dấu hiệu âm tính.

## Quy ước dữ liệu thiếu và phủ định

Trong cột triệu chứng thông thường, `0` nghĩa là **không được ghi nhận**, không được diễn giải là bệnh nhân chắc chắn không có triệu chứng. Một phủ định rõ ràng được lưu bằng cột `neg_*` riêng, ví dụ `neg_no_fever=1`.

## Cảnh báo

Đây là dữ liệu tổng hợp để phát triển và kiểm thử pipeline. Không phải hồ sơ bệnh án, không đại diện tỷ lệ lưu hành thực tế và chưa được bác sĩ thẩm định. Không sử dụng để chẩn đoán, cấp cứu hoặc điều trị. Trước khi triển khai phải có bác sĩ từng chuyên khoa rà soát bệnh, triệu chứng, dấu hiệu âm tính, quy tắc chuyển khoa và ngưỡng cảnh báo.

Khi đánh giá trên Kaggle, chia dữ liệu theo `prognosis` bằng `stratify`, tránh tối ưu trên tập test và báo cáo thêm confusion matrix/macro F1 thay vì chỉ accuracy.
