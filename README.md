# Hệ thống NLP gợi ý bệnh và chuyên khoa

Ứng dụng Streamlit nhận mô tả triệu chứng bằng tiếng Việt, chuyển văn bản thành các đặc trưng y khoa, dự đoán bệnh và tổng hợp xác suất theo 5 chuyên khoa:

- Nội tổng quát
- Ngoại
- Nhi
- Răng Hàm Mặt
- Tai Mũi Họng

Hệ thống dùng **100 bệnh**, **450 đặc trưng** và model v2 đã train trong thư mục `model_new`. Kết quả chỉ hỗ trợ định tuyến ban đầu, không thay thế chẩn đoán, cấp cứu hoặc điều trị của nhân viên y tế.

## Chức năng chính

- Nhập mô tả tự nhiên bằng tiếng Việt có dấu hoặc không dấu.
- Nhận diện alias, lỗi gõ gần đúng và phủ định như “không sốt”.
- Nhận diện tuổi, giới tính, thời gian, kiểu khởi phát, mức độ và yếu tố nguy cơ.
- Đọc nhiệt độ dạng số như `38.2 độ C`, `39,5°C` và `sốt 40 độ`.
- Dự đoán 5 bệnh có xác suất cao nhất.
- Tính độ tin cậy chuyên khoa riêng bằng tổng xác suất các bệnh thuộc khoa đó.
- Không xếp hạng khoa Nhi nếu chưa có bằng chứng người bệnh là trẻ em.
- Cảnh báo các dấu hiệu cần đánh giá khẩn cấp trước khi hiển thị dự đoán.
- Có chế độ chọn trực tiếp triệu chứng đã chuẩn hóa.
- Có bộ kiểm thử hồi quy NLP độc lập.

## Cấu trúc dự án

```text
TestAI/
├── test_ai_app.py                      # Giao diện Streamlit và logic dự đoán
├── nlp_engine.py                       # Chuẩn hóa, alias, fuzzy, phủ định, ngữ cảnh
├── check_nlp_algorithm.py              # Bộ kiểm thử NLP
├── nlp_report.json                     # Báo cáo kiểm thử gần nhất
├── requirements.txt                    # Thư viện chạy ứng dụng
├── training_data_5_specialties_v2.csv # Dữ liệu train tổng hợp
├── disease_specialty_map_v2.csv       # Ánh xạ bệnh sang chuyên khoa
├── feature_dictionary_v2.csv          # Từ điển 450 đặc trưng
├── generate_dataset_v2.py             # Tái tạo dataset
├── DATASET_V2_README.md                # Mô tả riêng của dataset
└── model_new/
    ├── ai_disease_model_v2.joblib
    ├── disease_specialty_map_v2.csv
    ├── feature_names_v2.json
    ├── model_metrics_v2.json
    ├── feature_importance_v2.csv
    ├── validation_report_v2.csv
    └── top_confusions_v2.csv
```

## Cài đặt

Yêu cầu Python 3.10 trở lên. Trên PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn script kích hoạt môi trường, có thể gọi Python trong môi trường trực tiếp mà không cần kích hoạt:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe -m streamlit run test_ai_app.py
```

Mở địa chỉ Streamlit hiển thị trong terminal, thường là `http://localhost:8501`.

Khi thay đổi `nlp_engine.py` hoặc `test_ai_app.py`, Streamlit thường tự tải lại. Nếu giao diện chưa cập nhật, dừng tiến trình bằng `Ctrl+C` rồi chạy lại lệnh trên.

## Cách sử dụng

### Nhập mô tả tự nhiên

Nên cung cấp tuổi, vị trí, thời gian, mức độ và dấu hiệu đi kèm. Ví dụ:

```text
Bé trai 6 tuổi sốt 39,5 độ trong 2 ngày, ho, thở nhanh và khò khè.
```

```text
Người lớn đau bụng dưới bên phải đột ngột, buồn nôn và ấn bụng đau.
```

```text
Sâu răng, đau khi uống lạnh, không sưng mặt.
```

Sau khi bấm **Phân tích NLP và dự đoán**, ứng dụng hiển thị:

1. Cảnh báo khẩn nếu phát hiện dấu hiệu nguy hiểm.
2. Bệnh có xác suất cao nhất.
3. Độ tin cậy của bệnh.
4. Khoa nên khám.
5. Độ tin cậy riêng của chuyên khoa.
6. Các đặc trưng NLP đã nhận diện.
7. Xếp hạng 5 chuyên khoa và 5 bệnh có khả năng cao nhất.

### Chọn triệu chứng

Tab **Chọn triệu chứng** cho phép chọn trực tiếp các đặc trưng đã chuẩn hóa. Chế độ này hiện yêu cầu tối thiểu hai triệu chứng.

## Luồng xử lý

```text
Văn bản tiếng Việt
       ↓
Chuẩn hóa chữ thường, dấu câu và dấu tiếng Việt
       ↓
Khớp alias → xử lý phủ định → fuzzy lỗi gõ → ngữ cảnh có cấu trúc
       ↓
Vector nhị phân 450 đặc trưng
       ↓
Model dự đoán xác suất 100 bệnh
       ↓
Tổng xác suất bệnh theo 5 chuyên khoa
       ↓
Kiểm tra ngưỡng định tuyến và hiển thị kết quả
```

### Độ tin cậy bệnh

Đây là xác suất của từng lớp bệnh do `predict_proba` trả về. Ứng dụng hiển thị bệnh có xác suất cao nhất và danh sách 5 bệnh đứng đầu.

### Độ tin cậy chuyên khoa

Độ tin cậy của một chuyên khoa bằng tổng xác suất của tất cả bệnh thuộc chuyên khoa đó:

```text
P(chuyên khoa | triệu chứng) = Σ P(bệnh thuộc chuyên khoa | triệu chứng)
```

Đây là điểm định tuyến riêng, không phải độ tin cậy của bệnh đứng đầu.

### Ngưỡng định tuyến

Kết quả được xem là chưa đủ chắc chắn khi:

- Xác suất chuyên khoa cao nhất dưới `50%`; hoặc
- Khoảng cách giữa hai chuyên khoa đứng đầu dưới `10%`.

Khi đó ứng dụng chuyển về **Nội tổng quát** để tiếp nhận ban đầu nhưng vẫn giữ bảng xếp hạng để tham khảo.

Một triệu chứng đơn lẻ chỉ được dùng để định tuyến khi đặc trưng đó:

- Có ít nhất `20` mẫu trong dữ liệu train; và
- Ít nhất `80%` số mẫu có đặc trưng đó thuộc cùng một chuyên khoa.

Ví dụ, “sâu răng” có thể đủ định hướng Răng Hàm Mặt; “đau đầu” hoặc “sốt” vẫn cần thêm ngữ cảnh vì xuất hiện ở nhiều bệnh và nhiều khoa.

### Quy tắc khoa Nhi

Nếu văn bản và các ô lựa chọn không chứa bằng chứng về trẻ em, xác suất các bệnh Nhi được loại khỏi xếp hạng rồi phần còn lại được chuẩn hóa lại. Bằng chứng hợp lệ gồm nhóm tuổi sơ sinh, nhũ nhi, trẻ em, vị thành niên hoặc các cụm như “bé”, “trẻ em”, “6 tuổi”.

## NLP tiếng Việt

`nlp_engine.py` xử lý các lớp sau:

1. **Chuẩn hóa:** chữ thường, bỏ dấu tiếng Việt và dấu câu để hỗ trợ câu có/không dấu.
2. **Khớp trực tiếp:** tìm alias theo cụm dài trước.
3. **Phủ định:** chuyển “không sốt” thành đặc trưng `neg_no_fever`, không đồng thời bật `fever`.
4. **Fuzzy bảo thủ:** sửa lỗi gõ nhỏ nhưng yêu cầu số từ và ký tự đầu tương thích để hạn chế ánh xạ sai.
5. **Nhiệt độ:** chỉ đọc số có ngữ cảnh `sốt`, `nhiệt độ`, `°C` hoặc `độ C`.
6. **Ngữ cảnh:** tuổi, giới, thời gian, khởi phát, mức độ và yếu tố nguy cơ.
7. **Cảnh báo:** kiểm tra luật nguy hiểm độc lập với xác suất của model.

Ví dụ nhiệt độ:

| Đầu vào | Đặc trưng |
|---|---|
| `sốt 38.2 độ C` | `fever` |
| `nhiệt độ 39,5°C` | `fever`, `high_fever` |
| `sốt 40 độ` | `fever`, `high_fever` và cảnh báo khẩn |
| `nhiệt độ 36,8 độ C` | Không nhận là sốt |
| `bệnh nhân 39 tuổi` | Không nhầm 39 thành nhiệt độ |

## Cảnh báo khẩn

Lớp cảnh báo chạy trước phần dự đoán bệnh. Một số nhóm hiện được hỗ trợ:

- Khó thở nghiêm trọng hoặc tím tái.
- Rối loạn ý thức.
- Dấu hiệu đột quỵ.
- Co giật.
- Chảy máu nhiều hoặc xuất huyết tiêu hóa.
- Sốt kèm cứng cổ.
- Đau bụng cấp.
- Mất thính lực đột ngột.
- Sốt rất cao từ `40°C`.
- Trẻ dưới 3 tháng sốt từ `38°C`.

Sốt rất cao đơn độc được hướng dẫn đánh giá y tế khẩn. Nếu kèm khó thở, co giật, lơ mơ, tím tái hoặc không thể di chuyển an toàn, giao diện khuyến cáo gọi 115.

Không được dùng tên bệnh hoặc xác suất model để bỏ qua cảnh báo luật.

## Kiểm thử NLP

Chạy toàn bộ bộ kiểm thử:

```powershell
.\.venv\Scripts\python.exe check_nlp_algorithm.py
```

Chạy chế độ nghiêm ngặt và xuất báo cáo JSON:

```powershell
.\.venv\Scripts\python.exe check_nlp_algorithm.py --strict --json nlp_report.json
```

`--strict` trả mã lỗi `1` nếu có ca thất bại, phù hợp để dùng trong CI. Muốn xem cả các ca đạt:

```powershell
.\.venv\Scripts\python.exe check_nlp_algorithm.py --show-passed
```

Bộ test kiểm tra:

- Toàn bộ alias có cột tương ứng trong model.
- Câu tự nhiên nhiều triệu chứng.
- Có dấu, không dấu và dấu câu.
- Lỗi chính tả nhỏ.
- Phủ định và biểu hiện âm tính.
- Tuổi, giới, thời gian, khởi phát và mức độ.
- Nhiệt độ số và trường hợp không được nhầm với tuổi.
- Những cụm từng dễ ánh xạ sai.
- Dấu hiệu cấp cứu.

Kết quả gần nhất: **437/437 ca đạt**. Con số này chỉ phản ánh bộ test đã định nghĩa, không chứng minh NLP hiểu được mọi cách diễn đạt trong thực tế.

## Alias chưa được model sử dụng

`check_nlp_algorithm.py` cũng báo các alias có trong kho từ vựng nhưng không có cột trong model. Hiện có 24 đặc trưng như `toothache`, `severe_headache`, `seizure`, `vomiting_blood`, `productive_cough` và `rash`.

Các dấu hiệu nguy hiểm liên quan vẫn có thể được luật cảnh báo phát hiện, nhưng chúng không tham gia xác suất dự đoán nếu không tồn tại trong 450 cột. Muốn model sử dụng chúng cần:

1. Thêm đặc trưng vào từ điển và dataset.
2. Gán đặc trưng cho các bệnh phù hợp.
3. Tạo đủ mẫu dương/âm có ý nghĩa.
4. Train lại model.
5. Chạy lại toàn bộ kiểm thử NLP và đánh giá model.

## Thay model mới

Model bundle `model_new/ai_disease_model_v2.joblib` phải có tối thiểu các khóa:

```python
{
    "model": ...,          # Có predict_proba và classes_
    "feature_names": ...,  # Danh sách đúng 450 đặc trưng
    "disease_map": ...,
}
```

Đồng thời cần giữ `model_new/disease_specialty_map_v2.csv` đồng bộ với `model.classes_`. Sau khi thay model:

```powershell
.\.venv\Scripts\python.exe check_nlp_algorithm.py --strict
.\.venv\Scripts\python.exe -m streamlit run test_ai_app.py
```

Kiểm tra thủ công ít nhất các nhóm Nội, Ngoại, Nhi, Răng Hàm Mặt, Tai Mũi Họng, phủ định, câu chỉ có một triệu chứng và cảnh báo khẩn.

## Chỉ số model hiện tại

Theo `model_new/model_metrics_v2.json`:

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 99.90% |
| Balanced accuracy | 99.90% |
| Macro F1 | 99.90% |
| Weighted F1 | 99.90% |
| OOB score | 99.85% |

Các chỉ số rất cao vì dữ liệu hiện tại là dữ liệu tổng hợp với quy tắc sinh tương đối rõ. Không được diễn giải đây là độ chính xác trên bệnh nhân thật.

## Giới hạn và an toàn

- Dataset là dữ liệu tổng hợp, không đại diện tỷ lệ bệnh thực tế.
- Model chưa được xác thực lâm sàng hoặc thẩm định đầy đủ bởi bác sĩ.
- Một từ có thể có nhiều nghĩa tùy ngữ cảnh; NLP dựa trên luật không hiểu ngôn ngữ như bác sĩ.
- Xác suất của mô hình phân loại không mặc nhiên là xác suất bệnh thật trong cộng đồng.
- Hệ thống chỉ có 5 chuyên khoa; trường hợp ngoài phạm vi cần được báo chưa hỗ trợ hoặc chuyển cơ sở phù hợp.
- Cảnh báo luật không thể bao phủ mọi tình huống nguy hiểm.
- Không sử dụng hệ thống để tự điều trị, trì hoãn cấp cứu hoặc thay thế khám trực tiếp.

Trước khi triển khai thực tế cần có dữ liệu bệnh nhân đã ẩn danh, quy trình phê duyệt đạo đức/pháp lý phù hợp, đánh giá hiệu năng ngoài mẫu, hiệu chỉnh xác suất, kiểm thử thiên lệch và bác sĩ của từng chuyên khoa rà soát toàn bộ quy tắc.

