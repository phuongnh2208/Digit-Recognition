# 🔢 Handwritten Digit Recognition (Nhận diện chữ số viết tay)

Ứng dụng web nhận diện chữ số viết tay từ 0 đến 9 được xây dựng bằng **Streamlit** và mô hình **Softmax Regression (Multiclass Logistic Regression)** huấn luyện trên bộ dữ liệu chuẩn **MNIST**.

---

## 🌟 Tính năng nổi bật

- **3 phương thức cung cấp chữ số**:
  - 📁 **Tải ảnh lên**: Hỗ trợ ảnh định dạng PNG, JPG, JPEG với cơ chế tự động phát hiện màu nền.
  - ✍️ **Vẽ trực tiếp (Canvas)**: Bảng vẽ tương tác hỗ trợ vẽ các số nhiều nét (như số 4, 5, 7...), tùy chỉnh độ dày cọ và nút xóa nét vẽ.
  - 🖼️ **Chọn ảnh mẫu**: Thư viện 10 ảnh mẫu chuẩn MNIST ($0 \to 9$) để thử nghiệm tức thì.
- **Tiền xử lý chuẩn MNIST**:
  - Cắt viền tự động (Bounding Box Crop).
  - Căn chỉnh vị trí theo trọng tâm (*Center of Mass*).
  - Chuẩn hóa kích thước 28×28 Grayscale và giá trị pixel về `[0, 1]`.
- **Hiển thị kết quả & Đánh giá**:
  - Chữ số dự đoán nổi bật và độ tin cậy (%).
  - Bảng xếp hạng Top 3 chữ số có xác suất cao nhất.
  - Biểu đồ phân phối xác suất 10 lớp tương tác (Altair Chart).
  - Phân tích cảnh báo các cặp chữ số dễ nhầm lẫn theo Ma trận nhầm lẫn (Confusion Matrix).
- **Trang báo cáo & hiệu năng chi tiết**:
  - Báo cáo phân loại (Precision, Recall, F1-Score, Support) từng chữ số.
  - Biểu đồ hội tụ Loss qua 150 Epochs.
  - Ma trận nhầm lẫn (Confusion Matrix 10×10).
  - Trình bày cơ sở lý thuyết toán học của Softmax Regression.

---

## 🏗️ Cấu trúc thư mục

```text
├── assets/
│   └── sample_images/          # Ảnh mẫu chuẩn MNIST 0-9
├── model/
│   ├── config.py               # Cấu hình thông số mô hình
│   ├── predict.py              # Hàm dự đoán & Softmax
│   ├── evaluation_metrics.json # Dữ liệu đánh giá từ notebook
│   └── weights_softmax_regression.npy # Trọng số mô hình đã huấn luyện
├── utils/
│   ├── image_processing.py     # Tiền xử lý ảnh chuẩn MNIST
│   └── styles.py               # CSS & Giao diện hiện đại
├── views/
│   ├── home.py                 # Trang chủ & Giới thiệu
│   ├── recognize.py            # Trang nhận diện chữ số
│   ├── metrics.py              # Trang đánh giá & hiệu năng
│   └── guide.py                # Trang hướng dẫn sử dụng
├── app.py                      # Điểm khởi chạy ứng dụng Streamlit
├── chu_so_viet_tay.ipynb       # Notebook huấn luyện & đánh giá
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu dự án
```

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Yêu cầu môi trường
- Python `>= 3.9`
- Git

### 2. Cài đặt các thư viện
Mở Terminal / Command Prompt tại thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng
```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại: `http://localhost:8501`.

---

## 📊 Hiệu năng mô hình thực nghiệm

| Chỉ số | Giá trị |
| :--- | :--- |
| **Thuật toán** | Softmax Regression (Linear Multiclass Classifier) |
| **Tập dữ liệu** | MNIST Handwritten Digits |
| **Độ chính xác (Test Accuracy)** | **88.77%** trên 10.000 mẫu kiểm thử độc lập |
| **Macro F1-Score** | **0.8860** |
| **Weighted F1-Score** | **0.8863** |
| **Thời gian suy luận** | `< 5 ms` / ảnh |

