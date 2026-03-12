# Plastic Bottles Classification

Dự án phân loại hình ảnh (Computer Vision) sử dụng Machine Learning để phân biệt **Chai nhựa (Plastic Bottle)** và **Các loại rác khác (Others)**. Dự án tích hợp quy trình MLOps cơ bản (DVC, MLflow), giao diện Web (Streamlit), và đóng gói bằng Docker.

## 🎯 Tính năng

- Huấn luyện và đánh giá trên 4 loại mô hình:
  - **Logistic Regression** (GridSearchCV tuning)
  - **Support Vector Machine (SVM)** (GridSearchCV tuning)
  - **Convolutional Neural Network (CNN)** (PyTorch, Custom architecture)
  - **EfficientNet-B0** (PyTorch, Transfer Learning với ImageNet weights)
- Theo dõi quá trình huấn luyện và versioning model tự động bằng **MLflow**.
- Tự động lưu mô hình tốt nhất (Best model) dựa trên chỉ số **F1-score (weighted)**.
- Giao diện so sánh hiệu năng các mô hình trên tập Test Set.

---

## 🚀 Hướng dẫn Cài đặt & Chạy dự án (Từ đầu đến train)

### Bước 1: Chuẩn bị môi trường
Yêu cầu: Python 3.9+ 

1. Khởi tạo môi trường ảo (tùy chọn nhưng khuyến khích):
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn bị Dữ liệu (Data)
Dự án yêu cầu dữ liệu ảnh thô (raw data) chia thành 2 thư mục theo tên class. Hãy copy ảnh của bạn vào đúng định dạng sau:
```text
data/raw/
  ├── others/           # Chứa các ảnh không phải chai nhựa (VD: others0.jpg, others1.png)
  └── plastic bottle/   # Chứa các ảnh chai nhựa (VD: plasticBottle1.jpg)
```

### Bước 3: Chia tập dữ liệu (Split Data)
Chạy script chia data thành 3 phần (Train 70% / Val 15% / Test 15%):
```bash
python src/split_data.py
```
> **Output:** File `data/splits/split.csv` sẽ được tạo ra, lưu thông tin của toàn bộ ảnh và nhãn của chúng.

### Bước 4: Huấn luyện Mô hình (Training)
Chạy 1 lệnh duy nhất dưới đây để tự động train lần lượt 4 mô hình (LogReg -> SVM -> CNN -> EfficientNet) và sau đó chạy đánh giá tổng quát:
```bash
python src/train_all.py
```

**Quá trình này sẽ thực hiện:**
- Load data theo kích thước `128x128`.
- Chạy GridSearchCV (cho LogReg, SVM) để tìm tham số tốt nhất.
- Chạy Train Loop qua số Epochs (cho CNN).
- **Lưu lại Model có thông số F1 trên tập Validation cao nhất**.

---

## 💾 Thư mục lưu trữ (Outputs)

Sau khi quá trình Training ở Bước 4 kết thúc, bạn có thể tìm thấy các file Output tại đây:

### 1. Saved Models (Mô hình đã lưu)
Mô hình sẽ lưu tại thư mục `models/` tại root của dự án.
- **LogReg:** `models/best_logistic_regression.pkl` (Dùng joblib để load)
- **SVM:** `models/best_svm.pkl` (Dùng joblib để load)
- **CNN:** `models/best_cnn.pth` (Dùng `torch.load` và nạp vào state_dict)
- **EfficientNet:** `models/best_efficientnet.pth` (Dùng `torch.load` và nạp vào state_dict)

### 2. Logs từ MLflow
Mọi thông số kỹ thuật (param), chỉ số biểu đồ (metrics: Loss, Accuracy, F1) và Model Artifacts đều được lưu tại thư mục `mlruns/`.

Để xem giao diện quản lý trên trình duyệt:
```bash
mlflow ui
```
Sau đó mở trình duyệt tại: [http://localhost:5000](http://localhost:5000)

---

## 🌐 Chạy Web App qua Docker

Dự án đã được đóng gói sẵn bằng Docker để bạn có thể triển khai ở bất kỳ đâu mà không lo về môi trường (sau khi bạn đã có models/ đã được train ở Bước 4).

### 1. Build Docker Image
Mở terminal ở thư mục gốc dự án và chạy:
```bash
docker build -t plastic-bottle-classifier .
```

### 2. Chạy Docker Container
```bash
docker run -p 8501:8501 plastic-bottle-classifier
```

Sau đó Web App sẽ hiển thị tại: [http://localhost:8501](http://localhost:8501)

---

## 🚧 Roadmap tiếp theo
- [x] Giao diện Web App (Streamlit) cho phép User tải ảnh lên và chọn Model dự đoán.
- [x] Đóng gói toàn bộ dự án bằng Docker.
- [ ] Pipeline CI/CD bằng DVC và Github Actions.