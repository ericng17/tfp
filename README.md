<<<<<<< HEAD
# Traffic Flow Prediction (TFP) - Real-time Traffic Forecasting using YOLO & STGNN

> **Đề tài Khóa luận tốt nghiệp**: Xây dựng hệ thống dự đoán lưu lượng giao thông theo thời gian thực sử dụng dữ liệu lịch sử và mạng tích chập đồ thị không-thời gian  
> **Sinh viên thực hiện**: Nguyễn Hồng Quân (23521262)  
> **Cán bộ hướng dẫn**: TS. Đỗ Trọng Hợp  
> **Đơn vị**: Khoa Công nghệ Phần mềm, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM (UIT)

---

## 📌 Tổng quan kiến trúc hệ thống (System Pipeline)

```text
Traffic Cameras (HCMC Portal)
            │ (12s-15s / frame)
            ▼
┌─────────────────────────┐
│  Data Collection & Preprocessing
│  (scripts/slot_fetch.py)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Vehicle Detection & Classification
│  (YOLO11l Finetuned: mAP50 ~80.7%)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Graph Construction & Time-Series Aggregation
│  (28 nodes Tan Binh cluster, Adjacency Matrix A)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Spatio-Temporal Forecasting
│  (STGCN Baseline, Graph WaveNet, DCRNN)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Evaluation & Real-time Demo
│  (MAE, RMSE, MAPE / Streamlit Dashboard)
└─────────────────────────┘
```

---

## 📁 Cấu trúc thư mục (Repository Structure)

```text
tfp/
├── docs/                 # Đề cương chi tiết Khóa luận tốt nghiệp
├── scripts/
│   ├── slot_fetch.py     # Script thu thập ảnh từ 28 camera giao thông
│   ├── crawler.ipynb     # Notebook thu thập dữ liệu
│   └── finetune-yolo.ipynb # Notebook huấn luyện YOLO11l
├── result_finetune_yolo/ # Biểu đồ và chỉ số đánh giá của mô hình YOLO11l
├── link_camera.py        # Danh sách 28 URL camera khu vực Tân Bình
├── .gitignore            # Loại trừ weights nặng (.pt), logs và ảnh thô
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & Chạy (Getting Started)

### 1. Khởi tạo môi trường ảo
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu camera
* **Chạy 1 lần (cho Cron / Task Scheduler)**:
  ```bash
  python scripts/slot_fetch.py
  ```
* **Chạy lặp tự động liên tục (mỗi 5 phút)**:
  ```bash
  python scripts/slot_fetch.py --loop 300
  ```

=======
# tfp
>>>>>>> e8173f8be85423d657d017d5a8b470560fcbd3e2
