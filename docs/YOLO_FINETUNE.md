# Upload dataset và fine-tune Colab

## Kiểm tra local

```bash
pip install Pillow PyYAML huggingface_hub python-dotenv
python scripts/validate_yolo_dataset.py finetune_data
```

Báo cáo: reports/dataset_audit.json. Kiểm tra định dạng class/xywh, ảnh lỗi,
file thiếu, box vượt biên, annotation trùng và ảnh trùng byte qua split.
Không chứng minh nhãn đầy đủ/đúng nội dung, không phát hiện ảnh gần trùng.

Mapping v20: 0=2-wheel vehicles, 1=bus, 2=car, 3=truck. Khi inference chọn [0,1,2,3].
YAML đã sửa thành train/images, valid/images, test/images. Colab thêm `path` tuyệt đối.
Ảnh export đều 640x640; imgsz=1024 chỉ phóng lại ảnh, không phục hồi chi tiết.

## Upload Hugging Face

Dùng HF_TOKEN trong .env hoặc môi trường; token cần quyền tạo/ghi repo đích.
Script mặc định chỉ validate và đóng gói; không upload:

```bash
python scripts/upload_finetune_dataset.py --repo-id ericng17/hcm-traffic-yolo-v20
```

Upload khi sẵn sàng:

```bash
python scripts/upload_finetune_dataset.py --repo-id ericng17/hcm-traffic-yolo-v20 --upload
```

Script tạo repo private nếu chưa tồn tại, giữ license/ghi nguồn Roboflow, upload
README.md + dataset.zip + audit JSON. Repo đã tồn tại giữ nguyên visibility.
Thư mục chuẩn bị mặc định /tmp/tfp-yolo-upload. Không upload .env hoặc token.
Không dùng load_dataset để tải ZIP này; Colab dùng hf_hub_download rồi giải nén.

## Colab

Import notebooks/colab_yolo11_finetune.ipynb. Bật GPU, thêm HF_TOKEN vào Secrets,
cho phép notebook truy cập, chạy lần lượt các cell. Token Colab chỉ cần read dataset.
Mount Drive theo hướng dẫn, sửa REPO_ID nếu đổi tên repo upload.

Mặc định pretrained yolo11l.pt, imgsz=1024, batch=4, epochs=100, patience=20.
Ảnh đặt ở /content, checkpoints ở Drive. Xem annotation gốc trước train.
Nếu run bị ngắt, cùng RUN_NAME, RESUME=True; notebook pin revision dataset.
Không dùng resume để thay đổi dữ liệu/cấu hình thực nghiệm. Run mới chọn tên mới.

Notebook đánh giá validation; test chỉ chạy khi bật RUN_FINAL_TEST=True sau khi chốt
phương pháp. Cần đánh giá thêm trên camera riêng, đặc biệt ban đêm và sai số đếm.

Nguồn: https://docs.ultralytics.com/datasets/detect/,
https://docs.ultralytics.com/modes/train/,
https://universe.roboflow.com/giaothongdetectai/hcm-traffic-detection-v2/dataset/20

## Kaggle chạy qua đêm

Import `notebooks/kaggle_yolo11_finetune.ipynb`, bật GPU + Internet, thêm HF_TOKEN
vào Secrets và cho phép notebook truy cập. Sửa REPO_ID nếu cần. Chọn Save Version →
Save & Run All; chờ version ở trạng thái Running trước khi đóng trình duyệt.
Đây là phiên mới chạy toàn notebook từ đầu, không tiếp tục phiên interactive.

Notebook dùng YOLO11l, batch=4, imgsz=1024, tối đa 100 epoch/8 giờ training
(có thể dừng sớm do patience hoặc giới hạn thời gian). Quota tài khoản vẫn áp dụng;
8 giờ không phải bảo đảm Kaggle cấp đủ tài nguyên. Chỉ dùng một GPU.
Ảnh và cache ở /tmp, output ở /kaggle/working/yolo-training. Cuối notebook tạo
yolo-training-results.zip; best.pt và last.pt có trong Output.

Để resume từ version cũ: Add Input output cũ, đặt RESUME=True, điền RESUME_FROM là
thư mục yolo-training chứa RUN_NAME và RUN_NAME-source.json. Dataset revision được
giữ nguyên. Checkpoint ghi cuối epoch; ngắt trước epoch đầu chưa có last.pt.
Không tự upload weights lên Hugging Face. Nếu runtime bị kill, kiểm tra checkpoint
trong Output; cell tạo ZIP có thể chưa kịp chạy.
