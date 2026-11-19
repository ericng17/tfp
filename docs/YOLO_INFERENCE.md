# Đếm xe bằng YOLO trong ROI đã duyệt

Dùng `scripts/infer_yolo_roi.py`, notebook Colab
`notebooks/colab_yolo_roi_inference.ipynb` hoặc notebook Kaggle
`notebooks/kaggle_yolo_roi_inference.ipynb` cho inference và đếm xe trong ROI.
Script nhận ảnh local hoặc dữ liệu Hugging Face, bắt buộc truyền `--roi`.
Các script và notebook inference toàn ảnh trước đây đã được thay thế.
Notebook fine-tune và công cụ duyệt ROI vẫn được giữ.

## Chạy bằng notebook Colab

Upload `notebooks/colab_yolo_roi_inference.ipynb` vào Google Colab và chọn runtime
GPU do Colab cung cấp. Notebook tự nhận diện môi trường; cũng hỗ trợ kernel
Jupyter trên máy GPU thuê như hướng dẫn bên dưới, không cần `google.colab`.
Notebook đã nhúng engine từ `scripts/infer_yolo_roi.py` cùng bộ ROI 28 camera
đã duyệt; không cần clone repo hay upload file `.py`/ROI riêng.

Chạy các cell theo thứ tự. Mặc định dùng nguồn `huggingface` với 5 repo snapshots
đến snapshots5 theo đúng thứ tự, `LIMIT = 0` để xử lý toàn bộ. Có thể chọn `local`.
Weights được đọc trực tiếp từ `/content/drive/MyDrive/tfp/models/yolo/best.pt`,
tương ứng `My Drive/tfp/models/yolo/best.pt`. Drive được mount trước khi đọc file;
notebook báo rõ nếu đường dẫn sai. Không có bước upload weights. Repo private cần
Secret `HF_TOKEN`. Xem mapping class của weights trước khi chạy inference.

Đặt `LIMIT = 1000` nếu muốn chạy thử: giới hạn áp dụng chung cho cả lần gọi và có
thể chỉ chạy dataset đầu, không phải 1.000 ảnh mỗi dataset. Sau khi kiểm tra preview,
đặt lại `LIMIT = 0`, chạy lại cell cấu hình và cell inference để tiếp tục toàn bộ.
Notebook xuất tổng/từng class, preview, checkpoint SQLite, thống kê theo repo và
có cell tải CSV/ZIP kết quả. Trên Colab, `USE_DRIVE = True` mặc định để mount Drive.
`BACKUP_ENABLED = True` khôi phục và sao lưu checkpoint định kỳ mỗi giờ
(`BACKUP_INTERVAL_HOURS = 1`), cùng một lần khi kết thúc
hoặc Stop. Worker nền dùng SQLite Backup API qua kết nối riêng để tạo snapshot
nhất quán từ các batch đã commit, xuất CSV của snapshot và thay file trên Drive
qua file tạm. Không copy trực tiếp database đang ghi. Mỗi chu kỳ một giờ được
tính sau khi lần sao lưu trước hoàn tất; lỗi sao lưu được ghi log và thử lại ở
chu kỳ sau. Khi dừng, chờ worker rồi sao lưu lần cuối để tránh hai bản sao ghi đè
đồng thời. Nếu runtime bị kill cứng, chỉ phục hồi tới lần sao lưu thành công gần
nhất. Thay cấu hình ảnh hưởng dự đoán cần `RUN_NAME` mới. CSV không thay thế
checkpoint để resume.

Mã và ROI nhúng là bản chụp tại lúc tạo notebook. Khi cập nhật script hoặc ROI trong
repo, cần cập nhật tương ứng các cell nhúng trước khi dùng notebook cho run mới.

## CKEY / Jupyter qua SSH, notebook lưu trên laptop

Kết nối Existing Jupyter Server qua SSH tunnel chỉ chuyển lệnh và kết quả giữa VS
Code và kernel. Mọi đường dẫn Python đều thuộc máy GPU: kernel không tự đọc được
file trên laptop. Máy GPU cần CPU/RAM/VRAM và ổ tạm để nạp model, decode ảnh và
inference; không thể chỉ trỏ `device=0` từ kernel laptop để dùng GPU qua SSH.

Notebook tự chọn `USE_DRIVE = False` ngoài Colab. Nếu server có `/workspace`, các
đường dẫn mặc định là:

- Weights: `/workspace/tfp/best.pt`.
- Output đang chạy: `/workspace/tfp/outputs/yolo-roi-v1`.
- Snapshot mỗi giờ: `/workspace/tfp/checkpoints/yolo-roi-v1`.

Sửa `RUNTIME_ROOT_PATH`, `WEIGHTS_PATH`, `BACKUP_ROOT_PATH` theo nơi lưu trên
server. Notebook không cần Google Drive API. `HF_TOKEN` lấy từ biến môi trường
của kernel hoặc nhập ẩn bằng `getpass`; ảnh Hugging Face tải tạm từng batch/shard.
Notebook lưu trên laptop, weights gốc và bản sao checkpoint vẫn có thể giữ local.

Trên laptop, chuyển weights một lần (thay `SSH_PORT` và `SERVER_IP` bằng thông tin
CKEY, sửa đường dẫn weights local nếu cần):

```bash
ssh -p 1602 root@n1.ckey.vn 'mkdir -p /workspace/tfp'
rsync -av -e 'ssh -p 1602' \
  src/models/yolo/yolo11l-v20-run1/weights/best.pt \
  root@n1.ckey.vn :/workspace/tfp/best.pt
```

Giữ terminal thứ hai trên laptop để kéo snapshot về local mỗi 5 phút. Snapshot
được tạo trên server mỗi giờ; lần copy đầu có thể báo chưa có thư mục cho tới
khi snapshot đầu tiên hoàn tất. Script không đồng bộ database đang ghi trực tiếp:

```bash
bash scripts/sync_yolo_checkpoint.sh root@SERVER_IP SSH_PORT \
  "$PWD/checkpoints/yolo-roi-v1" /workspace/tfp/checkpoints/yolo-roi-v1 300
```

Yêu cầu `ssh`, `rsync` trên laptop và `rsync` trên server. Script dùng kết nối SSH
bình thường, không chứa mật khẩu/token. Dùng khoảng `0` để copy một lần trước khi
tắt server. Không chạy inference trên bản local trong khi rsync đang cập nhật nó.
Nếu mất máy thuê, chuyển toàn bộ checkpoint local sang thư mục snapshot của máy
mới rồi chạy notebook cùng cấu hình để khôi phục. Giữ versions PyTorch/Ultralytics
tương thích với run đã lưu. Ổ `/workspace` có được giữ khi tắt/xóa máy hay không
phụ thuộc dịch vụ/gói thuê; bản đã copy về laptop mới độc lập với vòng đời server.

## Chạy bằng notebook Kaggle

Import `notebooks/kaggle_yolo_roi_inference.ipynb` vào Kaggle, bật GPU và Internet.
Notebook đã nhúng cùng engine và ROI 28 camera như bản Colab. Thêm Dataset chứa
`best.pt` hoặc Output của notebook training qua Add Input, rồi sửa `WEIGHTS_PATH`
theo đường dẫn thật trong `/kaggle/input`. Repo private cần Secret `HF_TOKEN`
và quyền truy cập của notebook trong Add-ons → Secrets.

Chạy các cell từ trên xuống hoặc Save Version → Save & Run All để chạy nền từ đầu.
Mặc định xử lý toàn bộ 5 repo theo thứ tự, dùng GPU 0. Đặt `LIMIT = 1000` để chạy
thử. CSV, preview và checkpoint nằm tại `/kaggle/working/<RUN_NAME>`; cache và mã/ROI
tạm ở `/tmp`. Cell cuối cung cấp link CSV và ZIP kết quả trong Kaggle Output.

Muốn resume ở phiên mới, lưu toàn bộ Output của version trước, thêm lại qua Add
Input và đặt `RESUME_FROM` là thư mục chứa `counts.sqlite`, `run.json`, `roi.json`.
Notebook copy checkpoint sang thư mục có quyền ghi trước khi inference; ảnh đã
commit được bỏ qua. Giữ cùng weights, cấu hình và phiên bản PyTorch/Ultralytics;
engine từ chối checkpoint khác cấu hình. Nếu runtime mất, chỉ output còn giữ được
hoặc version đã lưu mới dùng để resume; riêng CSV không đủ.

## Quy tắc đếm

- Dùng `configs/camera_rois.reviewed.json`: trạng thái chung và từng camera phải
  là `approved`; đa giác phải hợp lệ, tọa độ chuẩn hóa trong [0,1].
- YOLO phát hiện trên ảnh gốc; chỉ đếm các class được chỉ định bằng `--classes`.
- Điểm xét của xe là giữa cạnh đáy bounding box: `((x1+x2)/2, y2)`.
  Điểm nằm trong hoặc trên biên bất kỳ ROI nào của camera thì xe được tính.
- Các vùng được hợp lại: một detection nằm trong nhiều vùng vẫn chỉ tính một lần.
- `total_vehicles` bằng tổng các cột đếm từng class sau khi lọc ROI.
- Camera thiếu ROI làm dừng run. Không có chế độ đếm toàn ảnh dự phòng.
- Chấp nhận ảnh được resize cùng tỷ lệ. Sai lệch tỷ lệ khung hình quá 1% so với
  ảnh tham chiếu làm dừng run; không tự xoay ảnh theo EXIF. Cần kiểm tra bằng mắt
  nếu camera đổi góc, bị crop hoặc ảnh ngày/đêm có vùng nhìn khác nhau.

Đây là số xe quan sát tại từng ảnh, chưa phải lưu lượng xe đi qua hay nhãn kẹt xe.
Không cộng số đếm của nhiều snapshot thành số xe duy nhất.

## Chuẩn bị

Python >=3.11. Cài PyTorch phù hợp CUDA của môi trường, sau đó:

```bash
python -m pip install -r requirements-inference.txt
python scripts/infer_yolo_roi.py --help
```

Weights phải là file local của mô hình detection. Kiểm tra mapping trước khi chọn
class, không giả định ID COCO:

```python
from ultralytics import YOLO
model = YOLO('src/models/yolo/yolo11l-v20-run1/weights/best.pt')
print(model.names)
```

Mapping v20 dự kiến: 0 = xe hai bánh, 1 = bus, 2 = car, 3 = truck.
Chỉ dùng `--classes 0 1 2 3` khi weights thực tế xác nhận mapping này.

## Ảnh local

Đầu vào là thư mục ảnh đã giải nén:

```text
snapshots/
  <camera_id>/20261114_120000.jpg
  <camera_id>/20261114_120013.jpg
```

Tên thư mục cha là camera ID; tên file là `YYYYmmdd_HHMMSS.jpg` (hỗ trợ cả PNG/JPEG).
Tên không có timestamp hợp lệ làm dừng run. Mặc định dùng giờ Việt Nam;
truyền `--timezone UTC` nếu tên ảnh crawler theo UTC.

Chạy thử 100 ảnh từ thư mục gốc dự án:

```bash
python scripts/infer_yolo_roi.py \
  --source data/raw/snapshots \
  --weights src/models/yolo/yolo11l-v20-run1/weights/best.pt \
  --roi configs/camera_rois.reviewed.json \
  --classes 0 1 2 3 \
  --output outputs/yolo-roi \
  --device 0 --batch 4 --limit 100 --preview-count 28
```

Kiểm tra ảnh trong `outputs/yolo-roi/previews/`: đường xanh là ROI, khung và chấm
vàng là các xe được tính; nhãn có tổng và số xe từng class. Preview lấy các ảnh đầu
được xử lý, không bảo đảm đại diện đủ camera/ngày/đêm. Hãy chọn riêng các nhóm ảnh
đại diện để kiểm chứng ROI và độ chính xác đếm.

Chạy lại cùng lệnh, bỏ `--limit 100` để xử lý tiếp toàn bộ dữ liệu.
`--device auto` (mặc định) yêu cầu CUDA. Khi muốn chạy CPU, đặt rõ
`--device cpu --batch 1`. `--half` chỉ dành cho CUDA và cần kiểm chứng chất lượng
trong output riêng. Có thể giảm batch rồi resume nếu hết VRAM.

## Dữ liệu Hugging Face trên Kaggle / Colab

Dùng notebook Colab ở trên, hoặc đưa file `.py`, ROI đã duyệt và weights lên
runtime để chạy lệnh bên dưới. Bật GPU và Internet, cài dependencies như trên.
Trên Kaggle, lấy token đọc dữ liệu từ Secrets:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
```

Trên Colab, dùng `google.colab.userdata.get('HF_TOKEN')` để gán biến môi trường
tương tự. Không ghi token trực tiếp trong script hoặc lệnh shell.

Chạy lệnh dưới trong terminal của runtime, sửa đường dẫn weights/script/ROI theo
vị trí đã upload (nếu dùng cell notebook, thêm `!` trước `python`):

```bash
python scripts/infer_yolo_roi.py \
  --hf-repos \
    ericng17/hcm_traffic_snapshots \
    ericng17/hcm_traffic_snapshots2 \
    ericng17/hcm_traffic_snapshots3 \
    ericng17/hcm_traffic_snapshots4 \
    ericng17/hcm_traffic_snapshots5 \
  --weights /kaggle/input/YOUR-WEIGHTS/best.pt \
  --roi configs/camera_rois.reviewed.json \
  --classes 0 1 2 3 \
  --output /kaggle/working/yolo-roi \
  --device 0 --batch 4 --limit 1000 --preview-count 28
```

`--format auto` ưu tiên Parquet nếu repo có Parquet, nếu không sẽ chọn ảnh rời.
Không xử lý đồng thời cả hai biểu diễn. Có thể đặt `--format raw` hoặc `parquet`.
Ảnh rời dùng cấu trúc thư mục/tên file như ảnh local. Parquet phải chứa bytes ảnh
trong cột `image` hoặc struct `image = {bytes, path}`. Camera/thời gian lấy từ
metadata (`camera_id`/`cam_id`/`camera`, `timestamp`/`datetime`/`time`) hoặc đường dẫn
ảnh. Có thể chỉ định tên cột khác:

```text
--image-column image --path-column file_name
--camera-column camera_id --timestamp-column timestamp
```

Timestamp hỗ trợ datetime, ISO 8601 hoặc `YYYYmmdd_HHMMSS`; giá trị epoch số cần
chuyển đổi trước. Camera/thời gian không xác định làm dừng run. ZIP và Parquet chỉ
chứa đường dẫn ngoài, không có bytes ảnh, cần chuẩn bị dữ liệu trước.

Script tải từng batch ảnh rời hoặc một shard Parquet vào thư mục tạm rồi xóa;
không tải toàn bộ ảnh về ổ đĩa. Repo revision được ghim trong checkpoint. Khi chạy
tiếp, ảnh đã commit không inference lại. Shard Parquet chưa được đánh dấu hoàn tất
có thể cần tải/đọc lại để bỏ qua các hàng đã xử lý. Không tự gộp ảnh trùng giữa các
repo: giữ `image_sha256` và nguồn để kiểm tra trước khi tạo chuỗi thời gian.

## Đầu ra và resume

- `counts.csv`: nguồn ảnh, camera, timestamp, status, `total_vehicles`,
  `count_<id>_<model.names[id]>`, hash ảnh, hash ROI và lỗi.
- `counts.sqlite`: checkpoint theo batch. Ảnh hợp lệ không có xe trong ROI có số
  đếm 0; ảnh hỏng có status `error` và để trống số đếm. Lỗi model/OOM làm rollback
  batch và dừng, không được ghi thành ảnh không có xe.
- `run.json`: hash weights/ROI, mapping class, cấu hình và phiên bản thư viện.
- `roi.json`: bản sao cấu hình ROI dùng trong run.
- `previews/`: ảnh gốc với các vùng ROI và chỉ các detection được tính.

Thay weights, ROI, class, threshold, kích thước inference, precision hoặc nguồn
dữ liệu cần output mới. Batch, device, limit, workers và số preview có thể đổi
khi resume. `--limit` là số ảnh mới/thử lại tối đa trong lần gọi, không phải số
ảnh tổng của checkpoint. `--retry-errors` xử lý lại các ảnh đã ghi lỗi.
Ảnh local đã xử lý mà thay size/mtime sẽ bị từ chối; giữ nguồn bất biến trong run.

Giữ toàn bộ thư mục output để resume sau khi runtime mất; riêng CSV không đủ.
Script không tự upload hay sao lưu output. Trên Kaggle, tải/lưu Output rồi copy
về thư mục có quyền ghi ở phiên sau; trên Colab sao lưu sau khi dừng process hoặc
dùng SQLite backup API. Mỗi process phải có output riêng.

Với nhiều GPU, dùng cùng cấu hình nguồn và `--num-shards 2`, lần lượt
`--shard-index 0 --device 0` và `--shard-index 1 --device 1`, mỗi process output
riêng. ID nguồn được hash để phân shard; ghép CSV sau khi kiểm tra cấu hình và ID
không trùng. Giữ số lượng và chỉ số shard khi resume.

## Kiểm tra logic

```bash
python -m unittest discover -s tests -v
```

Các test mô phỏng đầu ra YOLO để kiểm tra lọc ROI, biên, vùng chồng nhau, tổng/từng
class, ảnh lỗi, rollback, resume, shard và các nguồn ảnh local/HF/Parquet. Chúng
không đo độ chính xác của weights trên ảnh giao thông thực tế.
