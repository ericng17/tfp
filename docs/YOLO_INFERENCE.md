# Inference YOLO11 trên Kaggle / Colab

## Thiết kế

Ưu tiên một process / GPU, batch ảnh; chưa cần Spark. Spark hữu ích khi có cluster
nhiều worker/GPU và cấu hình tài nguyên đúng; local Spark trên một notebook không tự
làm YOLO nhanh hơn, còn có thể tạo nhiều bản model tranh VRAM. Dùng shard nếu có
hai GPU. Sau inference, bảng số đếm nhỏ hơn ảnh nhiều; pandas đủ cho bước tổng hợp
ban đầu. Tham khảo: https://docs.ultralytics.com/modes/predict/ và
https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.functions.predict_batch_udf.html

Script không upload, không gửi webhook, không tự tải weights. Python >=3.11.
Đầu vào là **ảnh đã giải nén**, không phải ZIP hoặc Parquet chứa ảnh:

```text
snapshots/
  <camera_id>/20261112_120000.jpg
  <camera_id>/20261112_120013.jpg
```

Tên thư mục cha là camera ID; tên ảnh phải theo YYYYmmdd_HHMMSS. Sai định dạng
sẽ dừng để tránh gán nhầm timestamp. Mặc định timestamp là giờ Việt Nam; nếu crawler
chạy theo UTC, dùng `--timezone UTC`. Script hiện đếm toàn ảnh, chưa áp ROI.
Không trộn các phiên bản ROI/weights/threshold khi tạo một dataset forecasting.

## Chuẩn bị notebook cloud

Bật GPU trong notebook; kiểm tra bằng `!nvidia-smi`. Không giả định loại GPU/quota.
Đưa repo/script và weights `src/models/yolo/best.pt` lên môi trường cloud.
Cài dependencies riêng, không cần cài BasicTS:

```python
%pip install -r /kaggle/working/tfp/requirements-inference.txt
```

Trên Colab đổi đường dẫn repo thành `/content/tfp`. PyTorch CUDA thường do runtime
cung cấp; xác nhận `torch.cuda.is_available()` trước khi chạy.

Xem class mapping của **custom weights**, không dùng class ID COCO theo thói quen:

```python
from ultralytics import YOLO
model = YOLO('/kaggle/working/tfp/src/models/yolo/best.pt')
print(model.names)
```

Chỉ chọn các ID tương ứng phương tiện. Ví dụ lệnh dưới giả sử 0,1,2,3 đều là xe;
**thay bằng mapping vừa in ra**.

## Kaggle

Gắn dữ liệu ảnh qua Kaggle input hoặc tải từng camera/ngày từ kho dữ liệu.
Nếu lưu ở Hugging Face, dùng HF_TOKEN trong notebook Secrets, không hard-code token.
Có thể dùng `hf download --help` để kiểm tra CLI của môi trường, ví dụ:

```bash
hf download ericng17/hcm_traffic_snapshots --repo-type dataset \
  --include 'snapshots/CAMERA_ID/*' --local-dir /kaggle/working/traffic
```

Repo chứa Parquet cần decode/xuất ảnh về cấu trúc trên trước. Chia dữ liệu theo camera/ngày
để không vượt ổ đĩa notebook; không cần tải cả vài trăm GB cùng lúc.

Benchmark khoảng 1.000 ảnh trước:

```bash
python /kaggle/working/tfp/scripts/infer_yolo.py \
  --source /kaggle/input/traffic/snapshots \
  --weights /kaggle/working/tfp/src/models/yolo/best.pt \
  --output /kaggle/working/yolo-counts \
  --device 0 --batch 4 --imgsz 1024 --classes 0 1 2 3 --limit 1000
```

Sau đó chạy cùng lệnh bỏ `--limit 1000`: ảnh đã commit được bỏ qua.
Bắt đầu batch=4 với YOLO11l/imgsz=1024, tăng 8/16 nếu VRAM cho phép.
Nếu CUDA OOM, chạy lại với batch nhỏ hơn. Thử `--half` trong output riêng trước khi
chốt cấu hình; không thay precision giữa một run đang resume. Không giảm imgsz
chỉ vì tốc độ khi chưa kiểm tra chất lượng đếm xe máy.

Log images/s gồm decode và inference, không gồm download và export cuối. Dùng benchmark
để ước tính thời gian thực tế; không suy ra tốc độ từ dung lượng GB.

## Colab

Ưu tiên copy từng phần ảnh từ Drive về SSD `/content` trước khi inference để tránh
đọc hàng triệu file nhỏ trực tiếp từ Drive. Có thể mount Drive bằng notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Dùng cùng script với `--source /content/traffic/snapshots`, weights và output dưới
`/content`. Định kỳ sao lưu **toàn bộ thư mục output**, đặc biệt `counts.sqlite`,
sang Drive sau khi dừng process hoặc dùng SQLite backup API nếu process còn chạy.
Colab/Kaggle có thể mất filesystem tạm khi runtime kết thúc. Checkpoint chỉ giúp resume
khi bạn giữ lại file; export CSV không thay thế database checkpoint.
Trên Kaggle lưu/download output trước khi mất phiên; lần sau copy checkpoint cũ vào
thư mục output có quyền ghi rồi chạy lại cùng lệnh. Không chạy hai process chung DB.

## Hai GPU hoặc nhiều notebook

Mỗi process có output riêng; chạy lệnh shard 0 trên GPU 0 và shard 1 trên GPU 1:

```bash
python scripts/infer_yolo.py --source /path/snapshots --weights /path/best.pt \
  --output /path/counts-0 --device 0 --classes 0 1 2 3 --num-shards 2 --shard-index 0
python scripts/infer_yolo.py --source /path/snapshots --weights /path/best.pt \
  --output /path/counts-1 --device 1 --classes 0 1 2 3 --num-shards 2 --shard-index 1
```

Chạy hai lệnh ở hai process riêng để song song. Trên hai notebook một GPU, đều dùng
`--device 0`. Hash theo đường dẫn tương đối bảo đảm shard không chồng nhau, miễn source
có cùng cấu trúc. Giữ nguyên num-shards và shard-index khi resume. Mỗi process quét metadata
của source nhưng chỉ decode ảnh thuộc shard. Ghép CSV bằng pandas.concat, kiểm tra
trùng `path` và cùng cấu hình run.json trước khi dùng.

## CPU dự phòng

```bash
python scripts/infer_yolo.py --source /path/snapshots \
  --weights src/models/yolo/best.pt --output data/detections/cpu \
  --device cpu --batch 1 --classes 0 1 2 3 --limit 100
```

Mặc định `auto` yêu cầu CUDA; script không tự rơi xuống CPU ngoài ý muốn.

## Đầu ra và resume

- `counts.sqlite`: checkpoint transaction theo batch, truy vấn theo path; RAM không giữ
  toàn bộ triệu kết quả. Không sửa input đã xử lý; ảnh thay đổi size/mtime sẽ bị từ chối.
- `run.json`: hash weights, mapping class, threshold, kích thước ảnh, precision, version.
  Thay cấu hình ảnh hưởng dự đoán phải tạo output mới. Batch/device được phép đổi.
- `counts.csv`: path, camera_id, timestamp có timezone, status, tổng xe, số xe từng class,
  error. Ảnh hỏng có status=error và số đếm trống; ảnh hợp lệ không có xe có số đếm 0.

CSV được export khi kết thúc/dừng thông thường, kể cả khi lỗi inference. Nếu runtime bị
kill cứng, CSV có thể chưa có nhưng SQLite giữ các batch đã commit. Chạy lại để resume,
hoặc cùng tham số thêm `--export-only` để xuất dữ liệu đã có. Cần dependencies/model để export.
Dùng `--retry-errors` khi muốn thử lại ảnh lỗi; ảnh thay đổi cần output mới.

Chỉ dùng snapshot dữ liệu bất biến trong từng run. Dữ liệu mới có thể thêm vào source;
chạy lại cùng output để xử lý tiếp. Không dùng thư mục nguồn đang ghi dở file.

CSV này là **quan sát từng ảnh**, chưa phải tensor STGCN. Bước sau cần chốt thứ tự camera,
resample 1/5 phút bằng mean/median, reindex đầy đủ timeline và tạo missing mask.
Ảnh không tải được không tồn tại trong source, vì vậy không tự xuất thành hàng lỗi ở đây.
Không cộng số đếm ảnh thành lưu lượng xe đi qua.
