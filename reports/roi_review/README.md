# ROI camera — bản dự thảo để duyệt

Được vẽ từ ảnh mẫu người dùng cung cấp, chưa áp dụng vào inference.

## Xem và duyệt

- Mở `review.html` bằng trình duyệt (double-click hoặc Open With → Browser). File tự chứa ảnh, không cần server/network.
- Chọn camera, kéo các đỉnh để sửa, dùng Ẩn ROI để so với ảnh gốc.
- Bấm Duyệt camera này cho từng ảnh đồng ý; sửa tọa độ sẽ đưa camera trở lại trạng thái dự thảo.
- Bấm Tải JSON đã chỉnh / duyệt và giữ file `camera_rois.reviewed.json`. Đóng trang trước khi tải sẽ mất chỉnh sửa.
- Có thể phản hồi bằng số camera trên ảnh, ví dụ “#20 bỏ vùng cầu vượt”.

## Các file

- `overview_01.jpg` — camera 01–09.
- `overview_02.jpg` — camera 10–18.
- `overview_03.jpg` — camera 19–27.
- `overview_04.jpg` — camera 28, Hoàng Văn Thụ – Út Tịch.
- `previews/` — ảnh overlay riêng từng camera.
- `../../configs/camera_rois.draft.json` — cấu hình gốc theo camera ID, có tọa độ chuẩn hóa và pixel tham chiếu.

## Phạm vi

Đủ 28 camera, mỗi camera một ảnh tham chiếu 512×288: 27 ảnh ban đêm và ảnh ban ngày của Hoàng Văn Thụ – Út Tịch (#28). Camera #28 dùng ảnh `20261114_104306.jpg` lấy từ repo `ericng17/hcm_traffic_snapshots5`, trùng khung ảnh người dùng cung cấp (overlay 10:43:05). Chưa đủ để xác nhận camera không đổi góc và biên ROI hợp cả ngày/đêm.

Đếm nếu điểm giữa cạnh dưới bounding box nằm trong hoặc trên biên ít nhất một đa giác. Mỗi detection chỉ đếm một lần trong hợp các vùng. Các vùng R1/R2 là hình học trên ảnh, không xác nhận hướng cạnh graph.
ROI gồm cả xe đang chờ đèn trong phần đường được chọn. Không điều chỉnh ROI theo vị trí xe từng frame. Vùng phía xa, biển quảng cáo, vỉa hè, đảo cây hoặc dải phân cách bị loại khi có thể xác định.

## Camera cần xem kỹ

| Số | Camera | Lưu ý |
|---|---|---|
| 03 | Cong_Hoa_Truong_Chinh | Ảnh đêm tối và lóa đèn, ranh phải khó thấy. Hai ROI nằm hai phía dải phân cách; cần xác nhận lại bằng ảnh sáng trước khi duyệt. |
| 06 | Cong_Hoa_Ut_Tich1 | Tách hai vùng đường hai bên đảo cây. Dây điện, biển báo/cột và tán cây che một phần ảnh; ranh phức tạp, cần ảnh ban ngày và người dùng xác nhận. |
| 07 | Nut_Giao_Bay_Hien2 | Tách tiếp cận gần và vùng đường phải rào; bỏ đảo giao thông và nhánh xa chói sáng. Kết nối hình học không khẳng định hướng đi thực tế. |
| 13 | Hoang_Van_Thu_Cong_Hoa | Loại đảo cây/đài trang trí lớn bên phải. Chỉ giữ dải mặt đường quan sát được quanh đảo; vùng tối cần kiểm tra bằng ảnh ngày. Biên hai vùng sát nhau, cộng theo union thay vì cộng box hai lần. |
| 19 | Cong_Hoa_Duong_C4 | Tên trên ảnh là Cộng Hòa - Đường A4, trong danh sách là C4; cần xác nhận metadata. Loại vùng xa phía trên và dải phân cách; ảnh có đốm lóa/mờ bên trái. Lưu ý metadata: tên crawler ghi Đường C4 nhưng chữ trên ảnh ghi Đường A4; giữ camera ID làm định danh. |
| 20 | Hoang_Van_Thu_Bui_Thi_Xuan | Hai tầng đường khác nhau: mặt đất và cầu vượt. Cần quyết định tách thành hai tín hiệu hoặc tổng hai ROI; lan can che chân xe trên cầu làm quy tắc đáy box kém ổn định. |
| 21 | 225_Hoang_Van_Thu | Ảnh ghi Hoàng Văn Thụ - Số 255, tên cấu hình 225_Hoang_Van_Thu. Bỏ vùng xa và thân cây/vỉa hè bên trái; cành cây vẫn che một phần ảnh. Lưu ý metadata: tên crawler ghi 225 nhưng chữ trên ảnh ghi Số 255 Hoàng Văn Thụ; giữ camera ID làm định danh. |
| 25 | Truong_Chinh_Truong_Cong_Dinh | Dải phân cách chéo chiếm nhiều khung hình. Loại đoạn xa bên trái và công trình phía trên; nhãn xe xa ban đêm khó kiểm chứng. Mép ROI trên phần đường xa cần kiểm tra bằng ảnh ngày. |
| 28 | Hoang_Van_Thu_Ut_Tich | Loại tủ điện tiền cảnh và vỉa hè Co.opmart; cần kiểm tra ảnh đêm trước khi chốt. |

## Tái tạo ảnh duyệt

```bash
python scripts/build_roi_review.py --config configs/camera_rois.draft.json
```

Bộ ảnh này chỉ kiểm tra hình học, chưa kiểm tra độ chính xác detector trong ROI. Sau khi duyệt mới tích hợp bộ lọc ROI vào inference.
