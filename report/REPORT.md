# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-07

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Not recorded / Not recorded / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không thay đổi.

### Từ cần nhớ

| Từ            | Nghĩa ngắn                                                      |
| ------------- | --------------------------------------------------------------- |
| Prediction    | Kết quả model vừa dự đoán; chưa phải nhãn đúng                  |
| Ground truth  | Nhãn do con người gán theo quy tắc                              |
| Taxonomy      | Danh sách tên lớp đi kèm model, ví dụ ImageNet-1K và COCO-80    |
| Checkpoint    | File model `.pt` đã được huấn luyện sẵn                         |
| Guideline     | Quy tắc gán nhãn do người viết                                  |
| Annotator     | Người thực hiện gán nhãn                                        |
| Reviewer / QC | Người kiểm tra chất lượng nhãn                                  |
| Escalation    | Chuyển trường hợp khó cho Lab Coach để được hướng dẫn           |
| Model score   | Điểm tin cậy của model, không phải điểm chất lượng ground truth |

## 1. Phân loại ảnh

**Sample:** `traffic`

**Evidence:** `classification_predictions.json` và `visuals/classification_top5.png`

**1. Record hạng 1:**

- `sample_id`: `traffic`
- `class_id`: 468
- `class_name`: `cab`
- `rank`: 1
- `score`: 0.510915
- `taxonomy_name`: `ImageNet-1K`

**2. Record này nói về cả bức ảnh hay một vật thể riêng trong ảnh? Vì sao?**

Record này mô tả toàn bộ bức ảnh (`image-level classification`), không phải vị trí của một vật thể riêng lẻ. Model đưa ra các lớp có khả năng mô tả nội dung tổng thể của ảnh và xếp hạng chúng theo score.

**3. Danh sách tên lớp (class list) do đâu mà có?**

Danh sách tên lớp được định nghĩa sẵn trong taxonomy đi kèm checkpoint của model, chẳng hạn ImageNet-1K. Model dự đoán trong phạm vi các lớp đã được định nghĩa, không tự tạo ra tên lớp mới.

**4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`?**

- `class_id`: Mã số định danh của lớp.
- `class_name`: Tên dễ đọc của lớp.
- `taxonomy_name`: Cho biết lớp thuộc hệ thống phân loại nào.

Ghi cả ba trường giúp xác định lớp chính xác, tránh nhầm lẫn khi các taxonomy khác nhau có thể sử dụng ID hoặc tên lớp khác nhau.

**5. Nếu trong ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần nói rõ điều gì?**

Guideline cần quy định cách chọn một lớp đại diện cho toàn bộ ảnh, ví dụ ưu tiên chủ thể chính hoặc nội dung chiếm ưu thế. Guideline cũng cần có quy tắc xử lý khi nhiều chủ thể có mức độ quan trọng tương đương để các annotator đưa ra kết quả nhất quán.

**6. Vì sao không được lấy `score` của model làm ground truth?**

Score chỉ thể hiện mức độ tin cậy của model đối với dự đoán. Nó không chứng minh dự đoán là chính xác. Ground truth phải được con người xác định dựa trên ảnh thực tế và guideline, sau đó được kiểm tra theo quy trình QC.

## 2. Phát hiện vật thể

**Sample:** `kitchen`

**Evidence:** `detection_predictions.json` và `visuals/detection_predictions.png`

**1. Một detection record:**

- `sample_id`: `kitchen`
- `class_name`: `person`
- `score`: 0.912625
- `bbox_xyxy`: `[385.33, 69.24, 498.92, 348.92]`
- `bbox_width`: 113.58 pixels
- `bbox_height`: 279.68 pixels

**2. Giải thích bounding box bằng lời thường:**

Bounding box sử dụng định dạng `xyxy`:

- `x1 = 385.33`, `y1 = 69.24`: tọa độ góc trên bên trái.
- `x2 = 498.92`, `y2 = 348.92`: tọa độ góc dưới bên phải.
- Chiều rộng của box là khoảng 113.58 pixels.
- Chiều cao của box là khoảng 279.68 pixels.

Box xác định vùng hình chữ nhật chứa đối tượng `person` trong ảnh.

**3. Số lượng prediction theo các threshold:**

| Threshold | Number of predictions |
| --------- | --------------------: |
| 0.20      |                    17 |
| 0.35      |                    11 |
| 0.60      |                     6 |

Khi threshold giảm từ 0.60 xuống 0.20, số prediction được giữ lại tăng từ 6 lên 17. Ngược lại, khi threshold tăng, những prediction có score thấp hơn ngưỡng sẽ bị loại bỏ.

**4. Khi hạ threshold, reviewer phải làm thêm việc gì?**

Khi threshold thấp hơn, nhiều prediction được giữ lại hơn, bao gồm cả những dự đoán có độ tin cậy thấp. Reviewer cần kiểm tra thêm các false positives, bounding box không chính xác, detection trùng lặp và các đối tượng có thể bị bỏ sót. Vì vậy, khối lượng công việc review có thể tăng.

**5. Một quy tắc bounding box:**

Bounding box cần bao quanh sát phần đối tượng nhìn thấy được, hạn chế đưa vùng background không cần thiết vào bên trong box. Với đối tượng bị che khuất hoặc bị cắt khỏi mép ảnh, annotator phải xử lý nhất quán theo guideline.

**6. Một đối tượng bị cắt mép trong ảnh kitchen:**

Trong ảnh Detection, có một người ở sát mép bên trái ảnh chỉ xuất hiện một phần cơ thể và bị cắt bởi ranh giới ảnh. Đây là trường hợp `truncated object`.

- **Annotator:** Áp dụng guideline về cách gán nhãn đối tượng chỉ xuất hiện một phần trong ảnh.
- **Reviewer/QC:** Kiểm tra cách xử lý có nhất quán với guideline hay không.
- **Escalation:** Nếu guideline chưa quy định rõ cách xử lý trường hợp này, annotator cần hỏi Lab Coach hoặc reviewer thay vì tự đưa ra quy tắc mới.

## 3. Phân đoạn theo từng đối tượng

**Sample:** `kitchen`

**Evidence:** `segmentation_predictions.json` và `visuals/segmentation_prediction.png`

**1. Một segmentation record:**

- `sample_id`: `kitchen`
- `instance_id`: `kitchen-001`
- `class_id`: 0
- `class_name`: `person`
- `score`: 0.899318
- `polygon_point_count`: 348

**Năm điểm đầu của `polygon_xy`:**

```text
[446.0, 70.0]
[445.0, 71.0]
[444.0, 71.0]
[443.0, 72.0]
[442.0, 72.0]
```

**2. So với bounding box, polygon cung cấp thêm thông tin gì?**

Bounding box biểu diễn đối tượng bằng một hình chữ nhật, trong khi polygon biểu diễn đường biên và hình dạng thực tế của đối tượng bằng nhiều điểm tọa độ. Vì vậy, polygon cung cấp thông tin chi tiết hơn về hình dáng và vùng đối tượng chiếm trong ảnh.

**3. `instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID có nghĩa là gì?**

`instance_id` dùng để định danh một đối tượng cụ thể trong kết quả prediction.

- `instance_id` xác định từng instance riêng biệt.
- `class_id` xác định loại đối tượng, ví dụ `person`.
- `tracking ID` dùng để nhận diện cùng một đối tượng qua nhiều frame trong video.

Do đó, `instance_id` không đại diện cho loại đối tượng và cũng không có nghĩa là mã theo dõi đối tượng xuyên suốt các frame video.

**4. Một quy tắc vẽ biên mask:**

Polygon cần bám sát đường biên có thể quan sát được của đối tượng, tránh đưa background hoặc các đối tượng lân cận vào mask. Annotator cần áp dụng cùng một quy tắc xử lý vùng bị che khuất theo guideline.

**5. Một vùng biên khó xác định trong ảnh kitchen:**

Trong ảnh Segmentation, khu vực mặt bàn (`dining table`) có nhiều vật thể đặt trên đó như bát, cốc, dụng cụ nhà bếp và khăn. Ranh giới giữa mặt bàn và những vật thể này khá gần nhau, khiến việc xác định đường biên mask cần được chú ý.

- **Annotator:** Tách mask của mặt bàn khỏi các vật thể đặt trên bàn theo guideline.
- **Reviewer/QC:** Kiểm tra mask boundary, các vùng background bị đưa nhầm vào mask và những vùng đối tượng bị thiếu.
- **Escalation:** Nếu không thể xác định chính xác đường biên hoặc guideline chưa đề cập đến trường hợp này, cần hỏi reviewer hoặc Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Chuỗi dưới đây mô tả vòng đời production:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                             | Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON                                                                                           | Annotator làm gì?                                                                    | Reviewer xem gì?                                                                                                        |
| --------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một class label cho toàn bộ ảnh, gồm class ID và class name theo taxonomy | Ảnh traffic có nhiều phương tiện; cần xác định lớp đại diện. Prediction `cab` có score 0.510915 nhưng chưa chứng minh đó là ground truth | Xác định lớp đại diện theo guideline và taxonomy                                     | Kiểm tra lớp có phù hợp nội dung ảnh và quy tắc chọn lớp hay không                                                      |
| Phát hiện vật thể     | Mỗi đối tượng có class label và bounding box `xyxy`                       | Ảnh kitchen có người bị cắt ở mép trái; số prediction thay đổi theo threshold từ 6 đến 17                                                | Xác định đối tượng, class và box theo guideline; xử lý đối tượng bị cắt theo quy tắc | Kiểm tra false positives, missed objects, box alignment, class và các trường hợp bị che/cắt                             |
| Instance segmentation | Mỗi instance có class label và polygon/mask riêng                         | Khu vực mặt bàn có nhiều vật thể đặt trên đó; ranh giới giữa các vật thể cần được phân biệt                                              | Vẽ polygon theo đường biên đối tượng và quy tắc xử lý vùng bị che                    | Kiểm tra mask boundary, vùng background bị đưa nhầm vào mask, vùng đối tượng bị thiếu và sự phân biệt giữa các instance |

## 5. An toàn dữ liệu

**1. Quy tắc bảo vệ dữ liệu:**

Chỉ sử dụng các ảnh COCO được cho phép trong phạm vi bài lab, không tải lên hoặc chia sẻ dữ liệu cá nhân, dữ liệu nhạy cảm hay dữ liệu nằm ngoài phạm vi được cấp phép.

**2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**

Tôi sẽ dừng xử lý dữ liệu đó và báo cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn, không tự ý tiếp tục sử dụng.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Đã kiểm tra báo cáo/output không chứa họ tên, MSSV hoặc dữ liệu nhạy cảm
- [ ] Đã nộp **link repository** trên LMS
