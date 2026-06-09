# README cực chi tiết: cải thiện độ chính xác cho đồ án Fall Detection

File này viết theo kiểu dễ hiểu nhất có thể. Mục tiêu là nếu một người chưa rành kỹ thuật đọc vào thì vẫn hiểu được:

- Project hiện tại đang làm gì.
- Vì sao hệ thống đôi khi nhận diện sai.
- Có những hướng nào để cải thiện độ chính xác.
- Mỗi hướng cần dùng gì, chuẩn bị dataset như nào.
- Nếu muốn làm fine-tune, deep learning, kết hợp model thì đi từng bước ra sao.
- Cuối cùng tính chỉ số thế nào để viết báo cáo đồ án.

Nói đơn giản, bài toán của mình là:

```text
Camera nhìn thấy một người.
Hệ thống phải biết người đó đang bình thường hay bị ngã.
Nếu bị ngã thì phải cảnh báo nhanh và ít báo nhầm nhất có thể.
```

## 1. Giải thích mấy từ quan trọng trước

Trước khi nói sâu, mình nên hiểu vài khái niệm. Phần này cố tình viết hơi đời thường.

### Model là gì?

Model là "bộ não" của hệ thống. Mình đưa ảnh hoặc video vào, model sẽ dự đoán kết quả.

Ví dụ:

```text
Ảnh camera -> model -> "đây là người"
Ảnh camera -> model -> "người này đang nằm"
Chuỗi chuyển động -> model -> "khả năng cao là bị ngã"
```

### Dataset là gì?

Dataset là bộ dữ liệu dùng để dạy hoặc kiểm tra model.

Trong bài fall detection, dataset có thể là:

- Video người bị ngã.
- Video người đi lại bình thường.
- Video người ngồi xuống.
- Video người cúi xuống nhặt đồ.
- Video người nằm nghỉ nhưng không phải ngã.
- Ảnh đã vẽ khung quanh người.
- File chứa tọa độ các điểm khớp của cơ thể.

Nếu model là học sinh thì dataset là sách bài tập. Sách càng đa dạng, học sinh càng ít học vẹt.

### Label là gì?

Label là nhãn đúng do con người gán cho dữ liệu.

Ví dụ:

```text
video_001.mp4 -> fall
video_002.mp4 -> normal
frame_000120.jpg -> falling_person
frame_000250.jpg -> lying_person
```

Model học từ label. Nếu label sai thì model cũng học sai.

### Train là gì?

Train là quá trình cho model học từ dataset.

Ví dụ mình có 1000 video, trong đó có video ngã và video không ngã. Mình đưa cho model học để sau này gặp video mới, model biết đoán.

### Fine-tune là gì?

Fine-tune nghĩa là lấy một model đã được học sẵn rồi dạy thêm cho đúng bài toán của mình.

Ví dụ YOLOv8 ban đầu đã biết phát hiện "person". Nhưng nó chưa chắc biết phân biệt:

```text
người đứng
người ngồi
người nằm
người đang ngã
```

Mình fine-tune YOLO bằng dataset fall detection để nó học thêm mấy trạng thái đó.

### Inference là gì?

Inference là lúc mình dùng model đã train xong để dự đoán trên ảnh/video thật.

Ví dụ:

```text
Camera đang chạy -> model dự đoán realtime -> nếu FALL thì cảnh báo
```

### False alarm là gì?

False alarm là báo động giả.

Ví dụ người chỉ cúi xuống buộc dây giày nhưng hệ thống báo "ngã". Đây là false alarm.

### Miss detection là gì?

Miss detection là bỏ sót.

Ví dụ người bị ngã thật nhưng hệ thống không báo. Trong bài toán này, bỏ sót thường nguy hiểm hơn báo nhầm.

## 2. Project hiện tại đang hoạt động như thế nào?

Project hiện tại dùng 3 phần chính:

```text
YOLOv8 -> MediaPipe Pose -> Rule-based FallDetector
```

Hiểu kiểu đơn giản:

```text
YOLOv8: "Trong ảnh có người ở chỗ này"
MediaPipe: "Đây là vị trí đầu, vai, hông, gối, cổ chân..."
FallDetector: "Nhìn góc nghiêng và chuyển động này thì có vẻ người này bị ngã"
```

### 2.1. YOLOv8 làm gì?

YOLOv8 đang được dùng để phát hiện người trong khung hình.

Nó không kết luận người đó có ngã hay không. Nó chỉ nói:

```text
Ở tọa độ này có một người.
```

Ví dụ output của YOLO có thể là:

```text
x1 = 100
y1 = 80
x2 = 350
y2 = 460
confidence = 0.86
```

Nghĩa là model thấy một người nằm trong cái khung từ điểm `(100, 80)` đến `(350, 460)`, và độ tin cậy là 86%.

### 2.2. MediaPipe Pose làm gì?

Sau khi YOLO tìm được người, hệ thống crop vùng người đó ra rồi đưa vào MediaPipe.

MediaPipe Pose sẽ lấy các điểm khớp trên cơ thể, ví dụ:

- Mũi.
- Vai trái, vai phải.
- Hông trái, hông phải.
- Gối trái, gối phải.
- Cổ chân trái, cổ chân phải.

MediaPipe có thể trả về 33 pose landmarks. Mỗi landmark thường có:

```text
x: vị trí ngang
y: vị trí dọc
z: độ sâu tương đối
visibility: độ tin cậy điểm đó có nhìn rõ không
```

### 2.3. FallDetector hiện tại làm gì?

FallDetector trong project hiện tại chưa phải deep learning. Nó là rule-based, tức là dùng luật thủ công.

Nó kiểm tra vài dấu hiệu:

```text
1. Góc nghiêng cơ thể có lớn không?
2. Bounding box có bị nằm ngang không?
3. Người có rơi xuống nhanh không?
4. Vị trí các khớp có bất thường không?
```

Sau đó cộng điểm:

```text
Nếu có ít nhất 2 dấu hiệu nghi ngờ
và trạng thái này kéo dài đủ lâu
=> báo FALL
```

Nói nôm na:

```text
Nếu người nghiêng nhiều + khung người bè ngang
=> có thể ngã.

Nếu thêm chuyển động rơi xuống nhanh
=> càng chắc là ngã.
```

## 3. Vì sao hệ thống hiện tại có thể sai?

Hệ thống fall detection khó vì nhiều hành động nhìn khá giống nhau.

Ví dụ:

```text
Người bị ngã thật.
Người cúi xuống nhặt đồ.
Người ngồi xuống rất nhanh.
Người nằm nghỉ trên sàn.
Người tập thể dục.
Người bị che một phần cơ thể.
```

Camera chỉ nhìn thấy ảnh 2D, nên nhiều lúc rất khó phân biệt.

### 3.1. Sai vì góc camera

Nếu camera đặt ngang hông, người ngã sẽ nhìn khác với camera đặt trên cao.

Một ngưỡng `angle_threshold = 45` có thể đúng với camera này, nhưng sai với camera khác.

### 3.2. Sai vì người gần/xa camera

Hiện tại vận tốc được tính theo pixel/giây.

Người đứng gần camera chỉ cần di chuyển một chút là pixel thay đổi rất nhiều. Người đứng xa camera thì ngược lại.

Vì vậy nên chuẩn hóa vận tốc theo chiều cao người:

```text
vy_norm = vy / bbox_height
```

Nghĩa là vận tốc được tính theo tỉ lệ cơ thể, không phụ thuộc quá nhiều vào khoảng cách camera.

### 3.3. Sai vì tư thế giống ngã

Người cúi xuống, ngồi xuống, nằm nghỉ có thể làm:

```text
góc thân người lớn
bounding box bè ngang
đầu thấp hơn bình thường
```

Các dấu hiệu này khá giống ngã, nên rule-based dễ báo nhầm.

### 3.4. Sai vì chỉ nhìn từng khoảnh khắc

Ngã là một hành động theo thời gian.

Một frame đơn lẻ có thể không đủ thông tin. Ví dụ:

```text
Frame hiện tại: người đang nằm.
```

Nhưng có 2 khả năng:

```text
1. Người vừa bị ngã.
2. Người đang nằm nghỉ từ trước.
```

Muốn phân biệt, mình cần nhìn chuỗi frame trước đó:

```text
Trước đó người đang đứng -> rơi nhanh -> nằm im
=> khả năng cao là ngã.

Trước đó người đã nằm sẵn
=> không nên báo ngã.
```

## 4. Tổng quan các hướng cải thiện

Mình có thể cải thiện theo nhiều mức. Không nhất thiết làm hết. Nên đi từ dễ đến khó.

```text
Mức 1: Cải thiện rule-based hiện tại.
Mức 2: Chuẩn bị dataset và đánh giá nghiêm túc.
Mức 3: Fine-tune YOLO để nhận diện trạng thái người.
Mức 4: Train machine learning trên skeleton.
Mức 5: Train deep learning theo chuỗi thời gian như LSTM/GRU/TCN.
Mức 6: Dùng ST-GCN cho skeleton.
Mức 7: Kết hợp nhiều model để ra quyết định cuối.
```

Nếu cần chọn hướng thực tế nhất cho đồ án, mình đề xuất:

```text
Baseline rule-based
-> thêm temporal voting + normalized velocity
-> train skeleton model bằng LSTM hoặc TCN
-> nếu còn thời gian thì fusion với YOLO fine-tuned
```

Lý do:

- Dễ giải thích.
- Có học máy/deep learning thật.
- Không quá nặng như video transformer.
- Dễ viết phần thực nghiệm và so sánh.

## 5. Hướng 1: cải thiện rule-based hiện tại

Đây là hướng dễ nhất vì không cần train deep learning.

Mình vẫn dùng hệ thống hiện tại, nhưng sửa cách ra quyết định cho thông minh hơn.

### 5.1. Tune threshold bằng dữ liệu thật

Hiện tại các ngưỡng như:

```text
angle_threshold = 45
aspect_ratio_threshold = 1.2
vertical_velocity_threshold = 10
fall_time_threshold = 1.0
```

có vẻ được chọn thủ công.

Cách tốt hơn:

```text
1. Chuẩn bị một tập video validation.
2. Chạy hệ thống với nhiều bộ threshold khác nhau.
3. Tính Precision, Recall, F1-score.
4. Chọn bộ threshold có kết quả tốt nhất.
```

Ví dụ thử các giá trị:

```text
angle_threshold: 35, 40, 45, 50, 55, 60
aspect_ratio_threshold: 1.0, 1.1, 1.2, 1.3, 1.4
fall_time_threshold: 0.5, 0.75, 1.0, 1.25, 1.5
```

Sau đó mình có bảng:

| Angle | Aspect ratio | Time | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 45 | 1.2 | 1.0 | 0.70 | 0.82 | 0.75 |
| 50 | 1.2 | 1.0 | 0.78 | 0.76 | 0.77 |
| 40 | 1.1 | 0.75 | 0.62 | 0.91 | 0.74 |

Nhìn bảng này mình sẽ biết trade-off:

```text
Threshold dễ hơn -> Recall cao nhưng báo nhầm nhiều.
Threshold khó hơn -> Precision cao nhưng dễ bỏ sót.
```

Với fall detection, thường ưu tiên Recall cao, vì bỏ sót ngã nguy hiểm.

### 5.2. Dùng temporal voting

Temporal voting nghĩa là không quyết định chỉ bằng một frame, mà nhìn nhiều frame gần nhất.

Ví dụ:

```text
Trong 15 frame gần nhất:
Nếu có ít nhất 10 frame nghi ngờ ngã
=> báo WARNING hoặc FALL
```

Cách này giúp giảm báo động giả do một frame bị lỗi.

Ví dụ người xoay người nhanh:

```text
Frame 1: bình thường
Frame 2: góc nghiêng lớn
Frame 3: bình thường
```

Nếu chỉ nhìn frame 2 thì dễ báo nhầm. Nếu voting 15 frame thì sẽ ổn hơn.

### 5.3. Thêm logic "sau khi ngã"

Một cú ngã thường có 3 pha:

```text
1. Trước ngã: người đang đứng/đi.
2. Trong lúc ngã: cơ thể rơi nhanh, góc nghiêng thay đổi nhanh.
3. Sau ngã: người nằm thấp hoặc nằm ngang một thời gian.
```

Nếu chỉ có pha 2 mà không có pha 3, có thể đó là người ngồi xuống nhanh.

Vì vậy nên thêm điều kiện:

```text
Nếu nghi ngờ ngã
và sau đó người nằm/thấp trong ít nhất N frame
=> mới xác nhận FALL
```

### 5.4. Xử lý nhiều người bằng tracking

Hiện tại code đang lấy detection đầu tiên. Nếu trong camera có 2 người, hệ thống có thể bỏ qua người còn lại.

Cách cải thiện:

```text
YOLO detect tất cả người
-> tracker gán ID cho từng người
-> mỗi ID có lịch sử pose và vận tốc riêng
-> fall detector chạy riêng cho từng người
```

Ví dụ:

```text
person_id = 1: đang đứng
person_id = 2: bị ngã
```

Nếu không tracking, lịch sử vận tốc của người này có thể bị lẫn với người kia.

### 5.5. Khi nào nên chọn hướng rule-based?

Nên chọn nếu:

- Chưa có dataset đủ lớn.
- Cần demo nhanh.
- Muốn hệ thống chạy nhẹ.
- Muốn báo cáo dễ giải thích.

Không nên chỉ dừng ở rule-based nếu:

- Muốn độ chính xác cao ở nhiều bối cảnh.
- Camera thay đổi nhiều góc.
- Có nhiều hành động dễ nhầm như ngồi, cúi, nằm nghỉ.

## 6. Hướng 2: chuẩn bị dataset cho đúng

Đây là phần rất quan trọng. Model tốt hay không phụ thuộc rất nhiều vào dataset.

Nhiều đồ án bị lỗi ở chỗ model nhìn kết quả rất cao, nhưng ra video thật lại sai. Lý do thường là dataset chia không đúng hoặc quá đơn giản.

### 6.1. Dataset cần có những loại hành động nào?

Không nên chỉ có "ngã" và "đứng yên". Vì ngoài đời có nhiều hành động giống ngã.

Dataset nên có:

#### Nhóm fall

- Ngã về phía trước.
- Ngã về phía sau.
- Ngã sang trái.
- Ngã sang phải.
- Ngã từ tư thế đang đi.
- Ngã từ tư thế đang đứng.
- Ngã từ ghế nếu bài toán có liên quan.

#### Nhóm normal

- Đi bộ.
- Đứng yên.
- Ngồi xuống.
- Đứng dậy.
- Cúi xuống nhặt đồ.
- Nằm xuống nghỉ.
- Tập thể dục nhẹ.
- Quỳ xuống.
- Xoay người.

Điểm quan trọng:

```text
Phải có nhiều hành động normal nhìn giống fall.
Nếu không, model sẽ học quá dễ và ra ngoài đời dễ báo nhầm.
```

### 6.2. Cần quay bao nhiêu video?

Nếu là đồ án sinh viên, có thể bắt đầu như sau:

```text
Fall: 100-200 clip
Normal: 200-400 clip
```

Nếu ít thời gian hơn:

```text
Fall: 40-60 clip
Normal: 80-120 clip
```

Mỗi clip khoảng:

```text
3-10 giây
```

Không cần clip quá dài. Quan trọng là clip rõ ràng và đa dạng.

### 6.3. Nên quay ở những bối cảnh nào?

Dataset nên thay đổi:

- Người khác nhau.
- Quần áo khác nhau.
- Góc camera khác nhau.
- Khoảng cách camera khác nhau.
- Ánh sáng khác nhau.
- Nền phòng khác nhau.

Ví dụ:

```text
Camera ngang người.
Camera hơi cao.
Camera đặt góc phòng.
Người đứng gần camera.
Người đứng xa camera.
Phòng sáng.
Phòng hơi tối.
```

Nếu chỉ quay một người, một phòng, một góc camera thì model dễ học thuộc bối cảnh.

### 6.4. Cảnh báo an toàn khi tự quay video ngã

Nếu tự quay fall dataset, không nên ngã thật trên nền cứng.

Nên chuẩn bị:

- Nệm.
- Thảm dày.
- Người hỗ trợ.
- Không quay động tác nguy hiểm.
- Có thể giả lập ngã chậm hơn, miễn là label rõ ràng.

Đồ án không đáng để bị chấn thương.

### 6.5. Cách chia train/validation/test

Mình nên chia dataset thành 3 phần:

```text
Train: dùng để model học.
Validation: dùng để chọn threshold, chọn model, chỉnh tham số.
Test: chỉ dùng cuối cùng để báo cáo kết quả.
```

Tỉ lệ phổ biến:

```text
70% train
15% validation
15% test
```

Nhưng với fall detection, tốt nhất là chia theo người hoặc theo video.

Không nên làm kiểu:

```text
Lấy frame 1-100 của cùng một video cho train.
Lấy frame 101-150 của chính video đó cho test.
```

Vì như vậy model đã nhìn gần như cùng một cảnh rồi. Kết quả test sẽ đẹp nhưng không thật.

Cách tốt hơn:

```text
Người A, B, C -> train
Người D -> validation
Người E -> test
```

Hoặc:

```text
Video quay ở phòng 1 -> train
Video quay ở phòng 2 -> validation
Video quay ở phòng 3 -> test
```

### 6.6. Cấu trúc thư mục dataset raw

Mình có thể tổ chức video gốc như sau:

```text
datasets/
  fall_detection_raw/
    videos/
      fall/
        fall_001.mp4
        fall_002.mp4
      normal/
        normal_001.mp4
        normal_002.mp4
    metadata.csv
```

File `metadata.csv` có thể như này:

```csv
video_path,label,subject,scene,camera_view,note
videos/fall/fall_001.mp4,fall,S01,room_1,front,fall forward
videos/fall/fall_002.mp4,fall,S02,room_1,side,fall left
videos/normal/normal_001.mp4,normal,S01,room_1,front,sit down
videos/normal/normal_002.mp4,normal,S03,room_2,side,pick object
```

Tại sao cần `subject`, `scene`, `camera_view`?

Vì sau này khi chia train/test, mình dễ tránh việc cùng một người hoặc cùng một cảnh xuất hiện ở cả train và test.

## 7. Có mấy kiểu label dataset?

Tùy hướng cải thiện mà mình cần label khác nhau.

### 7.1. Label theo video

Dễ nhất:

```text
video_001.mp4 -> fall
video_002.mp4 -> normal
```

Phù hợp khi:

- Muốn train model theo clip.
- Muốn đánh giá video-level.
- Chưa cần biết chính xác frame nào bắt đầu ngã.

Nhược điểm:

- Không biết model báo sớm hay muộn.
- Khó tính latency chính xác.

### 7.2. Label theo frame

Chi tiết hơn:

```text
frame 0-80: normal
frame 81-110: falling
frame 111-180: fallen
```

Hoặc đơn giản:

```text
frame,label
0,normal
1,normal
...
81,fall
82,fall
```

Phù hợp khi:

- Muốn tính Precision/Recall theo frame.
- Muốn biết hệ thống báo trễ bao lâu.
- Muốn train model theo sliding window.

Nhược điểm:

- Mất công annotate hơn.

### 7.3. Label theo đoạn thời gian

Đây là cách cân bằng giữa dễ và chi tiết.

Ví dụ:

```csv
video,start_time,end_time,label
fall_001.mp4,0.0,2.1,normal
fall_001.mp4,2.1,3.0,falling
fall_001.mp4,3.0,6.0,fallen
normal_001.mp4,0.0,5.0,normal
```

Cách này rất hợp cho báo cáo vì mình biết:

```text
Ngã bắt đầu ở giây 2.1.
Hệ thống báo ở giây 2.6.
Latency = 0.5 giây.
```

### 7.4. Label bounding box cho YOLO

Nếu muốn fine-tune YOLO, mình cần annotate bounding box.

Ví dụ trong một frame, vẽ khung quanh người rồi gán class:

```text
normal_person
falling_person
fallen_person
sitting_person
```

YOLO label thường có dạng:

```text
class_id x_center y_center width height
```

Các giá trị `x_center`, `y_center`, `width`, `height` được chuẩn hóa từ 0 đến 1.

Ví dụ:

```text
0 0.512 0.433 0.280 0.620
```

Nghĩa là:

```text
class 0
tâm box ở 51.2% chiều rộng ảnh
tâm box ở 43.3% chiều cao ảnh
box rộng 28.0% ảnh
box cao 62.0% ảnh
```

### 7.5. Label cho skeleton model

Nếu dùng MediaPipe + LSTM/TCN/ST-GCN, mình không cần vẽ bounding box thủ công cho từng frame nếu đã có video label.

Mình sẽ:

```text
Video -> chạy MediaPipe -> lấy landmarks -> lưu thành file .npy hoặc .csv -> gán label fall/normal cho từng window
```

Ví dụ một sample:

```text
30 frame liên tiếp
mỗi frame có 33 điểm khớp
mỗi điểm có x, y, z, visibility
label = fall
```

Kích thước dữ liệu:

```text
30 x 33 x 4
```

Trong đó:

```text
30: số frame
33: số điểm khớp
4: x, y, z, visibility
```

## 8. Hướng 3: fine-tune YOLO

Fine-tune YOLO nghĩa là dạy YOLO nhận diện thêm các trạng thái liên quan đến ngã.

Hiện tại YOLO chỉ biết:

```text
person
```

Mình muốn nó biết thêm:

```text
normal_person
falling_person
fallen_person
sitting_person
```

### 8.1. Khi nào nên fine-tune YOLO?

Nên dùng nếu:

- Muốn detect người đang nằm/ngã trực tiếp từ ảnh.
- Muốn có một model chạy realtime.
- Có thể annotate bounding box.
- Muốn báo cáo có phần fine-tuning rõ ràng.

Không nên chỉ dùng YOLO nếu:

- Muốn phân biệt "nằm nghỉ" và "vừa bị ngã".
- Dữ liệu video có nhiều chuyển động cần hiểu theo thời gian.

Lý do là YOLO thường nhìn từng ảnh riêng lẻ. Nó không biết trước đó người đứng rồi rơi xuống hay đã nằm sẵn từ đầu.

### 8.2. Chuẩn bị dataset YOLO như nào?

Cấu trúc thư mục thường như sau:

```text
datasets/
  fall_yolo/
    images/
      train/
        img_000001.jpg
        img_000002.jpg
      val/
        img_000101.jpg
      test/
        img_000201.jpg
    labels/
      train/
        img_000001.txt
        img_000002.txt
      val/
        img_000101.txt
      test/
        img_000201.txt
    fall.yaml
```

Mỗi ảnh trong `images/train` phải có file label tương ứng trong `labels/train`.

Ví dụ:

```text
images/train/img_000001.jpg
labels/train/img_000001.txt
```

Nội dung file label:

```text
0 0.512 0.433 0.280 0.620
```

Nếu một ảnh có 2 người, file label có 2 dòng:

```text
0 0.512 0.433 0.280 0.620
2 0.220 0.700 0.300 0.250
```

### 8.3. File `fall.yaml`

Ví dụ:

```yaml
path: datasets/fall_yolo
train: images/train
val: images/val
test: images/test

names:
  0: normal_person
  1: falling_person
  2: fallen_person
  3: sitting_person
```

Trong đó:

- `path`: thư mục gốc dataset.
- `train`: ảnh dùng để train.
- `val`: ảnh dùng để validation.
- `test`: ảnh dùng để test.
- `names`: tên các class.

### 8.4. Lấy ảnh từ video thế nào?

Nếu video dài, không cần lấy mọi frame. Có thể lấy 5-10 frame/giây.

Ví dụ:

```text
Video 30 FPS
Lấy mỗi 5 frame một lần
=> khoảng 6 ảnh/giây
```

Lý do:

```text
Các frame liên tiếp thường gần giống nhau.
Lấy quá nhiều frame giống nhau làm dataset bị lặp.
```

### 8.5. Annotate bằng công cụ nào?

Có thể dùng:

- LabelImg.
- CVAT.
- Roboflow.
- MakeSense.ai.

Quy trình:

```text
1. Tách video thành ảnh.
2. Mở ảnh bằng tool annotate.
3. Vẽ box quanh người.
4. Chọn class đúng.
5. Export format YOLO.
6. Chia train/val/test.
7. Train YOLO.
```

### 8.6. Train YOLO

Sau khi có dataset, chạy:

```bash
yolo detect train model=yolov8s.pt data=datasets/fall_yolo/fall.yaml epochs=100 imgsz=640 batch=16
```

Giải thích:

```text
model=yolov8s.pt: dùng YOLOv8 small pretrained.
data=...: file cấu hình dataset.
epochs=100: học 100 vòng.
imgsz=640: resize ảnh về 640.
batch=16: mỗi lần học 16 ảnh.
```

Nếu máy yếu, dùng:

```bash
yolo detect train model=yolov8n.pt data=datasets/fall_yolo/fall.yaml epochs=50 imgsz=640 batch=8
```

Nếu có GPU:

```bash
yolo detect train model=yolov8s.pt data=datasets/fall_yolo/fall.yaml epochs=100 imgsz=640 batch=16 device=0
```

### 8.7. Test YOLO sau train

Sau khi train xong, model thường nằm ở:

```text
runs/detect/train/weights/best.pt
```

Chạy thử:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=test_video.mp4
```

Nếu kết quả ổn, mình có thể thay model trong project:

```python
YOLO_CONFIG = {
    'model': 'runs/detect/train/weights/best.pt',
    ...
}
```

### 8.8. YOLO giúp cải thiện phần nào?

YOLO fine-tuned giúp hệ thống biết trạng thái trực quan:

```text
người đứng
người ngồi
người nằm
người đang ngã
```

Nhưng nó vẫn yếu ở phần "hành động theo thời gian".

Vì vậy nên kết hợp YOLO với skeleton model hoặc rule-based temporal logic.

## 9. Hướng 4: machine learning trên skeleton

Đây là hướng rất hợp với project hiện tại.

Vì project đã có MediaPipe Pose rồi, mình có thể tận dụng landmarks để train model.

### 9.1. Ý tưởng chính

Thay vì tự viết luật:

```text
Nếu góc > 45 và aspect ratio > 1.2 thì nghi ngờ ngã
```

mình cho model học từ dữ liệu:

```text
Đây là pose và chuyển động của người bị ngã.
Đây là pose và chuyển động của người bình thường.
Model tự tìm pattern.
```

### 9.2. Dữ liệu đầu vào là gì?

Từ MediaPipe, mỗi frame có 33 điểm.

Mỗi điểm có:

```text
x, y, z, visibility
```

Nếu lấy một frame:

```text
33 x 4 = 132 giá trị
```

Nếu lấy 30 frame:

```text
30 x 33 x 4 = 3960 giá trị
```

### 9.3. Nên dùng feature nào?

Có 2 cách.

#### Cách A: dùng raw landmarks

Mình lấy thẳng:

```text
x, y, z, visibility của 33 landmarks
```

Ưu điểm:

- Dễ làm.
- Không cần nghĩ quá nhiều.
- Hợp với deep learning như LSTM/TCN.

Nhược điểm:

- Model có thể bị ảnh hưởng bởi vị trí người trong ảnh.
- Nên normalize trước khi train.

#### Cách B: tự tạo feature dễ hiểu

Mình tính thêm:

- Góc thân người.
- Tỉ lệ width/height.
- Vận tốc hông.
- Vận tốc vai.
- Vận tốc đầu.
- Khoảng cách vai-hông.
- Khoảng cách hông-cổ chân.
- Độ cao đầu so với hông.
- Độ cao hông so với cổ chân.

Ưu điểm:

- Dễ giải thích trong báo cáo.
- Hợp với Random Forest/SVM/XGBoost.

Nhược điểm:

- Phải tự thiết kế feature.
- Có thể bỏ sót pattern mà mình không nghĩ tới.

### 9.4. Normalize skeleton như nào?

Normalize nghĩa là đưa dữ liệu về dạng ổn định hơn.

Vì người trong ảnh có thể đứng gần/xa, cao/thấp, trái/phải, mình nên chuẩn hóa:

#### Bước 1: lấy hông làm gốc

Tính tâm hông:

```text
hip_center = trung bình của left_hip và right_hip
```

Sau đó trừ tất cả điểm cho `hip_center`.

Ý nghĩa:

```text
Người đứng bên trái hay bên phải ảnh đều được đưa về cùng hệ tọa độ tương đối.
```

#### Bước 2: chia theo kích thước cơ thể

Ví dụ dùng khoảng cách vai-hông hoặc chiều cao bounding box:

```text
normalized_point = point / body_size
```

Ý nghĩa:

```text
Người gần camera và xa camera sẽ bớt khác nhau.
```

#### Bước 3: xử lý điểm không nhìn rõ

Nếu `visibility` thấp, điểm đó không đáng tin.

Có thể:

- Giữ visibility làm feature.
- Thay điểm thiếu bằng giá trị frame trước đó.
- Bỏ frame nếu quá nhiều điểm bị mất.

### 9.5. Train model cổ điển: Random Forest, SVM, XGBoost

Nếu muốn dễ làm nhất, bắt đầu bằng Random Forest.

Pipeline:

```text
Video
-> MediaPipe lấy landmarks
-> Trích xuất feature
-> Lưu thành CSV
-> Train Random Forest/SVM/XGBoost
-> Dự đoán fall/normal
```

File CSV ví dụ:

```csv
sample_id,angle,aspect_ratio,hip_vy,head_vy,shoulder_hip_dist,label
001,20.5,0.65,0.02,0.01,0.45,normal
002,72.1,1.42,0.85,0.91,0.20,fall
```

Train thử:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("skeleton_features.csv")

X = df.drop(columns=["sample_id", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

Ưu điểm:

- Dễ train.
- Dễ hiểu.
- Không cần GPU.
- Dễ viết báo cáo.

Nhược điểm:

- Nếu feature không tốt thì kết quả không cao.
- Học temporal pattern chưa mạnh bằng LSTM/TCN.

## 10. Hướng 5: deep learning theo chuỗi thời gian

Ngã là một quá trình, nên model nên nhìn nhiều frame liên tiếp.

Ví dụ:

```text
Frame 1-10: người đứng
Frame 11-20: người mất thăng bằng
Frame 21-30: người rơi xuống
Frame 31-40: người nằm
```

Model nhìn chuỗi này sẽ phân biệt tốt hơn việc chỉ nhìn một ảnh.

### 10.1. Sliding window là gì?

Sliding window là cắt video thành nhiều đoạn ngắn.

Ví dụ video có 100 frame, window size là 30 frame:

```text
Sample 1: frame 0-29
Sample 2: frame 5-34
Sample 3: frame 10-39
Sample 4: frame 15-44
```

Mỗi sample có label:

```text
fall hoặc normal
```

Nếu trong window có đoạn ngã, label là `fall`. Nếu không có thì label là `normal`.

### 10.2. Input cho LSTM/GRU/TCN

Ví dụ:

```text
window_size = 30
num_landmarks = 33
num_features = 4
```

Input:

```text
30 x 33 x 4
```

Có thể flatten mỗi frame:

```text
30 x 132
```

Trong đó:

```text
132 = 33 landmarks * 4 features
```

### 10.3. LSTM là gì?

LSTM là model chuyên đọc dữ liệu theo chuỗi.

Nó phù hợp khi dữ liệu có thứ tự thời gian:

```text
frame 1 -> frame 2 -> frame 3 -> ...
```

Trong bài này, LSTM học:

```text
trước khi ngã cơ thể như nào
trong lúc ngã chuyển động ra sao
sau khi ngã tư thế thế nào
```

### 10.4. GRU là gì?

GRU giống LSTM nhưng nhẹ hơn một chút.

Nếu máy yếu hoặc dataset nhỏ, có thể thử GRU.

### 10.5. TCN là gì?

TCN là Temporal Convolutional Network.

Hiểu đơn giản:

```text
TCN dùng convolution theo trục thời gian để học chuyển động.
```

TCN thường train nhanh, ổn định, và rất hợp với dữ liệu chuỗi ngắn như 30-60 frame.

### 10.6. Nên chọn LSTM, GRU hay TCN?

Nếu muốn dễ giải thích:

```text
LSTM
```

Nếu muốn nhẹ:

```text
GRU
```

Nếu muốn kết quả thực tế tốt và train nhanh:

```text
TCN
```

Với đồ án này, mình đề xuất:

```text
Bắt đầu với LSTM hoặc TCN.
```

### 10.7. Dataset cho LSTM/TCN chuẩn bị như nào?

Cấu trúc có thể như sau:

```text
datasets/
  skeleton_windows/
    X_train.npy
    y_train.npy
    X_val.npy
    y_val.npy
    X_test.npy
    y_test.npy
```

Trong đó:

```text
X_train.shape = (num_samples, 30, 132)
y_train.shape = (num_samples,)
```

Ví dụ:

```text
X_train.shape = (2500, 30, 132)
y_train.shape = (2500,)
```

Nghĩa là có 2500 sample, mỗi sample là 30 frame, mỗi frame có 132 giá trị.

### 10.8. Model LSTM ví dụ

Ví dụ bằng PyTorch:

```python
import torch
import torch.nn as nn

class FallLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)
```

Output là xác suất:

```text
0.0 -> chắc là normal
1.0 -> chắc là fall
```

Mình chọn threshold:

```text
Nếu p_fall >= 0.7 => FALL
Nếu p_fall từ 0.4 đến 0.7 => WARNING
Nếu p_fall < 0.4 => NORMAL
```

Threshold này cũng nên tune bằng validation set.

## 11. Hướng 6: ST-GCN cho skeleton

ST-GCN là hướng nâng cao hơn.

Tên đầy đủ:

```text
Spatial Temporal Graph Convolutional Network
```

Nghe khó nhưng ý tưởng khá hợp lý.

### 11.1. Vì sao skeleton là graph?

Cơ thể người có các khớp:

```text
vai
khuỷu tay
cổ tay
hông
gối
cổ chân
```

Các khớp nối với nhau bằng xương.

Vậy mình có thể xem:

```text
Mỗi khớp = một node
Mỗi xương nối giữa hai khớp = một edge
```

Đây chính là graph.

### 11.2. ST-GCN học cái gì?

ST-GCN học 2 thứ:

```text
Spatial: quan hệ giữa các khớp trong cùng một frame.
Temporal: chuyển động của các khớp qua nhiều frame.
```

Ví dụ:

```text
Spatial: vai nối với hông, hông nối với gối.
Temporal: hông ở frame 1, frame 2, frame 3 di chuyển xuống rất nhanh.
```

### 11.3. Khi nào nên dùng ST-GCN?

Nên dùng nếu:

- Muốn hướng deep learning mạnh hơn LSTM.
- Dataset đủ nhiều.
- Có thời gian triển khai.
- Muốn báo cáo có phần mô hình nâng cao.

Không nên dùng nếu:

- Dataset ít.
- Chưa xử lý được landmarks sạch.
- Thời gian đồ án gấp.

### 11.4. ST-GCN cần dữ liệu gì?

Tương tự LSTM, nhưng giữ cấu trúc khớp rõ ràng:

```text
num_samples x channels x time x joints
```

Ví dụ:

```text
N x 4 x 30 x 33
```

Trong đó:

```text
N: số sample
4: x, y, z, visibility
30: số frame
33: số khớp
```

Ngoài ra cần adjacency matrix, tức là ma trận mô tả khớp nào nối với khớp nào.

Ví dụ:

```text
vai trái nối vai phải
vai trái nối hông trái
hông trái nối gối trái
gối trái nối cổ chân trái
```

### 11.5. ST-GCN có đáng làm không?

Nếu mục tiêu là đồ án tốt và vừa sức:

```text
LSTM/TCN là đủ.
```

Nếu mục tiêu là nâng cấp nghiên cứu hơn:

```text
ST-GCN rất đáng thử.
```

Mình có thể viết trong báo cáo:

```text
Do dữ liệu pose có cấu trúc graph tự nhiên, ST-GCN là hướng mở rộng phù hợp để mô hình học đồng thời quan hệ không gian giữa các khớp và quan hệ thời gian trong quá trình ngã.
```

## 12. Hướng 7: kết hợp nhiều model

Trong hệ thống thật, thường không nên phụ thuộc vào một nguồn thông tin duy nhất.

Mình có thể kết hợp:

```text
Rule-based detector
YOLO fine-tuned
Skeleton sequence model
```

### 12.1. Vì sao cần fusion?

Mỗi model có điểm mạnh/yếu riêng.

#### Rule-based

Mạnh:

- Dễ hiểu.
- Chạy nhanh.
- Không cần train.

Yếu:

- Nhạy với threshold.
- Dễ sai khi góc camera thay đổi.

#### YOLO fine-tuned

Mạnh:

- Detect trực tiếp người nằm/ngã.
- Chạy realtime.
- Hữu ích khi pose bị mất.

Yếu:

- Nhìn từng frame.
- Dễ nhầm nằm nghỉ với ngã.

#### Skeleton LSTM/TCN/ST-GCN

Mạnh:

- Hiểu chuyển động theo thời gian.
- Phân biệt tốt hơn giữa ngã và hành động giống ngã.

Yếu:

- Cần dataset.
- Nếu MediaPipe mất landmarks thì kết quả giảm.

### 12.2. Cách fusion đơn giản

Giả sử:

```text
p_skeleton = xác suất ngã từ LSTM/TCN
p_yolo = xác suất ngã từ YOLO fine-tuned
p_rule = fall_score / 4
```

Mình tính:

```text
p_final = 0.5 * p_skeleton + 0.3 * p_yolo + 0.2 * p_rule
```

Nếu:

```text
p_final >= 0.7
```

thì báo FALL.

Nếu:

```text
0.4 <= p_final < 0.7
```

thì báo WARNING.

Nếu:

```text
p_final < 0.4
```

thì NORMAL.

### 12.3. Cách fusion bằng luật dễ hiểu hơn

Có thể dùng logic:

```text
Nếu skeleton model báo fall mạnh
và rule_score cũng cao
=> FALL

Nếu YOLO báo fallen_person
nhưng skeleton không thấy pha rơi
=> WARNING hoặc NORMAL

Nếu MediaPipe mất pose
nhưng YOLO báo falling_person nhiều frame liên tiếp
=> WARNING

Nếu cả YOLO và skeleton đều báo fall
=> FALL rất chắc
```

Cách này dễ giải thích với hội đồng hơn công thức phức tạp.

## 13. Lộ trình triển khai đề xuất

Nếu làm đồ án này một cách hợp lý, mình nên đi theo thứ tự sau.

### Giai đoạn 1: giữ hệ thống hiện tại làm baseline

Mục tiêu:

```text
Biết hệ thống ban đầu tốt/xấu đến đâu.
```

Việc cần làm:

```text
1. Chuẩn bị một tập video test nhỏ.
2. Chạy project hiện tại.
3. Ghi lại hệ thống dự đoán FALL/NORMAL.
4. Tính Accuracy, Precision, Recall, F1, False Alarm Rate.
```

Kết quả cần có:

| Model | Accuracy | Precision | Recall | F1 | FPS |
|---|---:|---:|---:|---:|---:|
| Rule-based baseline |  |  |  |  |  |

### Giai đoạn 2: cải thiện rule-based

Mục tiêu:

```text
Giảm báo nhầm và bỏ sót mà chưa cần train model lớn.
```

Việc cần làm:

```text
1. Thêm normalized velocity.
2. Thêm temporal voting.
3. Thêm logic sau ngã.
4. Tune threshold bằng validation set.
```

Kết quả cần có:

| Model | Accuracy | Precision | Recall | F1 | False Alarm Rate |
|---|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |
| Rule-based improved |  |  |  |  |  |

### Giai đoạn 3: tạo dataset skeleton

Mục tiêu:

```text
Có dữ liệu để train machine learning/deep learning.
```

Việc cần làm:

```text
1. Chạy MediaPipe trên toàn bộ video.
2. Lưu landmarks từng frame.
3. Cắt sliding window 30-60 frame.
4. Gán label fall/normal cho từng window.
5. Lưu thành .npy hoặc .csv.
```

Cấu trúc đề xuất:

```text
datasets/
  skeleton_windows/
    X_train.npy
    y_train.npy
    X_val.npy
    y_val.npy
    X_test.npy
    y_test.npy
```

### Giai đoạn 4: train skeleton model

Mục tiêu:

```text
Cho model học chuyển động ngã thật thay vì chỉ dùng luật.
```

Model nên thử:

```text
Random Forest -> dễ nhất
LSTM -> dễ giải thích deep learning
TCN -> thực tế tốt, train nhanh
```

Kết quả cần có:

| Model | Accuracy | Precision | Recall | F1 | Latency | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based improved |  |  |  |  |  |  |
| Random Forest skeleton |  |  |  |  |  |  |
| LSTM skeleton |  |  |  |  |  |  |
| TCN skeleton |  |  |  |  |  |  |

### Giai đoạn 5: fine-tune YOLO nếu còn thời gian

Mục tiêu:

```text
Tạo thêm một nhánh nhận diện trạng thái người từ ảnh.
```

Việc cần làm:

```text
1. Tách frame từ video.
2. Annotate bounding box.
3. Gán class normal_person/falling_person/fallen_person/sitting_person.
4. Train YOLO.
5. Test YOLO.
6. Fusion với skeleton model.
```

### Giai đoạn 6: fusion model

Mục tiêu:

```text
Kết hợp nhiều nguồn để ra quyết định chắc hơn.
```

Input:

```text
p_skeleton
p_yolo
p_rule
```

Output:

```text
NORMAL / WARNING / FALL
```

Kết quả cuối:

| Model | Accuracy | Precision | Recall | F1 | FAR | Latency | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based baseline |  |  |  |  |  |  |  |
| Rule-based improved |  |  |  |  |  |  |  |
| Skeleton LSTM/TCN |  |  |  |  |  |  |  |
| YOLO fine-tuned |  |  |  |  |  |  |  |
| Fusion model |  |  |  |  |  |  |  |

## 14. Cách tính chỉ số đánh giá

Trong bài này, mình đặt:

```text
fall = positive
normal = negative
```

### 14.1. TP, FP, TN, FN là gì?

```text
TP = True Positive
FP = False Positive
TN = True Negative
FN = False Negative
```

Dịch ra tình huống cụ thể:

```text
TP: người ngã thật, hệ thống báo ngã.
FP: người không ngã, hệ thống vẫn báo ngã.
TN: người không ngã, hệ thống không báo.
FN: người ngã thật, hệ thống không báo.
```

Bảng dễ nhìn:

| Thực tế | Hệ thống dự đoán | Kết quả |
|---|---|---|
| Ngã | Báo ngã | TP |
| Không ngã | Báo ngã | FP |
| Không ngã | Không báo | TN |
| Ngã | Không báo | FN |

### 14.2. Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Accuracy trả lời câu hỏi:

```text
Trong tất cả trường hợp, hệ thống đúng bao nhiêu phần trăm?
```

Nhưng trong fall detection, không nên chỉ nhìn Accuracy.

Ví dụ:

```text
1000 frame
990 frame normal
10 frame fall
```

Nếu model luôn đoán normal:

```text
Accuracy = 990 / 1000 = 99%
```

Nghe rất cao, nhưng thật ra model bỏ sót toàn bộ ngã. Vậy là không dùng được.

### 14.3. Precision

```text
Precision = TP / (TP + FP)
```

Precision trả lời:

```text
Trong những lần hệ thống báo ngã, có bao nhiêu lần là ngã thật?
```

Precision thấp nghĩa là báo động giả nhiều.

Ví dụ:

```text
Hệ thống báo ngã 10 lần.
Trong đó chỉ có 6 lần là ngã thật.
Precision = 6 / 10 = 0.6
```

### 14.4. Recall

```text
Recall = TP / (TP + FN)
```

Recall trả lời:

```text
Trong tất cả lần ngã thật, hệ thống phát hiện được bao nhiêu?
```

Recall rất quan trọng với fall detection.

Ví dụ:

```text
Có 10 lần ngã thật.
Hệ thống phát hiện được 8 lần.
Recall = 8 / 10 = 0.8
```

Nếu recall thấp, nghĩa là hệ thống bỏ sót nhiều cú ngã.

### 14.5. F1-score

```text
F1-score = 2 * Precision * Recall / (Precision + Recall)
```

F1-score cân bằng giữa Precision và Recall.

Nếu muốn một chỉ số tổng hợp để so sánh model, F1-score khá hợp lý.

### 14.6. Specificity

```text
Specificity = TN / (TN + FP)
```

Specificity trả lời:

```text
Trong các trường hợp không ngã, hệ thống nhận đúng bình thường được bao nhiêu?
```

Specificity cao nghĩa là ít báo nhầm với hoạt động bình thường.

### 14.7. False Alarm Rate

```text
False Alarm Rate = FP / (FP + TN)
```

False Alarm Rate là tỉ lệ báo động giả.

Chỉ số này càng thấp càng tốt.

Trong hệ thống thực tế, có thể tính thêm:

```text
Số báo động giả / giờ
```

Ví dụ:

```text
Camera chạy 2 giờ.
Có 6 lần báo nhầm.
False alarms per hour = 6 / 2 = 3 lần/giờ.
```

### 14.8. Miss Rate

```text
Miss Rate = FN / (TP + FN)
```

Miss Rate là tỉ lệ bỏ sót.

Chỉ số này càng thấp càng tốt.

### 14.9. Latency

Latency là độ trễ cảnh báo.

```text
Latency = thời điểm hệ thống báo ngã - thời điểm bắt đầu ngã
```

Ví dụ:

```text
Người bắt đầu ngã ở giây 2.0.
Hệ thống báo ở giây 2.7.
Latency = 0.7 giây.
```

Trong fall detection, latency nên thấp. Nhưng nếu giảm latency quá mạnh, hệ thống có thể báo nhầm nhiều hơn.

Đây là trade-off:

```text
Báo nhanh -> dễ báo nhầm.
Báo chắc -> có thể báo trễ.
```

### 14.10. FPS

FPS là số frame xử lý mỗi giây.

```text
FPS = số frame xử lý / số giây
```

Nếu camera 30 FPS mà hệ thống chỉ xử lý được 5 FPS thì cảnh báo sẽ chậm.

Mục tiêu thực tế:

```text
20-30 FPS: tốt cho realtime.
10-20 FPS: tạm chấp nhận.
< 10 FPS: hơi chậm.
```

## 15. Code tính chỉ số bằng Python

Giả sử:

```text
0 = normal
1 = fall
```

Code:

```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = [0, 0, 1, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 0, 0, 1, 0, 1]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0

print("TP:", tp)
print("FP:", fp)
print("TN:", tn)
print("FN:", fn)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1-score:", f1)
print("False Alarm Rate:", false_alarm_rate)
print("Miss Rate:", miss_rate)

print(classification_report(y_true, y_pred, target_names=["normal", "fall"]))
```

### 15.1. Tính latency theo video

Ví dụ mình có:

```text
fall_start_time = 2.0
alert_time = 2.7
```

Code:

```python
latency = alert_time - fall_start_time
print("Latency:", latency)
```

Nếu một video có nhiều lần ngã:

```python
latencies = [0.4, 0.7, 0.6, 1.0]
average_latency = sum(latencies) / len(latencies)
print("Average latency:", average_latency)
```

## 16. Đánh giá theo frame và theo video khác nhau thế nào?

### 16.1. Đánh giá theo frame

Mỗi frame là một mẫu.

Ví dụ:

```text
frame 1: normal
frame 2: normal
frame 3: fall
```

Ưu điểm:

- Tính metrics dễ.
- Biết model sai ở frame nào.

Nhược điểm:

- Dữ liệu bị lệch rất mạnh vì frame normal thường nhiều hơn frame fall.
- Một video dài có thể chiếm quá nhiều ảnh hưởng.

### 16.2. Đánh giá theo video/event

Mỗi cú ngã là một event.

Ví dụ:

```text
Video có 1 lần ngã.
Nếu hệ thống báo trong khoảng chấp nhận được thì tính là detect đúng.
```

Ưu điểm:

- Gần thực tế hơn.
- Hợp với hệ thống cảnh báo.

Nhược điểm:

- Cần label thời điểm bắt đầu ngã.
- Cần định nghĩa cửa sổ thời gian hợp lệ.

Ví dụ quy tắc:

```text
Nếu người bắt đầu ngã ở giây 2.0.
Nếu hệ thống báo từ giây 2.0 đến 4.0 thì tính là TP.
Nếu không báo trong khoảng đó thì FN.
Nếu video normal mà có báo FALL thì FP.
```

Trong báo cáo nên có cả:

```text
Frame-level metrics
Video-level/event-level metrics
```

Nếu chỉ chọn một, mình chọn video-level/event-level vì hợp với bài toán cảnh báo hơn.

## 17. Các lỗi thường gặp khi làm dataset và train model

### 17.1. Dataset quá sạch

Nếu dataset chỉ có cảnh ngã rõ ràng, nền trống, người mặc đồ dễ nhìn, model sẽ đẹp trên test nhưng kém ngoài đời.

Nên thêm:

- Ánh sáng kém.
- Người bị che một phần.
- Nền phòng hơi rối.
- Góc camera khác.
- Hành động dễ nhầm.

### 17.2. Train/test bị lẫn dữ liệu

Nếu cùng một video xuất hiện ở cả train và test, kết quả sẽ bị ảo.

Cần chia theo:

```text
video
subject
scene
```

Không chia ngẫu nhiên từng frame nếu frame đến từ cùng video.

### 17.3. Label không nhất quán

Ví dụ:

```text
Người đang rơi xuống được label là fall ở video này.
Nhưng ở video khác lại label là normal.
```

Model sẽ bị rối.

Nên định nghĩa label rõ:

```text
normal: hoạt động bình thường.
falling: giai đoạn đang ngã.
fallen: đã nằm sau khi ngã.
fall: có falling hoặc fallen trong window.
```

### 17.4. Dữ liệu fall quá ít

Fall thường ít hơn normal rất nhiều.

Cách xử lý:

- Thu thêm clip fall.
- Augmentation.
- Class weighting.
- Oversampling class fall.
- Chọn threshold ưu tiên recall.

### 17.5. Chỉ nhìn Accuracy

Accuracy cao chưa chắc tốt.

Với fall detection nên báo cáo thêm:

```text
Precision
Recall
F1-score
False Alarm Rate
Miss Rate
Latency
FPS
```

## 18. Data augmentation nên dùng

Data augmentation là tạo biến thể từ dữ liệu gốc để model học tốt hơn.

### 18.1. Với ảnh YOLO

Có thể dùng:

- Lật ngang ảnh.
- Thay đổi độ sáng.
- Thêm noise nhẹ.
- Crop/scale nhẹ.
- Blur nhẹ.

Không nên augmentation quá mạnh làm sai bản chất hành động.

Ví dụ:

```text
Ngã sang trái sau khi lật ngang thành ngã sang phải -> vẫn hợp lý.
Ảnh quá blur đến mức người không nhìn rõ -> không nên.
```

### 18.2. Với skeleton

Có thể dùng:

- Thêm noise nhỏ vào tọa độ.
- Scale nhẹ skeleton.
- Dịch skeleton nhẹ.
- Lật trái/phải.
- Thay đổi tốc độ chuỗi frame nhẹ.

Ví dụ:

```text
Một cú ngã 30 frame có thể resample thành 25 hoặc 35 frame.
```

Điều này giúp model bớt phụ thuộc vào tốc độ ngã cố định.

## 19. Nên viết phần báo cáo như thế nào?

Cấu trúc phần cải tiến có thể viết như này:

```text
1. Mô tả baseline hiện tại.
2. Phân tích hạn chế của baseline.
3. Đề xuất cải tiến.
4. Mô tả dataset.
5. Mô tả phương pháp train/evaluate.
6. Bảng kết quả.
7. Nhận xét.
8. Hạn chế còn lại và hướng phát triển.
```

### 19.1. Đoạn mô tả baseline

Có thể viết:

```text
Hệ thống ban đầu sử dụng YOLOv8 để phát hiện người trong khung hình, sau đó sử dụng MediaPipe Pose để trích xuất các điểm khớp chính trên cơ thể. Dựa trên các điểm khớp này, hệ thống tính toán một số đặc trưng hình học như góc nghiêng thân người, tỉ lệ chiều rộng/chiều cao của bounding box, vận tốc chuyển động theo phương dọc và vị trí tương đối giữa các khớp. Nếu nhiều đặc trưng vượt qua ngưỡng định trước trong một khoảng thời gian nhất định, hệ thống kết luận có sự kiện ngã.
```

### 19.2. Đoạn phân tích hạn chế

```text
Mặc dù phương pháp rule-based có ưu điểm là đơn giản, dễ triển khai và tốc độ xử lý nhanh, độ chính xác của hệ thống phụ thuộc nhiều vào các ngưỡng thủ công. Khi góc đặt camera, khoảng cách người đến camera hoặc điều kiện ánh sáng thay đổi, các ngưỡng này có thể không còn phù hợp. Ngoài ra, một số hành động bình thường như cúi xuống, ngồi nhanh hoặc nằm nghỉ có đặc điểm hình học tương tự hành động ngã, làm hệ thống dễ phát sinh báo động giả.
```

### 19.3. Đoạn đề xuất cải tiến bằng skeleton model

```text
Để cải thiện khả năng nhận diện, đồ án đề xuất sử dụng mô hình học máy trên dữ liệu skeleton theo thời gian. Các điểm khớp được trích xuất từ MediaPipe trong nhiều frame liên tiếp, sau đó được chuẩn hóa và đưa vào mô hình phân loại như Random Forest, LSTM hoặc TCN. Cách tiếp cận này cho phép mô hình học được đặc trưng chuyển động trước, trong và sau khi ngã, thay vì chỉ dựa vào tư thế tại một frame đơn lẻ.
```

### 19.4. Đoạn đề xuất fusion

```text
Ngoài ra, hệ thống có thể kết hợp nhiều nguồn thông tin gồm điểm đánh giá từ rule-based detector, xác suất ngã từ mô hình skeleton theo thời gian và kết quả nhận diện trạng thái người từ YOLO fine-tuned. Việc kết hợp này giúp tận dụng ưu điểm của từng phương pháp: rule-based có tốc độ nhanh và dễ giải thích, YOLO hỗ trợ nhận diện trực quan khi pose bị mất, còn skeleton model học được đặc trưng chuyển động theo thời gian.
```

### 19.5. Đoạn nói về đánh giá

```text
Do bài toán fall detection có dữ liệu mất cân bằng, trong đó số frame bình thường thường nhiều hơn số frame ngã, đồ án không chỉ sử dụng Accuracy mà còn đánh giá bằng Precision, Recall, F1-score, False Alarm Rate, Miss Rate, Latency và FPS. Trong đó Recall và Miss Rate đặc biệt quan trọng vì bỏ sót sự kiện ngã có thể gây hậu quả nghiêm trọng trong ứng dụng thực tế.
```

## 20. Kết luận nên chọn hướng nào?

Nếu muốn làm đồ án chắc, vừa sức và có chiều sâu, mình nên chọn lộ trình:

```text
1. Giữ hệ thống hiện tại làm baseline.
2. Cải thiện rule-based bằng normalized velocity và temporal voting.
3. Chuẩn bị dataset video có fall và normal.
4. Trích xuất skeleton bằng MediaPipe.
5. Train LSTM hoặc TCN trên chuỗi skeleton.
6. So sánh baseline và model cải tiến bằng Precision, Recall, F1, FAR, Latency, FPS.
7. Nếu còn thời gian, fine-tune YOLO và fusion kết quả.
```

Nếu phải chọn một hướng duy nhất để tập trung:

```text
Skeleton sequence model bằng LSTM hoặc TCN.
```

Vì hướng này giải quyết đúng bản chất bài toán:

```text
Ngã không chỉ là một tư thế.
Ngã là một chuỗi chuyển động.
```

Và đây cũng là câu chuyện rất dễ bảo vệ:

```text
Baseline dùng luật thủ công nên còn nhạy với ngưỡng.
Mô hình đề xuất dùng skeleton theo thời gian để học chuyển động ngã.
Kết quả được đánh giá bằng các chỉ số phù hợp với hệ thống cảnh báo.
```

## 21. Nguồn đọc thêm

Một số nguồn chính để đọc thêm khi viết báo cáo:

- Ultralytics YOLO training docs: https://docs.ultralytics.com/modes/train/
- Ultralytics object detection dataset format: https://docs.ultralytics.com/datasets/detect/
- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- scikit-learn classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- ST-GCN paper: https://arxiv.org/abs/1801.07455

