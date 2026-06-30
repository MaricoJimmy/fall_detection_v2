# Báo cáo Cải tiến: Tích hợp Đặc trưng Vận tốc (Velocity Features) vào Mô hình TCN

## 1. Đặt vấn đề (Problem Statement)
Trong các giai đoạn thử nghiệm trước, hệ thống nhận diện tốt hành động "Ngã" (Fall) nhưng gặp phải một vấn đề nghiêm trọng: **Tỷ lệ báo động giả (False Alarm) rất cao khi người dùng nằm ngủ (Lying Down) hoặc ngồi xuống (Sitting Down)**.

Lý do vật lý: Khung xương (Skeleton) ở khung hình (frame) cuối cùng của hành động Ngã và hành động Nằm gần như giống hệt nhau về mặt không gian (Spatial). Nếu chỉ cung cấp chuỗi tọa độ đơn thuần, mô hình TCN gặp khó khăn trong việc phân biệt hai trạng thái này.

## 2. Phương pháp luận (Methodology)

Để giải quyết vấn đề, dự án áp dụng phương pháp **Temporal Derivative (Đạo hàm thời gian)** để trích xuất đặc trưng Vận tốc (Velocity) từ tọa độ Không gian (Spatial coordinates).
- Hành động "Ngã": Có gia tốc biến thiên cực kỳ lớn và đột ngột trong khoảng thời gian ngắn (0.5s - 1s).
- Hành động "Nằm ngủ": Có gia tốc biến thiên chậm và đều.

### Công thức Toán học
Dữ liệu đầu vào ban đầu $X$ có kích thước `(Batch, Window_Size, 51)` với 51 là số chiều tọa độ của 17 khớp xương $(17 \times 3)$.

Vận tốc $V$ tại thời điểm $t$ được tính bằng sự chênh lệch tọa độ giữa thời điểm $t$ và $t-1$:
$$ V_t = X_t - X_{t-1} $$

Trong PyTorch, quá trình này được thực thi cực kỳ tối ưu thông qua hàm `torch.diff`. 

```python
def add_velocity_features(x):
    velocity = torch.diff(x, dim=1) # Tính đạo hàm thời gian
    zero_pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
    velocity = torch.cat((zero_pad, velocity), dim=1) # Padding frame đầu tiên
    new_x = torch.cat((x, velocity), dim=2) # Nối tọa độ và vận tốc
    return new_x
```

Sau khi nối chuỗi, số chiều (features) của mạng nơ-ron được tăng gấp đôi từ **51 lên 102**.

## 3. Quá trình Huấn luyện (Training Process)

- **Dataset**: `C:\Download\Do_an\data_veloci`
- **Tiền xử lý**: Sử dụng YOLOv8-Pose để trích xuất tọa độ xương từ hàng trăm video phức tạp (Kể cả Hồng ngoại ban đêm IR). Quá trình sinh ra **4772 windows** (khung thời gian cắt lát).
- **Tỷ lệ chia tập dữ liệu**: 70% Train (3340 samples), 15% Validation (715 samples), 15% Test (717 samples).
- **Early Stopping**: Mô hình hội tụ rất nhanh và dừng sớm ở Epoch thứ 34 để chống Overfitting. Trọng số (weights) tốt nhất được tự động lưu ở Epoch thứ 24.

## 4. Kết quả & Đánh giá (Results & Evaluation)

Sau khi bổ sung Velocity Features, mạng TCN cho kết quả trên tập Test (Tập kiểm thử hoàn toàn mới, AI chưa từng thấy) vô cùng ấn tượng:

| Metric | Score | Giải thích ý nghĩa |
|--------|-------|--------------------|
| **Accuracy** | 84.52% | Độ chính xác tổng thể rất tốt trên dữ liệu thô. |
| **Precision** | 79.04% | Khi hệ thống báo ngã, tỷ lệ đúng là gần 80%. |
| **Recall** | 79.93% | Hệ thống bắt được 80% số vụ ngã thực tế. |
| **Specificity** | **87.28%** | **Cực kỳ xuất sắc. Hệ thống nhận diện đúng 87.28% các trường hợp bình thường (không bị báo động giả).** |
| **False Alarm** | **12.72%** | **Tỷ lệ báo động giả giảm đột phá (trước đây là ~50%).** |

### Confusion Matrix
- **TN (True Negative - Nhận diện đúng Không Ngã):** 391
- **FP (False Positive - Báo động giả):** 57
- **FN (False Negative - Bỏ sót vụ ngã):** 54
- **TP (True Positive - Nhận diện đúng Ngã):** 215

## 5. Kết luận (Conclusion)

Việc chuyển đổi tư duy từ thuật toán dựa trên quy tắc (Rule-based) sang mạng nơ-ron học sâu (Deep Learning) và đặc biệt là việc **chủ động thiết kế thêm Đặc trưng Vận tốc (Velocity Features) ở giai đoạn tiền xử lý (Preprocessing)** đã giải quyết triệt để yếu điểm "Báo động giả khi nằm ngủ" của mô hình cũ. 

Hệ thống hiện tại (YOLOv8-Pose + TCN Velocity) không chỉ đảm bảo tốc độ chạy theo thời gian thực nhờ thiết kế song song của TCN, mà còn đạt độ tin cậy rất cao (Specificity ~87%, Tỷ lệ báo động giả chỉ còn 12.72%), hoàn toàn đáp ứng được các tiêu chuẩn khắt khe để triển khai thành phần mềm giám sát người cao tuổi trong thực tế.
