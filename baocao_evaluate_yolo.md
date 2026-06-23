# Báo Cáo: Đánh Giá Tác Động Khi Nâng Cấp YOLOv8n → YOLOv8m

## Kết quả đánh giá trên videoevaluate (60 video: 31 Fall / 29 No_Fall)

### So sánh TCN + YOLOv8n vs TCN + YOLOv8m

| Chỉ số | TCN + YOLOv8n | TCN + YOLOv8m | Thay đổi |
|--------|--------------|--------------|----------|
| Accuracy | 46.67% | **60.0%** | +13.33% |
| Precision | 48.65% | **57.45%** | +8.8% |
| Recall | 58.06% | **87.1%** | **+29.03%** |
| Specificity | 34.48% | 31.03% | -3.45% |
| F1-Score | 52.94% | **69.23%** | **+16.29%** |
| TP | 18/31 | **27/31** | +9 |
| FP | 19 | 20 | +1 |
| TN | 10 | 9 | -1 |
| FN | 13 | **4** | -9 |

### Kết luận

1. **YOLOv8m cải thiện đáng kể recall** (58% → 87%) - bắt được thêm 9 ca ngã mà YOLOv8n bỏ sót
2. **Precision cũng tăng** (48.65% → 57.45%) - chất lượng dự đoán Fall tốt hơn
3. Specificity giảm nhẹ (34.48% → 31.03%) - chỉ thêm 1 false alarm
4. **F1 tăng 16.29%** (52.94% → 69.23%) - cải thiện tổng thể rõ rệt
5. **Khuyến nghị sử dụng YOLOv8m** cho pipeline TCN

### So sánh với Rule-based

| Chỉ số | Rule-based | TCN + YOLOv8n | TCN + YOLOv8m |
|--------|-----------|--------------|--------------|
| Recall | 16.13% | 58.06% | **87.1%** |
| Specificity | **100%** | 34.48% | 31.03% |
| F1 | 27.78% | 52.94% | **69.23%** |

### Cấu hình đã dùng
- YOLOv8n (nano, ~6.2 MB) vs YOLOv8m (medium, ~52 MB)
- Device: CUDA
- Confidence threshold: 0.5
- Window size: 30 frames
- TCN threshold: 0.5
- Frame skip: 5 (xử lý mỗi frame thứ 5)
