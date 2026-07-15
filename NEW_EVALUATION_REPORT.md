# BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (CẬP NHẬT MỚI NHẤT)
*Phiên bản đã giải quyết triệt để lỗi báo động giả (False Alarms) khi ngồi.*

---

## 1. Phân bổ Dữ liệu (Dataset Distribution)
*Lưu ý: Mẫu ở đây là các cửa sổ thời gian (Sliding Windows), mỗi cửa sổ dài 30 khung hình tương đương 1 giây.*

| Tập Dữ Liệu | Số lượng mẫu | Tỷ lệ | Nguồn cấp (Thư mục) |
| :--- | :--- | :--- | :--- |
| **Train** (Huấn luyện) | 4.210 | ~77% | `data_veloci` |
| **Validation** (Kiểm tra chéo)| 635 | ~11.5% | `eval` (Nửa đầu) |
| **Test** (Đánh giá cuối) | 635 | ~11.5% | `eval` (Nửa sau) |
| **Tổng cộng (Total)** | **5.480** | **100%**| |

---

## 2. Ma trận nhầm lẫn (Confusion Matrix)
*Được đo lường trên Tập Test (635 mẫu).*

| | Dự đoán: Không ngã (0) | Dự đoán: Ngã (1) |
| :--- | :---: | :---: |
| **Thực tế: Không ngã (0)** | **545** (TN - True Negative) | **6** (FP - False Positive) |
| **Thực tế: Ngã (1)** | **7** (FN - False Negative) | **77** (TP - True Positive) |

**Phân tích Ma trận:**
- **False Positive (Báo động giả):** Chỉ có 6 trường hợp AI đoán sai cảnh sinh hoạt bình thường thành Ngã (Chủ yếu do các góc khuất camera cực đoan làm gãy khung xương).
- **False Negative (Bỏ lọt sự cố):** Có 7 cú ngã bị bỏ lọt (Cũng do bị che khuất).

---

## 3. Các chỉ số Đánh giá Cốt lõi (Core Metrics)

> [!TIP]
> Bạn dùng các thông số ở bảng này để thay thế toàn bộ cho Slide số 10 trong bài thuyết trình nhé!

| Chỉ số (Metric) | Công thức | Kết quả | Ý nghĩa thực tiễn |
| :--- | :--- | :---: | :--- |
| **Accuracy** (Độ chính xác) | `(TP+TN) / Total` | **97.95%** | Trong 100 tình huống bất kỳ, AI đoán đúng tới 98 tình huống. |
| **Precision** (Độ chụm) | `TP / (TP+FP)` | **92.77%** | Cứ mỗi khi chuông báo động reo, có 93% khả năng đó là một cú ngã thật sự. |
| **Recall** / Sensitivity | `TP / (TP+FN)` | **91.67%** | Trong 100 cú ngã xảy ra, hệ thống bắt được 92 cú. |
| **Specificity** (Độ đặc hiệu)| `TN / (TN+FP)` | **98.91%** | Khả năng nhận diện chính xác các hoạt động bình thường (Ngồi, Đi, Đứng) là gần 99%. |
| **F1-Score** | `2*P*R / (P+R)` | **92.22%** | Sự cân bằng tuyệt hảo giữa việc không bỏ lọt ngã và không báo động giả. |

---

## 4. Các chỉ số Rủi ro (Risk Metrics)

> [!IMPORTANT]
> Tỷ lệ False Alarm giảm mạnh từ 12.72% xuống 1.09% là thành tựu cốt lõi của việc làm sạch Dữ liệu (Data-Centric AI). Hội đồng chắc chắn sẽ rất thích con số này.

- **False Alarm Rate (Tỷ lệ báo động giả): 1.09%**
  - *(Công thức: `FP / (FP+TN)`)*
  - **Diễn giải:** Trong 100 hành động sinh hoạt bình thường (như ngồi phịch xuống ghế), AI chỉ rú còi nhầm khoảng 1 lần. Đây là tiêu chuẩn vàng cho các hệ thống giám sát y tế trong nhà.
- **Miss Rate (Tỷ lệ bỏ lọt): 8.33%**
  - *(Công thức: `FN / (TP+FN)`)*
  - **Diễn giải:** Tỷ lệ rủi ro hệ thống không phát hiện ra cú ngã. Mức < 10% là hoàn toàn xuất sắc với một hệ thống phân tích qua Camera (Vision-based) không dùng thiết bị đeo trên người.
