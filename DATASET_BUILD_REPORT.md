# Báo cáo chi tiết: Xây dựng Skeleton Dataset (Cập nhật phiên bản TCN_Veloci)

**Đồ án:** AI-Based Human Fall Detection System  
**Ngày cập nhật:** 2026-07-15  

---

## 1. Nguồn dữ liệu (Data Source)

Dữ liệu được chia thành hai bộ video riêng biệt để Huấn luyện và Đánh giá:

**Thư mục 1: Tập Huấn luyện (Train)**
- **Đường dẫn:** `C:\Download\Do_an\data_veloci`
- **Fall:** 103 videos
- **No_Fall:** 132 videos

**Thư mục 2: Tập Đánh giá (Validation & Test)**
- **Đường dẫn:** `C:\Download\Do_an\eval`
- **Fall:** 14 videos
- **No_Fall:** 21 videos

*Các video No_Fall đã được bổ sung thêm nhiều hành động gây nhiễu (ngồi xuống nhanh, cúi nhặt đồ) để đóng vai trò làm Hard Negatives, giúp mô hình luyện được khả năng chống báo động giả.*

---

## 2. Format dữ liệu và Trích xuất Đặc trưng (Feature Extraction)

Quá trình trích xuất hiện tại không dùng các file CSV cũ nữa mà chạy trực tiếp mô hình AI **YOLOv8-Pose (`yolov8m-pose.pt`)** để bóc tách tọa độ 17 điểm khớp xương từ video MP4.

**Số lượng Đặc trưng (Features) mỗi Frame: 102 chiều**
- **51 chiều Spatial (Không gian):** 17 điểm khớp × 3 giá trị (X_norm, Y_norm, Độ tin cậy Confidence).
- **51 chiều Velocity (Vận tốc):** Đạo hàm bậc 1 của tọa độ theo thời gian (Tính bằng cách lấy Frame hiện tại trừ đi Frame trước đó).

**Chuẩn hóa Không gian (Spatial Normalization):**
- **Dịch chuyển gốc:** Lấy tâm hông (Hip Center) làm gốc tọa độ.
- **Co giãn tỉ lệ:** Chia toàn bộ tọa độ cho chiều dài thân người (khoảng cách từ tâm Vai đến tâm Hông).
- **Mục đích:** Triệt tiêu hoàn toàn sự sai khác do người đứng gần hay đứng xa camera.

---

## 3. Kỹ thuật Cửa sổ trượt (Sliding Windows)

Vì ngã là một "quá trình", nên dữ liệu không thể đưa vào từng Frame lẻ tẻ mà phải đưa vào dưới dạng một chuỗi thời gian (Window).

**Tham số:**
- **Window Size:** 30 frames (Tương đương 1 giây thao tác với video 30 FPS).
- **Stride:** 10 frames (Bước nhảy 1/3 giây). Nếu video dài, cứ trượt đi 10 frame sẽ cắt ra 1 window mới.
- **Min Confidence:** 0.3 (Những điểm bị che khuất sẽ bị gán tọa độ về 0).

**Làm phẳng dữ liệu:**
Quá trình xử lý sẽ sinh ra các mảng dữ liệu có shape là `(30, 51)`. Sau đó trong lúc Training, module Vận tốc sẽ tự động tính toán và gắn thêm 51 chiều vận tốc, biến đổi kích thước mạng đầu vào thành `(30, 102)`.

---

## 4. Kết quả thống kê (Dataset Statistics)

| Tập dữ liệu | Số lượng Mẫu (Windows) | Tỷ lệ | Nguồn cấp |
| :--- | :--- | :--- | :--- |
| **Train** | 4.210 | 76.8% | `data_veloci` |
| **Validation** | 635 | 11.6% | `eval` |
| **Test** | 635 | 11.6% | `eval` |
| **Tổng cộng** | **5.480** | **100%** | |

*(Quá trình trích xuất, làm sạch, và chia tệp được thực hiện hoàn toàn tự động bằng script `build_dataset_split.py`)*

---

## 5. File Output

Dữ liệu sau khi xử lý được lưu dưới dạng file mảng Numpy (`.npy`) cực kỳ gọn nhẹ tại thư mục `datasets/skeleton_windows_veloci/`:

- `X_train.npy` (Dữ liệu Train)
- `y_train.npy` (Nhãn Train)
- `X_val.npy` (Dữ liệu Validation)
- `y_val.npy` (Nhãn Validation)
- `X_test.npy` (Dữ liệu Test)
- `y_test.npy` (Nhãn Test)

**Quy ước Nhãn (Labels):**
- `0`: Normal (Các hành động sinh hoạt bình thường).
- `1`: Fall (Hành động ngã/rơi tự do).

---

## 6. Ghi chú và Ý nghĩa Thực tiễn

1. **Bộ lọc rác:** Các video quá ngắn hoặc camera bị mờ khiến YOLOv8 không bắt được khung xương đều bị loại bỏ tự động để đảm bảo dữ liệu đưa cho TCN học là tinh khiết nhất.
2. **Khắc nghiệt hóa Đánh giá:** Tập Test cố tình được thiết kế bất đối xứng (Nhiều video No_Fall hơn Fall). Việc này ép mô hình phải thể hiện bản lĩnh phát hiện báo động giả trong một môi trường mà đa số thời gian con người sinh hoạt bình thường.
3. **Tuyệt đối không rò rỉ dữ liệu (No Data Leakage):** Việc lấy riêng rẽ 2 thư mục `data_veloci` và `eval` để chia Train/Test từ gốc rễ video giúp triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu (trường hợp các window từ cùng một video vừa nằm trong Train vừa lọt vào Test). Mô hình lúc thi cuối kỳ là hoàn toàn mù tịt, chưa từng nhìn thấy video đó bao giờ.
