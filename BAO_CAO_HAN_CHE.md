# BAO CAO: HAN CHE VA HUONG CAI THIEN HE THONG PHAT HIEN NGA

## Tong quan

**Du an:** He thong phat hien nga su dung YOLO + MediaPipe + TCN  
**File phan tich:** `test_video.py`  
**Ngay phan tich:** 12/06/2026

---

## 1. Han che cua MediaPipe trong Pipeline Crop

### Mo ta van de

Hien tai, he thong su dung pipeline 2 buoc:
1. YOLO phat hien nguoi → tra ve bounding box
2. Crop vung nguoi theo bbox → chay MediaPipe Pose tren vung crop

### Tai sao day la han che

**MediaPipe Pose duoc thiet ke de hoat dong tren toan bo frame**, nhin thay day du co the. Khi crop chi mot phan:

| Tinh huong | Ket qua |
|------------|---------|
| Crop cat mat chan | MediaPipe khong detect duoc khop co chan, chan |
| Crop cat mat tay | Khop co tay, ban tay bi thieu |
| Nguoi o sat rai frame | Bounding box khong day du, skeleton bi dut doan |
| Nguoi nam ngang | MediaPipe co the khong nhan ra duoc dang nguoi |

### Anh huong den do chinh xac

- **Skeleton khong day du:** Mot so keypoints bi thieu hoac co confidence thap (< 0.3)
- **Du lieu dau vao model bi nhieu:** TCN model nhan input 17 keypoints x 3 (x, y, visibility). Neu nhieu keypoints bi thieu, model co the du doan sai
- **Khong on dinh giua cac frame:** Skeleton co the "nhay" lung tung giua cac frame vi MediaPipe xu ly tung frame doc lap

### Minh hoa

```
Frame goc (640x480):          Crop theo YOLO bbox:
+------------------+          +----------+
|                  |          |  Person  |
|   Person         |   -->   |  (chi    |
|                  |          |  thay    |
|                  |          |  nua     |
+------------------+          |  nguoi)  |
                              +----------+
MediaPipe nhin thay           MediaPipe chi thay
toan bo co the                mot phan co the
→ Skeleton day du             → Skeleton thieu khop
```

---

## 2. Han che cua Thuat toan Tracking (IoU Matching)

### Mo ta van de

Hien tai su dung **IoU matching don gian** de theo doi nguoi qua cac frame:
- Tinh IoU giua bounding box frame truoc va frame hien tai
- Neu IoU >= 0.3 → cung mot nguoi (giu nguyen ID)
- Neu IoU < 0.3 → nguoi moi (gan ID moi)

### Cac tinh huong tracking bi loi

| Tinh huong | Nguyen nhan | Hau qua |
|------------|-------------|---------|
| Nguoi di chuyen nhanh | Bounding box frame truoc va hien tai khong chong len nhieu (IoU < 0.3) | Gan ID moi, mat theo doi |
| Nguoi bi che khuat tam | Bounding box bi mat 1-2 frame | Khi xuat hien lai, gan ID moi |
| 2 nguoi di sat nhau | Bounding box chong len nhieu | Co the swap ID (nguoi A thanh B va nguoc lai) |
| Camera chuyen dong | Tat ca bounding box thay doi vi tri | Tracking fail hoan toan |
| Nguoi roi frame roi quay lai | ID cu da bi xoa | Gan ID moi, mat du lieu buffer |

### Vi du cu the

```
Frame 1: Nguoi A (ID:0) o vi tri (100, 100, 200, 300)
Frame 2: Nguoi A di nhanh den vi tri (300, 100, 400, 300)
IoU giua 2 bbox = 0 (khong chong len)
→ He thong gan ID:1 cho nguoi A
→ Buffer cua ID:0 bi mat, phai thu thap lai 30 frame moi
```

### Anh huong den phat hien nga

- **Mat du lieu buffer:** Khi ID bi nhay, buffer skeleton bi reset, phai thu thap lai 30 frame moi co du du lieu de predict
- **Cham phat hien:** Neu nguoi dang nga ma bi mat tracking, he thong phai cho them 30 frame (1 giay) moi co the phat hien
- **Bao sai:** Neu 2 nguoi swap ID, buffer cua nguoi A co the chua du lieu cua nguoi B → du doan sai

---

## 3. Han che cua Co che Canh bao FALL

### Mo da thuc hien

Them timer 60 frame (2 giay) de kiem soat thoi gian hien thi "WARNING: FALL DETECTED!"

### Van con han che

#### 3.1. Status van la "FALL" sau khi timer het

```python
if prob >= CONFIG['threshold']:
    self.statuses[person_id] = "FALL"
    self.fall_alert_timers[person_id] = 60
```

**Van de:**
- Timer chi kiem soat **thoi gian hien thi canh bao**
- Status "FALL" van duoc giu nguyen
- Bounding box van mau do
- Panel van hien thi "FALL"

**Hau qua:** Nguoi dung co the hieu nham la he thong van dang phat hien nga, thuc te chi la status cu.

#### 3.2. Khong tu dong reset khi nguoi dung day

**Kich ban:**
1. Nguoi nga → he thong bao FALL
2. Nguoi dung day, di lai binh thuong
3. Nhung buffer van chua nhieu frame "nga"
4. Model van predict probability cao
5. Status van la "FALL"

**Nguyen nhan:** Buffer luu 30 frame gan nhat, khong phan biet frame nao la "nga that", frame nao la "da dung day".

#### 3.3. Video nhieu canh phan doan

**Kich ban:**
1. Canh 1: Nguoi nga → bao FALL
2. Canh 2: Chuyen sang camera khac, nguoi dung binh thuong
3. Nhung ID tracking co the bi nhay (nguoi moi)
4. Buffer bi reset, nhung neu model van predict FALL (do du lieu ngau nhien)
5. Bao FALL sai

**Nguyen nhan:** Tracking khong du tot de xu ly video nhieu canh.

---

## 4. Han che ve Performance

### 4.1. MediaPipe xu ly tuan tu

```python
for person_id, det_idx in matched.items():
    # ...
    skeleton = self.extract_skeleton(frame, bbox)
    # MediaPipe chay tuan tu cho tung nguoi
```

**Van de:**
- Neu co 3 nguoi trong frame, MediaPipe phai chay 3 lan
- Moi lan chay MediaPipe mat ~10-20ms
- Tong thoi gian: 30-60ms chi cho MediaPipe
- Cong them YOLO (~20-30ms), TCN (~5-10ms)
- **Tong: ~60-100ms/frame → ~10-15 FPS**

**Anh huong:**
- Video 30 FPS → he thong chi xu ly duoc 10-15 FPS
- Cac frame bi bo qua → co the bo sot su kien nga
- Do tre cao → canh bao cham

### 4.2. YOLO predict toan bo frame

```python
results = self.yolo(frame, conf=0.5, classes=[0], verbose=False)
```

**Van de:**
- YOLO predict tren toan bo frame (640x480 hoac 1920x1080)
- Du chi can phat hien nguoi, khong can phat hien vat khac
- Ton thoi gian tinh toan khong can thiet

### 4.3. Buffer khong duoc toi uu

```python
self.skeleton_buffers[person_id] = deque(maxlen=CONFIG['window_size'])
```

**Van de:**
- Moi nguoi co 1 buffer rieng
- Neu co 10 nguoi trong frame → 10 buffer × 30 frames × 17 keypoints × 3 values
- **Tong: 10 × 30 × 17 × 3 × 4 bytes = ~61 KB**
- Khong nhieu, nhung neu chay lien tuc nhieu gio → memory leak neu khong xoa buffer nguoi da roi frame

---

## 5. Han che ve Do chinh xac

### 5.1. Phu thuoc vao Chat luong Video

| Chat luong video | Anh huong |
|------------------|-----------|
| Do phan giai thap (< 480p) | YOLO va MediaPipe khong detect duoc nguoi o xa |
| Anh sang yeu | YOLO confidence thap, MediaPipe fail |
| Nguoi bi che khuat | YOLO khong detect, MediaPipe khong co input |
| Goc quay tu tren cao | MediaPipe khong nhan ra dang nguoi |
| Camera rung lac | Bounding box khong on dinh, skeleton nhay |

### 5.2. Nguong phat hien (Threshold) co dinh

```python
CONFIG = {
    'threshold': 0.5,  # Co dinh cho tat ca truong hop
}
```

**Van de:**
- Threshold 0.5 co the qua thap cho mot so truong hop (bao sai nhieu)
- Hoac qua cao cho truong hop khac (bo sot nhieu)
- Khong tu dong dieu chinh theo chat luong video, khoang cach camera, goc quay

### 5.3. Khong phan biet duoc loai "nga"

**Hien tai:** Chi co 2 trang thai: NORMAL va FALL

**Thuc te:** Co nhieu truong hop de nham voi nga:
- Ngoi xuong ghe nhanh
- Cuoi xuong nhay vat
- Nam xuong giuong
- Chay nhanh roi dung lai
- Nhay mua

**Van de:** Model TCN duoc train tren tap du lieu co han, co the khong phan biet duoc cac truong hop tren.

---

## 6. Han che ve Kien truc He thong

### 6.1. Khong co co che Xac nhan Sau khi Nga (Post-fall Confirmation)

**Ly thuyet:** Sau khi nga, nguoi thuong:
- Nam im tren san
- Khong dung day ngay lap tuc
- Co the co dau hieu dau don

**Hien tai:**
- Chi can 1 frame predict FALL → bao canh
- Khong kiem tra xem nguoi co van nam im sau khi nga khong
- Co the bao sai khi nguoi chi cuoi xuong roi dung day ngay

### 6.2. Khong co co che Loc Nhieu (Noise Filtering)

**Hien tai:**
- Moi frame duoc xu ly doc lap
- Khong co logic de loc bo cac du doan bat thuong
- Vi du: Frame 1 = NORMAL, Frame 2 = FALL, Frame 3 = NORMAL → van bao FALL o frame 2

**Cai thien:** Co the dung **temporal voting** (xet 5 frame gan nhat, neu co >= 3 frame FALL thi moi bao).

### 6.3. Khong co Logging va Thong ke

**Hien tai:**
- Chi in ket qua ra console
- Khong luu log chi tiet (thoi gian phat hien, do chinh xac, so lan bao sai)
- Khong co dashboard de theo doi

**Hau qua:**
- Kho danh gia do chinh xac thuc te
- Khong biet he thong bao sai bao nhieu lan
- Khong debug duoc khi co van de

---

## 7. Huong Cai thien

### 7.1. Cai thien Pipeline MediaPipe

| Phuong an | Uu diem | Nhoc diem |
|-----------|---------|-----------|
| **Chay MediaPipe tren full frame** | Skeleton day du hon, on dinh hon | Cham hon, co the bi nhieu khi co nhieu nguoi |
| **Dung MediaPipe Holistic** | Co them tay va mat, chi tiet hon | Cham hon, nang hon |
| **Dung OpenPose** | Chinh xac hon, ho tro nhieu nguoi | Can GPU manh, kho cai dat |
| **Dung YOLO-Pose** | 1 model duy nhat, nhanh hon | Can train lai, do chinh xac co the thap hon |

### 7.2. Cai thien Tracking

| Phuong an | Uu diem | Nhoc diem |
|-----------|---------|-----------|
| **DeepSORT** | Tracking on dinh, co re-ID | Cham hon, can them model |
| **ByteTrack** | Nhanh, chinh xac | Phuc tap hon |
| **StrongSORT** | Tot nhat hien tai | Rat cham, can GPU manh |

### 7.3. Cai thien Logic Phat hien

| Phuong an | Mo ta |
|-----------|-------|
| **Temporal voting** | Xet N frame gan nhat, chi bao FALL neu >= M frame la FALL |
| **Post-fall confirmation** | Kiem tra nguoi co van nam im sau khi nga khong |
| **Recovery detection** | Tu dong reset status khi nguoi dung day |
| **Dynamic threshold** | Tu dong dieu chinh threshold theo chat luong video |

### 7.4. Cai thien Performance

| Phuong an | Mo ta |
|-----------|-------|
| **Xu ly song song** | Chay MediaPipe cho nhieu nguoi cung luc (multi-threading) |
| **Skip frames** | Chi xu ly moi 2-3 frame, giua nguyen ket qua cho cac frame bo qua |
| **Model nho hon** | Dung YOLOv8n thay vi YOLOv8s, MediaPipe Lite thay vi full |
| **TensorRT** | Toi uu model cho GPU NVIDIA |

### 7.5. Them Tinh nang

| Tinh nang | Mo ta |
|-----------|-------|
| **Ghi log chi tiet** | Luu thoi gian phat hien, hinh anh, video clip |
| **Dashboard web** | Hien thi trang thai real-time, thong ke |
| **Canh bao qua email/SMS** | Gui thong bao khi phat hien nga |
| **Phan biet loai nga** | Nga truot chan, nga vi chong mat, nga vi dot quy |
| **Uoc luong muc do nghiem trong** | Dua vao thoi gian nam im, goc nghieng |

---

## 8. Ket luan

### Diem manh cua he thong hien tai

1. **Don gian, de hieu:** Pipeline ro rang, code de doc
2. **Phu hop do an/luan van:** Du phuc tap de trinh bay, bao ve
3. **Hoat dong duoc:** Co the phat hien nga trong video test
4. **Ho tro da nguoi:** Co the theo doi nhieu nguoi cung luc

### Diem can cai thien

1. **Do chinh xac chua cao:** Do han che cua MediaPipe trong pipeline crop
2. **Tracking khong on dinh:** De mat ID khi nguoi di chuyen nhanh
3. **Performance thap:** Chi xu ly duoc 10-15 FPS
4. **Thieu tinh nang:** Khong co post-fall confirmation, temporal voting
5. **Chua co logging:** Kho danh gia va debug

### Khuyen nghi

**Cho muc dich do an/luan van:**
- Giu nguyen hien tai, tap trung vao viec trinh bay ro rang
- Ghi nhan cac han che trong bao cao
- De xuat huong cai thien trong phan "Huong phat trien"

**Cho muc dich ung dung thuc te:**
- Can nang cap tracking (DeepSORT/ByteTrack)
- Them temporal voting va post-fall confirmation
- Toi uu performance (TensorRT, multi-threading)
- Them logging va dashboard
- Train lai model voi nhieu du lieu hon

---

## Tham khao

- MediaPipe Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- YOLOv8: https://docs.ultralytics.com/
- DeepSORT: https://github.com/nwojke/deep_sort
- ByteTrack: https://github.com/ifzhang/ByteTrack
