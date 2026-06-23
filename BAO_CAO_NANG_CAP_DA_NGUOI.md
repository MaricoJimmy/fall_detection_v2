# BAO CAO CHI TIET: NANG CAP PHAT HIEN NGA NHIEU NGUOI

## Tong quan

**File thay doi:** `test_video.py`  
**Muc tieu:** Nang cap he thong tu phat hien nga 1 nguoi sang ho tro phat hien nga nhieu nguoi cung luc trong video  
**Ngay thuc hien:** 12/06/2026

---

## Van de truoc khi sua

Truoc khi nang cap, ham `process_frame` chi xu ly **1 nguoi duy nhat** trong moi frame:

```python
if detections:
    bbox = detections[0]  # Chi lay detection dau tien
```

Mac du YOLO co the phat hien nhieu nguoi (tra ve nhieu bounding box), he thong chi lay `detections[0]` va bo qua tat ca nhung nguoi con lai. Dieu nay co nghia:

- Chi co 1 nguoi duoc theo doi va phan tich
- Neu co 2 nguoi trong frame, nguoi thu 2 bi bo qua hoan toan
- Buffer skeleton chi luu du lieu cua 1 nguoi
- Ket qua chi hien thi trang thai cua 1 nguoi

---

## Cac thay doi chi tiet

### 1. Thay doi cau truc du lieu trong `__init__`

**Truoc (don nguoi):**

```python
self.skeleton_buffer = deque(maxlen=CONFIG['window_size'])
self.fall_probability = 0.0
self.status = "NORMAL"
```

**Sau (da nguoi):**

```python
self.skeleton_buffers = {}       # Dict: {person_id: deque}
self.fall_probabilities = {}     # Dict: {person_id: float}
self.statuses = {}               # Dict: {person_id: "NORMAL"/"FALL"}
self.next_id = 0                 # ID tiep theo cho nguoi moi
self.tracked_bboxes = {}         # Dict: {person_id: [x1,y1,x2,y2]}
self.iou_threshold = 0.3         # Nguong IoU de match nguoi giua cac frame
```

**Ly do:** Moi nguoi can co buffer skeleton rieng, ket qua rieng va duoc theo doi bang mot ID duy nhat. Su dung dictionary thay vi bien don cho phep luu tru thong tin cho N nguoi cung luc.

---

### 2. Them ham `compute_iou` (dong 116-125)

Ham tinh **Intersection over Union (IoU)** giua 2 bounding box:

```python
def compute_iou(self, box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0
```

**Cong thuc:** `IoU = Dien tich giao / Dien tich hop`

- IoU = 1.0: Hai box trung khit hoan toan
- IoU = 0.0: Hai box khong chong len nhau
- Nguong 0.3 duoc chon de can bang giua do nhay va do chinh xac

---

### 3. Them ham `track_persons` (dong 127-151)

Ham theo doi nguoi qua cac frame bang phuong phap **IoU matching**:

```python
def track_persons(self, detections):
    matched = {}
    new_tracked = {}
    used_det = set()
    
    # Buoc 1: Match nguoi cu voi detection moi
    for tid, tbox in self.tracked_bboxes.items():
        best_iou = 0
        best_idx = -1
        for i, det in enumerate(detections):
            if i in used_det:
                continue
            iou = self.compute_iou(tbox, det[:4])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= self.iou_threshold and best_idx >= 0:
            matched[tid] = best_idx
            new_tracked[tid] = detections[best_idx][:4]
            used_det.add(best_idx)
    
    # Buoc 2: Gan ID moi cho nguoi chua duoc match
    for i, det in enumerate(detections):
        if i not in used_det:
            new_tracked[self.next_id] = det[:4]
            matched[self.next_id] = i
            self.next_id += 1
    
    self.tracked_bboxes = new_tracked
    return matched
```

**Thuat toan hoat dong nhu sau:**

1. **Buoc 1 - Match nguoi cu:** Duyet qua tat ca nguoi dang duoc theo doi. Voi moi nguoi, tim detection moi co IoU cao nhat. Neu IoU >= 0.3, giu nguyen ID cu.
2. **Buoc 2 - Nguoi moi:** Cac detection chua duoc match voi nguoi cu se duoc gan ID moi tang dan.
3. **Nguoi roi khoi frame:** Neu mot ID cu khong match duoc voi bat ky detection nao, no se tu dong bi loai bo khoi `tracked_bboxes`.

**Tra ve:** Dictionary `{person_id: detection_index}` de biet moi ID tuong ung voi detection nao.

---

### 4. Sua ham `predict` (dong 246-259)

**Truoc:**

```python
def predict(self):
    if len(self.skeleton_buffer) < CONFIG['window_size']:
        ...
```

**Sau:**

```python
def predict(self, buffer):
    if len(buffer) < CONFIG['window_size']:
        ...
```

**Ly do:** Ham nhan buffer lam tham so thay vi dung `self.skeleton_buffer`, cho phep goi predict voi buffer cua tung nguoi khac nhau.

---

### 5. Sua ham `process_frame` (dong 261-307) - Thay doi lon nhat

**Truoc (chi xu ly 1 nguoi):**

```python
if detections:
    bbox = detections[0]
    # ... xu ly 1 nguoi
```

**Sau (xu ly tat ca nguoi):**

```python
if detections:
    matched = self.track_persons(detections)
    
    for person_id, det_idx in matched.items():
        bbox = detections[det_idx]
        
        # Khoi tao buffer cho nguoi moi
        if person_id not in self.skeleton_buffers:
            self.skeleton_buffers[person_id] = deque(maxlen=CONFIG['window_size'])
            self.fall_probabilities[person_id] = 0.0
            self.statuses[person_id] = "NORMAL"
        
        buffer = self.skeleton_buffers[person_id]
        
        # Ve bounding box theo trang thai
        status = self.statuses.get(person_id, "NORMAL")
        box_color = (0, 0, 255) if status == "FALL" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"ID:{person_id} {bbox[4]:.2f}", ...)
        
        # Trich xuat skeleton, chuan hoa, du doan cho tung nguoi
        skeleton = self.extract_skeleton(frame, bbox)
        if skeleton is not None:
            skeleton_norm = self.normalize_skeleton(skeleton)
            if skeleton_norm is not None:
                buffer.append(skeleton_norm)
                if len(buffer) >= CONFIG['window_size']:
                    prob = self.predict(buffer)
                    self.fall_probabilities[person_id] = prob
                    if prob >= CONFIG['threshold']:
                        self.statuses[person_id] = "FALL"
                    else:
                        self.statuses[person_id] = "NORMAL"
                self.draw_skeleton(frame, skeleton, x1, y1)
```

**Diem khac biet chinh:**

| Truoc | Sau |
|--------|-----|
| Chi lay `detections[0]` | Lap qua tat ca detections |
| 1 buffer chung | Moi nguoi 1 buffer rieng |
| 1 status chung | Moi nguoi 1 status rieng |
| Khong co tracking | Tracking bang IoU matching |
| Bounding box luon mau xanh | Mau xanh = NORMAL, mau do = FALL |
| Hien thi "Person" | Hien thi "ID:0", "ID:1", ... |

---

### 6. Sua ham `draw_results` (dong 334-382)

**Truoc:** Panel co kich thuoc co dinh 350x180, hien thi 1 status va 1 probability.

**Sau:** Panel co kich thuoc dong theo so nguoi, hien thi ket qua tung nguoi:

```python
num_persons = len(self.statuses)
panel_h = 70 + num_persons * 35   # Cao them theo so nguoi
panel_w = 380

# Hien thi tong so nguoi
cv2.putText(frame, f"Persons: {num_persons}", ...)

# Hien thi tung nguoi
for pid in sorted(self.statuses.keys()):
    text = f"ID:{pid} {status} {prob*100:.1f}% [{buf_size}/{CONFIG['window_size']}]"
```

**Hien thi tren man hinh:**

```
+--------------------------------------+
| FALL DETECTION (TCN)                 |
| Persons: 2                           |
| ID:0 NORMAL 12.3% [30/30]           |
| ID:1 FALL 87.5% [30/30]             |
| 14:30:25                             |
+--------------------------------------+
```

**Canh bao FALL:** Chi can **bat ky nguoi nao** bi FALL la hien vien do va canh bao.

---

### 7. Sua progress print trong `test_video` (dong 449-452)

**Truoc:**

```
Progress: 50.0% (150/300) - Status: NORMAL - Prob: 12.3%
```

**Sau:**

```
Progress: 50.0% (150/300) - ID:0=NORMAL, ID:1=FALL
```

### 8. Sua ket qua cuoi (dong 484-485)

**Truoc:**

```
Final status: NORMAL
Final probability: 12.3%
```

**Sau:**

```
ID:0 - Status: NORMAL - Prob: 12.3%
ID:1 - Status: FALL - Prob: 87.5%
```

---

## So do luong xu ly

```
Frame moi
    |
    v
YOLO detect_person() --> [det_0, det_1, det_2, ...]
    |
    v
track_persons() --> {ID:0 -> det_0, ID:1 -> det_1, ...}
    |
    v
Vong lap qua moi nguoi:
    |
    +---> ID:0 --> extract_skeleton --> normalize --> buffer[0] --> predict --> status[0]
    |
    +---> ID:1 --> extract_skeleton --> normalize --> buffer[1] --> predict --> status[1]
    |
    +---> ...
    |
    v
draw_results() --> Hien thi ket qua tat ca nguoi
```

---

## Han che va huong phat trien

| Han che | Mo ta | Huong cai thien |
|----------|-------|-----------------|
| Tracking don gian | Chi dung IoU, khong xu ly duoc khi nguoi di chuyen nhanh hoac bi che | Co the dung DeepSORT, ByteTrack |
| MediaPipe xu ly tuan tu | Moi nguoi duoc process lan luot, cham khi co nhieu nguoi | Co the xu ly song song hoac chi process nguoi gan nhat |
| Khong xu ly nguoi roi frame | Buffer cu van con trong memory | Them co che tu dong xoa buffer cua nguoi da roi frame |
| ID co the bi nhay | Neu nguoi bi mat tracking 1 frame roi xuat hien lai, se duoc gan ID moi | Them logic re-ID dua dac trung ngoai hinh |

---

## Cap nhat: Co che hien thi canh bao FALL co thoi han

**Ngay cap nhat:** 12/06/2026  
**Van de:** Canh bao "WARNING: FALL DETECTED!" hien thi lien tuc sau khi phat hien nga, ke ca khi nguoi da dung day hoac video chuyen sang canh khac.

### Mo ta van de

Truoc khi sua, khi model phat hien nga (probability >= threshold), status chuyen thanh "FALL" va canh bao "WARNING: FALL DETECTED!" hien thi o giua man hinh. Tuy nhien:

- Buffer luu 30 frame gan nhat, nen neu co nhieu frame "nga" trong buffer, cac frame tiep theo van bi du doan la FALL
- Status "FALL" van con duoc giu nguyen, khien canh bao hien thi lien tuc
- Khi video chuyen sang canh khac (nguoi dung day, hoac chuyen camera), canh bao van con hien thi

### Giai phap: Timer canh bao

Them co che **timer** de kiem soat thoi gian hien thi canh bao "WARNING: FALL DETECTED!" o giua man hinh.

### Cac thay doi chi tiet

#### 1. Them bien timer trong `__init__`

```python
# Timer hien thi canh bao FALL (so frame)
self.fall_alert_timers = {}
self.fall_alert_duration = 60  # 60 frame = 2 giay o 30fps
```

**Giai thich:**
- `fall_alert_timers`: Dictionary luu so frame con lai cua timer cho tung nguoi
- `fall_alert_duration`: Thoi gian hien thi canh bao (60 frame = 2 giay neu video 30fps)

#### 2. Khoi tao timer khi tao nguoi moi

Trong ham `process_frame`, khi khoi tao buffer cho nguoi moi:

```python
if person_id not in self.skeleton_buffers:
    self.skeleton_buffers[person_id] = deque(maxlen=CONFIG['window_size'])
    self.fall_probabilities[person_id] = 0.0
    self.statuses[person_id] = "NORMAL"
    self.fall_alert_timers[person_id] = 0  # Khoi tao timer = 0
```

#### 3. Giam timer moi frame

Trong vong lap xu ly tung nguoi:

```python
buffer = self.skeleton_buffers[person_id]

if self.fall_alert_timers.get(person_id, 0) > 0:
    self.fall_alert_timers[person_id] -= 1
```

**Giai thich:** Moi frame, timer giam di 1. Khi timer = 0, canh bao se tu dong tat.

#### 4. Bat timer khi phat hien nga

Khi model du doan probability >= threshold:

```python
if prob >= CONFIG['threshold']:
    self.statuses[person_id] = "FALL"
    self.fall_alert_timers[person_id] = self.fall_alert_duration  # Bat timer
else:
    self.statuses[person_id] = "NORMAL"
```

**Giai thich:** Khi phat hien nga, timer duoc dat thanh 60 frame. Canh bao se hien thi trong 60 frame tiep theo.

#### 5. Sua ham `draw_results` de kiem tra timer

```python
y_offset = 95
any_fall = False
for pid in sorted(self.statuses.keys()):
    status = self.statuses[pid]
    prob = self.fall_probabilities.get(pid, 0.0)
    buf_size = len(self.skeleton_buffers.get(pid, []))
    
    if status == "FALL":
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    
    # Chi hien thi canh bao neu timer > 0
    if self.fall_alert_timers.get(pid, 0) > 0:
        any_fall = True
    
    text = f"ID:{pid} {status} {prob*100:.1f}% [{buf_size}/{CONFIG['window_size']}]"
    cv2.putText(frame, text, (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    y_offset += 35
```

**Thay doi chinh:**
- Truoc: `any_fall = True` neu `status == "FALL"`
- Sau: `any_fall = True` chi khi `fall_alert_timers[pid] > 0`

**Ket qua:**
- Status va bounding box van hien thi mau do khi `status == "FALL"`
- Nhung canh bao "WARNING: FALL DETECTED!" o giua man hinh chi hien khi timer > 0
- Sau 60 frame, timer = 0, canh bao tu dong tat

### So do hoat dong

```
Frame 1: Model predict prob = 0.85 >= 0.5
    |
    v
Status = "FALL"
fall_alert_timers[pid] = 60
    |
    v
Hien thi:
- Bounding box mau do
- Panel: "ID:0 FALL 85.0% [30/30]"
- WARNING: FALL DETECTED! (giua man hinh)
- Vien do quanh frame

Frame 2-60: Timer giam dan (60 -> 59 -> ... -> 1)
    |
    v
Van hien thi:
- Bounding box mau do
- Panel: "ID:0 FALL ..."
- WARNING: FALL DETECTED! (giua man hinh)
- Vien do quanh frame

Frame 61: Timer = 0
    |
    v
Hien thi:
- Bounding box mau do (status van la "FALL")
- Panel: "ID:0 FALL ..."
- KHONG con WARNING: FALL DETECTED!
- KHONG con vien do quanh frame
```

### Cach chinh thoi gian hien thi canh bao

Trong ham `__init__`, dong 97:

```python
self.fall_alert_duration = 60  # 2 giay o 30fps
```

**Cong thuc:** `Thoi gian (giay) = fall_alert_duration / FPS`

| `fall_alert_duration` | FPS 30 | FPS 25 | FPS 60 |
|----------------------|--------|--------|--------|
| 30 | 1 giay | 1.2 giay | 0.5 giay |
| 60 | 2 giay | 2.4 giay | 1 giay |
| 90 | 3 giay | 3.6 giay | 1.5 giay |
| 150 | 5 giay | 6 giay | 2.5 giay |

### Uu diem cua giai phap

1. **Khong lam thay doi logic phat hien nga:** Model van predict binh thuong, chi thay doi thoi gian hien thi canh bao
2. **Tu dong reset:** Sau khi timer het, canh bao tu dong tat, khong can logic phuc tap
3. **Doc lap cho tung nguoi:** Moi nguoi co timer rieng, neu 2 nguoi nga cung luc, moi nguoi co canh bao rieng
4. **De dang chinh sua:** Chi can thay doi 1 bien `fall_alert_duration`

### Han che

1. **Status van la "FALL":** Sau khi timer het, status van la "FALL" (bounding box mau do), chi tat canh bao giua man hinh
2. **Khong tu dong reset status:** Neu nguoi da dung day nhung buffer van chua nhieu frame "nga", status van la "FALL"
3. **Phu thuoc FPS:** Thoi gian thuc te phu thuoc vao FPS cua video

### Huong cai thien tuong lai

1. **Them co che reset status:** Neu nguoi dung day (goc nghieng < 30 do lien tuc 30 frame), tu dong reset status ve "NORMAL"
2. **Them co che xoa buffer:** Khi nguoi roi khoi frame, tu dong xoa buffer va reset status
3. **Dung model phuc tap hon:** Su dung DeepSORT hoac ByteTrack de tracking tot hon, tranh ID nhay

---

## Cach chay test

```bash
python test_video.py --video duong_dan_video.mp4 --threshold 0.5
```

**Tham so:**

| Tham so | Mo ta | Mac dinh |
|----------|-------|----------|
| `--video` / `-v` | Duong dan den file video test | Bat buoc |
| `--output` / `-o` | Duong dan luu video output | Tu dong |
| `--threshold` / `-t` | Nguong phat hien nga (0.0 - 1.0) | 0.5 |
