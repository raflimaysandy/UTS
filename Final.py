import cv2
from ultralytics import YOLO

# ── Daftar kelas (urut sesuai training)
CLASS_NAMES = [
    "crack",
    "corrosion",
    "squat",
    "peeling",
    "scratch",
    "normal"
]

# ── Muat model YOLOv8
model = YOLO("best.pt")        # ubah path jika perlu

# ── Confidence threshold
CONF_THRES = 0.25            # atur sesuai kebutuhan

# ── Buka kamera (atau ganti dengan path video)
cap = cv2.VideoCapture(1)      # 0 = webcam default

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference; YOLO sudah mem‑filter prediksi di bawah CONF_THRES
    results = model(frame, conf=CONF_THRES)

    # Gambar bounding‑box & label
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRES:          # filter tambahan (opsional)
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id       = int(box.cls[0])
            class_name     = CLASS_NAMES[class_id]
            label          = f"{class_name}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Tampilkan hasil
    cv2.imshow("Deteksi Kerusakan Rel Kereta", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

