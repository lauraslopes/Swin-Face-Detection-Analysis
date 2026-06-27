from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-face.pt")

img = cv2.imread("screenshot.png")
results = model(img, conf=0.4)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f"{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1
        )

cv2.imwrite("out.jpg", img)
