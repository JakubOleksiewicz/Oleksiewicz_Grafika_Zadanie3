
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"
VIDEO_PATH = "traffic_test_video.mp4"

CONFIDENCE = 0.50
IMAGE_SIZE = 640

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)


def preprocess(image):
    image = cv2.medianBlur(image, 3)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )


if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    processed = preprocess(frame)

    results = model(
        processed,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 9:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "TRAFFIC LIGHT",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Traffic Light Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
