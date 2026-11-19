
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"
VIDEO_PATH = "traffic_test_video.mp4"

CONFIDENCE = 0.50
IMAGE_SIZE = 640

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

COLOR_RANGES = {
    "CZERWONE": [
        (np.array([0, 60, 55]), np.array([12, 255, 255])),
        (np.array([165, 60, 55]), np.array([180, 255, 255]))
    ],
    "ŻÓŁTE": [
        (np.array([15, 60, 55]), np.array([35, 255, 255]))
    ],
    "ZIELONE": [
        (np.array([36, 50, 55]), np.array([95, 255, 255]))
    ]
}

COLORS = {
    "CZERWONE": (0, 0, 255),
    "ŻÓŁTE": (0, 255, 255),
    "ZIELONE": (0, 255, 0),
    "NIEZNANY": (255, 255, 255)
}



def detect_color(roi):
    if roi.size == 0:
        return "NIEZNANY"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

    scores = {}
    area = hsv.shape[0] * hsv.shape[1]

    for name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8)
        )

        pixels = cv2.countNonZero(mask)
        scores[name] = pixels / max(area, 1)

        if pixels < max(4, int(area * 0.002)):
            scores[name] = 0

    best = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)

    if scores[best] == 0:
        return "NIEZNANY"

    if sorted_scores[1] > scores[best] * 0.85:
        return "NIEZNANY"

    return best



if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 9:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            roi = frame[y1:y2, x1:x2]
            color_name = detect_color(roi)
            color = COLORS[color_name]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                color_name,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Traffic Light Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
