import time
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"
VIDEO_PATH = "traffic_test_video.mp4"

CONFIDENCE = 0.50
IMAGE_SIZE = 960
HISTORY_SIZE = 9

tracks = {}
next_track_id = 0
frame_number = 0

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

COLOR_RANGES = {
    "CZERWONE": [
        (np.array([0, 60, 55]), np.array([12, 255, 255])),
        (np.array([165, 60, 55]), np.array([180, 255, 255])),
    ],
    "ŻÓŁTE": [(np.array([15, 60, 55]), np.array([35, 255, 255]))],
    "ZIELONE": [(np.array([36, 50, 55]), np.array([95, 255, 255]))],
}

COLORS = {
    "CZERWONE": (0, 0, 255),
    "ŻÓŁTE": (0, 255, 255),
    "ZIELONE": (0, 255, 0),
    "NIEZNANY": (255, 255, 255),
}


def stabilize_color(track_id, color):
    history = tracks[track_id]["history"]
    history.append(color)

    recognized = [value for value in history if value != "NIEZNANY"]

    if len(recognized) < 3:
        return "NIEZNANY"

    best_color, votes = Counter(recognized).most_common(1)[0]

    return best_color if votes >= 3 else "NIEZNANY"


def preprocess(image):
    image = cv2.medianBlur(image, 3)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))

    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def get_track(center):
    global next_track_id

    best_id = None
    min_distance = 100

    for track_id, track in tracks.items():
        if track["used"]:
            continue

        old_x, old_y = track["center"]

        distance = np.hypot(center[0] - old_x, center[1] - old_y)

        if distance < min_distance:
            min_distance = distance
            best_id = track_id

    if best_id is None:
        best_id = next_track_id
        next_track_id += 1

        tracks[best_id] = {
            "center": center,
            "history": deque(maxlen=HISTORY_SIZE),
            "used": False,
            "last_seen": frame_number,
        }

    track = tracks[best_id]
    track["center"] = center
    track["used"] = True
    track["last_seen"] = frame_number

    return best_id


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

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

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

previous_time = time.perf_counter()
fps = 0

video_fps = cap.get(cv2.CAP_PROP_FPS)

if video_fps <= 0:
    video_fps = 30

frame_time = 1 / video_fps
start_time = time.perf_counter()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    for track in tracks.values():
        track["used"] = False

    processed = preprocess(frame)

    results = model(processed, conf=CONFIDENCE, imgsz=IMAGE_SIZE, verbose=False)

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 9:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1

            if width < 12 or height < 25:
                continue

            ratio = height / max(width, 1)

            if ratio < 0.8 or ratio > 5.0:
                continue

            roi = frame[y1:y2, x1:x2]
            color_name = detect_color(roi)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            track_id = get_track(center)

            stable_color = stabilize_color(track_id, color_name)

            color = COLORS[stable_color]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID: {track_id} {stable_color}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    old_tracks = [
        track_id
        for track_id, track in tracks.items()
        if frame_number - track["last_seen"] > 20
    ]

    for track_id in old_tracks:
        del tracks[track_id]

    current_time = time.perf_counter()
    elapsed = current_time - previous_time

    if elapsed > 0:
        current_fps = 1 / elapsed
        fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps

    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Traffic Light Recognition", frame)

    elapsed_video = time.perf_counter() - start_time
    expected_time = frame_number * frame_time

    wait_time = max(1, int((expected_time - elapsed_video) * 1000))

    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break


def main():
    if not cap.isOpened():
        print("Cannot open video:", VIDEO_PATH)
        return


if __name__ == "__main__":
    try:
        main()
    finally:
        cap.release()
        cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
