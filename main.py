import cv2
from ultralytics import YOLO

VIDEO_PATH = "traffic_test_video.mp4.mp4"

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5, imgsz=640, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id == 9:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "TRAFFIC LIGHT",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

    cv2.imshow("Traffic Light Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()