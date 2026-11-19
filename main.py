import cv2

VIDEO_PATH = "traffic_test_video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Traffic Light Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()