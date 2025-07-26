import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Start video
cap = cv2.VideoCapture("v.mp4")  # Or "your_video.mp4"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    detections = model(frame)[0]

    for box, cls in zip(detections.boxes.xyxy, detections.boxes.cls):
        if int(cls) != 0:  # 0 = person
            continue

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Convert to RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            for lm in result.pose_landmarks.landmark:
                cx = int(lm.x * (x2 - x1)) + x1
                cy = int(lm.y * (y2 - y1)) + y1
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            # Optional: draw lines
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                ),
            )

    cv2.imshow("YOLO + MediaPipe Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
