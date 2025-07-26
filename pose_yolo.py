import cv2
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLOv8 for person detection
yolo = YOLO("yolov8n.pt")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture("v.mp4")  # Or use a video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 object detection
    results = yolo(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if int(cls) != 0:  # class 0 = person
            continue

        x1, y1, x2, y2 = map(int, box)
        cropped = frame[y1:y2, x1:x2]

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Pose estimation on cropped person
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            # Scale landmarks back to original frame size
            for landmark in result.pose_landmarks.landmark:
                lx = int(landmark.x * (x2 - x1)) + x1
                ly = int(landmark.y * (y2 - y1)) + y1
                cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)

            # Optional: Draw full pose on cropped image and paste back
            # mp_drawing.draw_landmarks(cropped, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show result
    cv2.imshow("YOLO + MediaPipe Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
