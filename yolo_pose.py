from ultralytics import YOLO
import cv2

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

cv2.namedWindow("YOLOv8 Pose", cv2.WINDOW_NORMAL)
# Open webcam
# cap = cv2.VideoCapture("v.mp4")
cap = cv2.VideoCapture("IMG_3281.MOV")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Plot pose on frame
    annotated_frame = results[0].plot()

    # Show
    cv2.imshow("YOLOv8 Pose", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
