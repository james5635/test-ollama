from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model (you can change to 'yolov8m.pt' or 'yolov8l.pt')
model = YOLO("yolov8n.pt")
# model = YOLO("yolov8s.pt")

# Start webcam (0 = default camera)
cap = cv2.VideoCapture("v.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame, stream=True)

    # Draw results on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Show frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
