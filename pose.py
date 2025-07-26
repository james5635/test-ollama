import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
# Open webcam or video
cap = cv2.VideoCapture("IMG_3281.MOV")  # Change to 'video.mp4' for video file

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(image)

    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

    # Display
    cv2.imshow("Pose Estimation", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break  # Press ESC to exit

cap.release()
cv2.destroyAllWindows()
