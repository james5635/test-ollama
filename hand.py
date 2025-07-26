import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
hand_raised = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get y-coordinates of wrist and middle finger tip
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            middle_tip_y = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ].y

            # If wrist is lower than middle fingertip => hand is raised
            if middle_tip_y < wrist_y:
                if not hand_raised:
                    print("Hello")
                    hand_raised = True
            else:
                hand_raised = False

    cv2.imshow("Hand Tracking", frame)
    # cv2.imshow("Hand Trackingx", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
