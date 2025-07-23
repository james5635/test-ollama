import cv2

# url = "http://192.168.100.11:4747/video"
url = "http://192.168.100.11:4747/video"
cap = cv2.VideoCapture(url)

cv2.namedWindow("iPhone Stream", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("iPhone Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
