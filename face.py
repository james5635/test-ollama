import cv2
import requests
import time

# OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_URL = " https://273bfefb3b1e.ngrok-free.app/api/generate"
YOUR_NAME = "Rojame"

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

greeted = False


def ask_ollama(name):
    prompt = f"Greet this person. Their name is {name}. don't send emoji"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "gemma3:4b",  # or "mistral" or any model you've pulled
            "prompt": prompt,
            "stream": False,
        },
    )
    return response.json()["response"]


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0 and not greeted:
        print("Face detected!")
        message = ask_ollama(YOUR_NAME)
        print("Ollama says:", message)
        # greeted = True  # Avoid repeating
        import pyttsx3

        # time.sleep(10)
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()

    # Draw rectangles around faces
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
