import cv2
import requests
import base64
import time
import pyttsx3
import subprocess
# OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_URL = "    https://f0d3249881f0.ngrok-free.app/api/generate"
MODEL = "llava:7b"  # You can use "bakllava", "llava", etc.
# MODEL = "gemma3:4b"  # You can use "bakllava", "llava", etc.
# MODEL = "gemma3:12b"  # You can use "bakllava", "llava", etc.


def capture_image():
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Let camera adjust

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Failed to capture image")
    return frame


def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


def describe_image(image_base64):
    payload = {
        "model": MODEL,
        "prompt": "Describe the image in detail.",
        "images": [image_base64],
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, )
    return response.json()["response"]


# Step 1: Capture
img = capture_image()
cv2.imshow("Captured Image", img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Step 2: Encode & send to Ollama
img_b64 = encode_image_to_base64(img)
description = describe_image(img_b64)

# Step 3: Output
# print("Ollama says:")
# print(description)

def ask_ollama(txt):
    prompt = f"{txt} \n summary this"
    response = requests.post(
        OLLAMA_URL,
        json={
            # "model": "gemma3:4b",  # or "mistral" or any model you've pulled
            # "model": "gemma3:27b",  # or "mistral" or any model you've pulled
            "model": MODEL,  # or "mistral" or any model you've pulled
            "prompt": prompt,
            "stream": False,
        },
    )
    return response.json()["response"]

message = ask_ollama(description)
print(description)
print("=======================")
print(message)
# for txt in message.split('\n'):
#     for txt2 in txt.split("."):
#     # time.sleep(10)
#         engine = pyttsx3.init()
#         engine.say(txt2)
#         engine.runAndWait()
# subprocess.run(["espeak", f'"{message}"'])

from gtts import gTTS
import os

def text_to_speech(text, lang='en', slow=False):
    """
    Converts text to speech using gTTS.

    Args:
        text: The text to be converted.
        lang: The language code (default is 'en' for English).
        slow: Whether to use a slower speed (default is False).
    """
    try:
        myobj = gTTS(text=text, lang=lang, slow=slow)
        myobj.save("welcome.mp3")
        os.system("xdg-open welcome.mp3")  # Plays the audio (Windows)
    except Exception as e:
        print(f"Error during text-to-speech: {e}")

from googletrans import Translator

translator = Translator()

# text = "Hello, how are you?"
text = message
result = translator.translate(text, src='en', dest='km')

print("Original:", text)
print("Translated:", result.text)


text_to_speech(result.text, lang='km')