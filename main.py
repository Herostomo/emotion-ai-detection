import cv2
import torch
import sys
import os
from tkinter import *
from PIL import Image, ImageTk
from transformers import AutoImageProcessor, AutoModelForImageClassification
from collections import deque, Counter

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load HuggingFace emotion model
processor = AutoImageProcessor.from_pretrained(resource_path("model"))
model = AutoModelForImageClassification.from_pretrained(resource_path("model"))
model.eval()

face_cascade = cv2.CascadeClassifier(resource_path("haarcascade_frontalface_default.xml"))

emoji_dict = {
    "angry": resource_path("emojis/angry.png"),
    "disgust": resource_path("emojis/disgust.png"),
    "fear": resource_path("emojis/fear.png"),
    "happy": resource_path("emojis/happy.png"),
    "sad": resource_path("emojis/sad.png"),
    "surprise": resource_path("emojis/surprised.png"),
    "neutral": resource_path("emojis/neutral.png")
}

root = Tk()
root.title("Emotion Detection System (Optimized AI)")
root.geometry("1000x600")
root.configure(bg="black")

video_label = Label(root)
video_label.pack(side=LEFT, padx=10, pady=10)

emoji_label = Label(root, bg="black")
emoji_label.pack(side=RIGHT, padx=50)

emotion_text = Label(root, font=("Arial", 30), bg="black", fg="white")
emotion_text.pack(side=BOTTOM, pady=20)

button_frame = Frame(root, bg="black")
button_frame.pack(side=BOTTOM)

cap = None
running = False
current_emotion = "neutral"

emotion_history = deque(maxlen=8)
frame_count = 0

def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        detect_emotion()

def stop_camera():
    global running
    running = False
    if cap:
        cap.release()

def predict_emotion(face_img):
    face_pil = Image.fromarray(face_img)
    inputs = processor(images=face_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()
        label = model.config.id2label[predicted_class]
    return label.lower()

def detect_emotion():
    global current_emotion, frame_count

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    frame_count += 1

    for (x, y, w, h) in faces:
        if w < 80 or h < 80:
            continue

        pad = 15
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]

        if frame_count % 3 == 0:
            pred = predict_emotion(face)
            emotion_history.append(pred)
            current_emotion = Counter(emotion_history).most_common(1)[0][0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, current_emotion, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video_label.imgtk = img
    video_label.config(image=img)

    if current_emotion not in emoji_dict:
        current_emotion = "neutral"

    emoji_img = ImageTk.PhotoImage(Image.open(emoji_dict[current_emotion]).resize((150,150)))
    emoji_label.imgtk = emoji_img
    emoji_label.config(image=emoji_img)

    emotion_text.config(text=f"Emotion: {current_emotion.capitalize()}")

    root.after(60, detect_emotion)

Button(button_frame, text="Start Camera", command=start_camera,
       font=("Arial",14), bg="green", fg="white").pack(side=LEFT, padx=20)

Button(button_frame, text="Stop Camera", command=stop_camera,
       font=("Arial",14), bg="red", fg="white").pack(side=LEFT, padx=20)

root.mainloop()
