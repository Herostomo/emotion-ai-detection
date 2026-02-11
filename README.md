# 🎭 Emotion AI Detection System

A real-time **facial emotion recognition desktop application** that uses Artificial Intelligence to analyze human facial expressions through a webcam and display the detected emotion with expressive emojis.

Built with **Python, Computer Vision, and Deep Learning**, this project demonstrates how AI can interpret human emotions in real time.

---

## 🚀 Features

✨ Real-time webcam face detection  
🧠 AI-powered facial emotion classification  
😀 Emoji feedback based on detected emotion  
🖥 Interactive desktop GUI using Tkinter  
⚡ Smooth and stable prediction system  
🔌 Works offline after initial model setup  

---

## 🛠 Tech Stack

| Technology | Purpose |
|-----------|---------|
Python | Core programming language |
OpenCV | Webcam capture and face detection |
HuggingFace Transformers | Pretrained Vision Transformer emotion model |
PyTorch | Deep learning inference engine |
Tkinter | Graphical User Interface (GUI) |
Pillow | Emoji image handling |

---

## 🧠 How It Works

1. The webcam captures live video frames.
2. OpenCV detects faces in each frame.
3. The detected face is processed by a **Vision Transformer deep learning model** trained for emotion recognition.
4. The system predicts one of the following emotions:

   - 😄 Happy  
   - 😢 Sad  
   - 😠 Angry  
   - 😲 Surprise  
   - 😨 Fear  
   - 🤢 Disgust  
   - 😐 Neutral  

5. The GUI displays:
   - The detected emotion label  
   - A matching emoji  

---

## 💻 Running the Project (For Developers)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Herostomo/emotion-ai-detection.git
cd emotion-ai-detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python main.py

```
📦 Windows Executable:
A packaged .exe version of this application is available separately for users who do not want to install Python.
⚠ Large AI model files and executables are not stored in this repository due to GitHub size limits.

🎯 Project Goal

This project showcases the integration of Computer Vision + Deep Learning + GUI Development to build an interactive AI system capable of understanding human emotions in real time.

🔮 Future Improvements:

🔹 Emotion history tracking
🔹 Confidence score visualization
🔹 Voice feedback based on emotion
🔹 Multi-face emotion detection
🔹 Web-based version

👨‍💻 Author

Kshitij Hedau
AI & Software Enthusiast 🚀

