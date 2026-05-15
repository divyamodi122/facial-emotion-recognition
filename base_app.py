import os
from fastapi import FastAPI, File, UploadFile
import cv2 as cv
from mtcnn_cv2 import MTCNN
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

# Device configuration
device = 'cpu'

# Load pretrained emotion detection model
emotion_model = torch.load('./weights/emotion_detector_v1.pt',
                           map_location=torch.device('cpu'))
emotion_model.eval()

# Emotion class labels
idx_to_class = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Neutral',
    5: 'Sadness',
    6: 'Surprise'
}

# Image preprocessing pipeline
test_transforms = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Initialize face detector
face_detector = MTCNN()

def detect_emotions(image):
    """Detect faces and predict emotion for each face."""
    detected_faces = face_detector.detect_faces(image)
    emotion_results = []

    for face in detected_faces:
        box = face['box']
        cropped_face = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

        img_tensor = test_transforms(Image.fromarray(cropped_face))
        img_tensor.unsqueeze_(0)

        with torch.no_grad():
            scores = emotion_model(img_tensor.to(device))

        scores = scores[0].data.cpu().numpy()
        predicted_emotion = idx_to_class[np.argmax(scores)]
        emotion_results.append(predicted_emotion)

    for idx, emotion in enumerate(emotion_results):
        detected_faces[idx]['emotion'] = emotion

    return detected_faces

app = FastAPI(
    title="Emotion Detection API",
    description="Upload an image to detect faces and recognize emotions.",
    version="1.0",
    contact={
        "name": "Divya Modi",
        "github": "https://github.com/divyamodi122"
    }
)

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "running", "message": "API is live and ready to detect emotions!"}

@app.post("/predict", tags=["Prediction"])
async def predict_emotion(image: UploadFile = File(...)):
    """Upload an image and get emotion predictions for all detected faces."""
    if not os.path.exists('./uploaded_images'):
        os.makedirs('./uploaded_images')

    temp_path = os.path.join("uploaded_images", "temp.jpg")
    with open(temp_path, "wb+") as f:
        f.write(image.file.read())

    img = cv.imread(temp_path)
    results = detect_emotions(img)

    if len(results) == 0:
        return {"status": 404, "message": "No faces detected in the image"}

    # Save prediction results to log file
    with open("predictions_log.txt", "a") as log:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n[{timestamp}] Image: {image.filename}\n")
        for idx, result in enumerate(results):
            log.write(f"  Face {idx+1}: {result['emotion']} (confidence: {round(result['confidence'], 4)})\n")

    return [
        {
            f"Face_{idx+1}": result['box'],
            "Confidence": round(result['confidence'], 4),
            "Emotion": result['emotion']
        }
        for idx, result in enumerate(results)
    ]