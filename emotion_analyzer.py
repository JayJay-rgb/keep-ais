try:
    from fer.fer import FER
    detector = FER(mtcnn=True)
    FER_AVAILABLE = True
except Exception as e:
    print(f"Warning: FER not available, emotion analysis disabled. Reason: {e}")
    detector = None
    FER_AVAILABLE = False

import numpy as np
import requests
from io import BytesIO
from PIL import Image

def download_np_image(url):
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

def analyze_emotions(image_url):
    if not FER_AVAILABLE:
        return None

    img = download_np_image(image_url)
    results = detector.detect_emotions(img)

    if not results:
        return None

    combined = {}
    for face in results:
        for emotion, score in face["emotions"].items():
            combined[emotion] = combined.get(emotion, 0) + score

    face_count = len(results)
    averaged = {k: round(v / face_count, 3) for k, v in combined.items()}
    return averaged