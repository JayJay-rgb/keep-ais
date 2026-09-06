import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

def download_cv_image(url):
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def score_image(url):
    img = download_cv_image(url)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness: variance of Laplacian (higher = sharper, less blurry)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness: average pixel intensity, penalize too dark/too bright
    brightness = np.mean(gray)
    brightness_score = 100 - abs(brightness - 128)

    # Combine into one score (weighted toward sharpness, since blur matters most)
    total_score = (sharpness * 0.7) + (brightness_score * 0.3)
    return round(total_score, 2)

def pick_best_from_group(media_items):
    """
    media_items: list of dicts { "id": ..., "url": ... } that are duplicates of each other
    Returns: id of the best one
    """
    scores = []
    for item in media_items:
        try:
            score = score_image(item["url"])
            scores.append((item["id"], score))
        except Exception as e:
            print(f"Skipping {item['id']}: {e}")

    if not scores:
        return None

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]