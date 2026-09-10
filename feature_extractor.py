import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO

model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def get_image_from_url(url):
    response = requests.get(url, timeout=(10, 30))
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))
    return img

def extract_features(image_url):
    img = get_image_from_url(image_url)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0)
    return features.flatten()