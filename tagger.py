import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from feature_extractor import get_image_from_url

tagging_model = MobileNetV2(weights="imagenet")

def get_tags(image_url, top_n=5):
    img = get_image_from_url(image_url)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = tagging_model.predict(img_array, verbose=0)
    decoded = decode_predictions(predictions, top=top_n)[0]

    tags = [label.replace("_", " ") for (_, label, confidence) in decoded if confidence > 0.1]
    return tags