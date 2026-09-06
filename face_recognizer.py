import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import get_image_from_url, model as feature_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
MATCH_THRESHOLD = 0.75

def get_embedding_from_crop(crop_img):
    img_array = keras_image.img_to_array(crop_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_model.predict(img_array, verbose=0)
    return features.flatten()

def get_face_embeddings(image_url):
    """Detect faces in an image and return their embeddings + bounding boxes."""
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get(image_url, timeout=10)
    pil_img = Image.open(BytesIO(response.content)).convert("RGB")
    np_img = np.array(pil_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    embeddings = []
    for (x, y, w, h) in faces:
        face_crop = pil_img.crop((x, y, x + w, y + h)).resize((224, 224))
        embedding = get_embedding_from_crop(face_crop)
        embeddings.append(embedding)

    return embeddings

def enroll_and_match(target_image_url, known_users):
    """
    known_users: list of { "userId": ..., "profilePictureUrl": ... }
    Returns: list of userIds matched in the target image
    """
    known_embeddings = []
    for user in known_users:
        try:
            faces = get_face_embeddings(user["profilePictureUrl"])
            if faces:
                known_embeddings.append({"userId": user["userId"], "embedding": faces[0]})
        except Exception as e:
            print(f"Couldn't enroll {user['userId']}: {e}")

    if not known_embeddings:
        return []

    target_faces = get_face_embeddings(target_image_url)
    matched_user_ids = []

    for face_embedding in target_faces:
        best_match = None
        best_score = 0

        for known in known_embeddings:
            score = cosine_similarity([face_embedding], [known["embedding"]])[0][0]
            if score > best_score:
                best_score = score
                best_match = known["userId"]

        if best_score >= MATCH_THRESHOLD:
            matched_user_ids.append(best_match)

    return matched_user_ids