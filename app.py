from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from duplicate_detector import find_duplicates
from quality_scorer import pick_best_from_group
from tagger import get_tags
from emotion_analyzer import analyze_emotions
from face_recognizer import enroll_and_match

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "AI service is running"})

@app.route("/analyze/duplicates", methods=["POST"])
def analyze_duplicates():
    data = request.get_json()
    media_items = data.get("media", [])
    if not media_items:
        return jsonify({"error": "No media provided"}), 400
    results = find_duplicates(media_items)
    return jsonify({"results": results})

@app.route("/analyze/best-photo", methods=["POST"])
def analyze_best_photo():
    data = request.get_json()
    media_items = data.get("media", [])
    if not media_items:
        return jsonify({"error": "No media provided"}), 400
    best_id = pick_best_from_group(media_items)
    return jsonify({"bestId": best_id})

@app.route("/analyze/tags", methods=["POST"])
def analyze_tags():
    data = request.get_json()
    image_url = data.get("url")
    if not image_url:
        return jsonify({"error": "No url provided"}), 400
    tags = get_tags(image_url)
    return jsonify({"tags": tags})

@app.route("/analyze/emotions", methods=["POST"])
def analyze_emotions_route():
    data = request.get_json()
    image_url = data.get("url")
    if not image_url:
        return jsonify({"error": "No url provided"}), 400
    emotions = analyze_emotions(image_url)
    return jsonify({"emotions": emotions})

@app.route("/analyze/faces", methods=["POST"])
def analyze_faces():
    data = request.get_json()
    target_url = data.get("url")
    known_users = data.get("knownUsers", [])
    if not target_url or not known_users:
        return jsonify({"error": "Missing url or knownUsers"}), 400
    matched = enroll_and_match(target_url, known_users)
    return jsonify({"matchedUserIds": matched})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)