import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_features

SIMILARITY_THRESHOLD = 0.90

def find_duplicates(media_items):
    """
    media_items: list of dicts, each like { "id": "...", "url": "..." }
    Returns: list of dicts { "id": ..., "isDuplicateOf": ... or None }
    """
    features_list = []
    valid_items = []

    for item in media_items:
        try:
            features = extract_features(item["url"])
            features_list.append(features)
            valid_items.append(item)
        except Exception as e:
            print(f"Skipping {item['id']}: {e}")

    if len(valid_items) < 2:
        return [{"id": item["id"], "isDuplicateOf": None} for item in valid_items]

    feature_matrix = np.array(features_list)
    similarity_matrix = cosine_similarity(feature_matrix)

    results = []
    already_matched = set()

    for i, item in enumerate(valid_items):
        if item["id"] in already_matched:
            continue

        duplicate_of = None
        for j in range(i):
            if similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
                duplicate_of = valid_items[j]["id"]
                already_matched.add(item["id"])
                break

        results.append({"id": item["id"], "isDuplicateOf": duplicate_of})

    return results