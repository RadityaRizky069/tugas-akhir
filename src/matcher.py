import cv2
import numpy as np

MATCH_THRESHOLD = 40
RATIO_THRESHOLD = 0.75


def match_features(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0, 0.0

    if len(desc1) == 0 or len(desc2) == 0:
        return 0, 0.0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < RATIO_THRESHOLD * n.distance:
                good_matches.append(m)

    total = max(len(desc1), len(desc2))
    similarity = (len(good_matches) / total * 100) if total > 0 else 0.0
    similarity = min(similarity, 100.0)

    return len(good_matches), similarity


def verify_palmprint(num_matches):
    if num_matches >= MATCH_THRESHOLD:
        return True
    return False


def calculate_score(num_matches, total_keypoints):
    if total_keypoints == 0:
        return 0.0
    score = (num_matches / total_keypoints * 100)
    return min(score, 100.0)
