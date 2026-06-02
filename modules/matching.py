import cv2
import numpy as np

# Configuration thresholds (can be tuned)
DECISION_THRESHOLD = 75.0      # minimum inlier percentage to accept
MIN_INLIERS = 15               # minimum absolute number of inliers
LOWE_RATIO = 0.75              # Lowe's ratio threshold for good matches
SYMMETRY_CHECK = True          # enable cross-check symmetry
RANSAC_REPROJ_THRESH = 4.0     # RANSAC reprojection threshold


def match_features(descriptors1, descriptors2, kp1, kp2):
    if descriptors1 is None or descriptors2 is None:
        return 0, 0.0, []

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return 0, 0.0, []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    # matches from 1 to 2
    matches12 = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches21 = bf.knnMatch(descriptors2, descriptors1, k=2)

    good1 = []
    for m, n in matches12:
        if m.distance < LOWE_RATIO * n.distance:
            good1.append(m)

    good2 = []
    for m, n in matches21:
        if m.distance < LOWE_RATIO * n.distance:
            good2.append(m)

    # symmetry check
    good_matches = []
    if SYMMETRY_CHECK:
        # create set of (queryIdx, trainIdx) for good2
        good2_set = {(m.trainIdx, m.queryIdx) for m in good2}
        for m in good1:
            if (m.queryIdx, m.trainIdx) in good2_set:
                good_matches.append(m)
    else:
        good_matches = good1

    total_good = len(good_matches)
    if total_good < 4:
        return 0, 0.0, []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)

    if H is None or mask is None:
        return 0, 0.0, []

    inliers_count = int(mask.sum())
    percentage = min(100.0, (inliers_count / total_good) * 100.0)

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i] == 1]

    return inliers_count, percentage, inlier_matches


def make_decision(percentage, inliers_count):
    return percentage >= DECISION_THRESHOLD and inliers_count >= MIN_INLIERS
