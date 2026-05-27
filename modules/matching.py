import cv2
import numpy as np


def match_features(descriptors1, descriptors2, kp1, kp2):
    if descriptors1 is None or descriptors2 is None:
        return 0, 0.0, []

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return 0, 0.0, []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    total_good = len(good_matches)
    if total_good < 4:
        return 0, 0.0, []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        return 0, 0.0, []

    inliers_count = int(mask.sum())
    percentage = min(100.0, (inliers_count / total_good) * 100.0)

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i] == 1]

    return inliers_count, percentage, inlier_matches


def make_decision(percentage):
    return percentage >= 50.0
