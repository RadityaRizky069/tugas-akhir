import cv2


def extract_sift_features(image):
    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=8)
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if descriptors is None:
        return [], None

    return keypoints, descriptors
