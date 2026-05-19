import cv2


def extract_sift_features(image):
    if image is None:
        return None, None

    gray = image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def draw_sift_keypoints(image, keypoints, max_size=400):
    if image is None:
        return None

    gray = image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if len(image.shape) == 3:
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if keypoints is not None and len(keypoints) > 0:
        display = cv2.drawKeypoints(
            display, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    h, w = display.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(display, (new_w, new_h))

    return display
