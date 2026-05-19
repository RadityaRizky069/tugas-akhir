import cv2

IMG_SIZE = 400


def preprocess_image(image):
    if image is None:
        return None, None, None

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    noise_reduced = cv2.GaussianBlur(contrast, (5, 5), 0)

    return gray, contrast, noise_reduced


def preprocess_single_channel(gray_img):
    if gray_img is None:
        return None, None

    resized = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(resized)

    noise_reduced = cv2.GaussianBlur(contrast, (5, 5), 0)

    return contrast, noise_reduced
