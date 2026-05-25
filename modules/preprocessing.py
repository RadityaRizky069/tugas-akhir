import cv2
import numpy as np


def preprocess_and_extract_roi(image_input):
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    else:
        return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None

    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    half = 67
    H, W = blurred.shape

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half + 1
    y2 = cy + half + 1

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - W)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        blurred = cv2.copyMakeBorder(blurred, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    roi = blurred[y1:y2, x1:x2]

    if roi.shape != (135, 135):
        roi = cv2.resize(roi, (135, 135))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)

    return roi


def preprocess_and_extract_roi_detailed(image_input):
    """Preprocesses input and returns both final ROI and dict of intermediate images."""
    steps = {}
    
    # 1. Original
    if isinstance(image_input, str):
        img_color = cv2.imread(image_input)
        if img_color is None:
            return None, None
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        steps['original'] = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img_color = image_input.copy()
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            steps['original'] = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        else:
            img_gray = image_input.copy()
            steps['original'] = img_gray.copy()
    else:
        return None, None

    steps['grayscale'] = img_gray.copy()

    # 2. Gaussian Blur (Noise Removal)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    steps['blurred'] = blurred.copy()

    # 3. Thresholding (Otsu's Binarization)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps['thresh'] = thresh.copy()

    # 4. Morphological Operations (Close & Open)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    steps['morph'] = morph.copy()

    # 5. Contour & Centroid Extraction
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None, None

    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None, None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Create visualization with centroid marker and palm contour
    centroid_viz = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(centroid_viz, [largest], -1, (16, 185, 129), 2)  # Emerald green contour
    cv2.circle(centroid_viz, (cx, cy), 8, (239, 68, 68), -1)          # Red centroid dot
    steps['centroid'] = centroid_viz

    # 6. ROI Crop
    half = 67
    H, W = blurred.shape

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half + 1
    y2 = cy + half + 1

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - W)

    temp_blurred = blurred.copy()
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        temp_blurred = cv2.copyMakeBorder(temp_blurred, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    roi = temp_blurred[y1:y2, x1:x2]

    if roi.shape != (135, 135):
        roi = cv2.resize(roi, (135, 135))
    steps['roi_crop'] = roi.copy()

    # 7. CLAHE (Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_clahe = clahe.apply(roi)
    steps['roi_clahe'] = roi_clahe.copy()

    return roi_clahe, steps
