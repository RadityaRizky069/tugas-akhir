import cv2
import numpy as np

HALF = 100
ROI_SIZE = 2 * HALF
MIN_CONTOUR_AREA = 500
BLUR_VARIANCE_MIN = 15
PCA_ALIGN = True


def check_image_quality(image_input):
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, "Cannot read image"
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    else:
        return False, "Invalid input type"

    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    if laplacian_var < BLUR_VARIANCE_MIN:
        return False, f"Blurry (Laplacian variance={laplacian_var:.1f}, min={BLUR_VARIANCE_MIN})"

    return True, f"OK (variance={laplacian_var:.1f})"


def _segment_hand(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    return blurred, morph


def _get_largest_contour(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    return largest


def _pca_align(contour, image_shape):
    pts = contour.squeeze().astype(np.float64)
    if pts.ndim != 2 or pts.shape[0] < 10:
        return None, None

    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    angle_norm = angle % 180
    rot_angle = 90 - angle_norm

    h, w = image_shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    return rot_angle, M


def _extract_centered_roi(image, cx, cy, half=HALF):
    H, W = image.shape
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half + 1
    y2 = cy + half + 1

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - W)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    if roi.shape != (ROI_SIZE, ROI_SIZE):
        roi = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))
    return roi


def _apply_clahe(roi):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(roi)


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

    blurred, morph = _segment_hand(img)
    largest = _get_largest_contour(morph)
    if largest is None:
        return None

    H, W = blurred.shape
    use_img = blurred
    final_contour = largest

    if PCA_ALIGN:
        _, rot_M = _pca_align(largest, (H, W))
        if rot_M is not None:
            rotated = cv2.warpAffine(blurred, rot_M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rotated_mask = cv2.warpAffine(morph, rot_M, (W, H), flags=cv2.INTER_NEAREST)
            use_img = rotated
            cr = _get_largest_contour(rotated_mask)
            if cr is not None:
                final_contour = cr

    M = cv2.moments(final_contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    roi = _extract_centered_roi(use_img, cx, cy)
    if roi is None:
        return None
    roi = _apply_clahe(roi)
    return roi


def preprocess_and_extract_roi_detailed(image_input):
    steps = {}

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

    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    steps['blurred'] = blurred.copy()

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps['thresh'] = thresh.copy()

    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    steps['morph'] = morph.copy()

    largest = _get_largest_contour(morph)
    if largest is None:
        return None, None

    H, W = blurred.shape
    use_img = blurred
    final_contour = largest

    centroid_viz = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(centroid_viz, [largest], -1, (16, 185, 129), 2)
    steps['centroid'] = centroid_viz

    if PCA_ALIGN:
        _, rot_M = _pca_align(largest, (H, W))
        if rot_M is not None:
            rotated = cv2.warpAffine(blurred, rot_M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rotated_mask = cv2.warpAffine(morph, rot_M, (W, H), flags=cv2.INTER_NEAREST)
            steps['rotated'] = rotated.copy()
            use_img = rotated
            cr = _get_largest_contour(rotated_mask)
            if cr is not None:
                final_contour = cr
        else:
            steps['rotated'] = blurred.copy()
    else:
        steps['rotated'] = blurred.copy()

    M_c = cv2.moments(final_contour)
    if M_c['m00'] == 0:
        return None, None
    cx = int(M_c['m10'] / M_c['m00'])
    cy = int(M_c['m01'] / M_c['m00'])

    cv2.circle(steps['centroid'], (cx, cy), 8, (239, 68, 68), -1)

    roi = _extract_centered_roi(use_img, cx, cy)
    if roi is None:
        return None, None
    steps['roi_crop'] = roi.copy()

    roi = _apply_clahe(roi)
    steps['roi_clahe'] = roi.copy()

    return roi, steps
