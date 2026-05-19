import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


def convert_cv_to_qpixmap(cv_img):
    if cv_img is None:
        return QPixmap()

    height, width = cv_img.shape[:2]
    bytes_per_line = 3 * width

    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
    else:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    q_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image)


def resize_to_fit(pixmap, max_w, max_h):
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def create_placeholder_pixmap(width=300, height=300):
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    cv2.putText(img, "Belum ada", (width // 2 - 60, height // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    cv2.putText(img, "gambar", (width // 2 - 40, height // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    return convert_cv_to_qpixmap(img)
