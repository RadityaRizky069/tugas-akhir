"""
Modul Validasi Input Telapak Tangan
Menggunakan MediaPipe Hands untuk memastikan citra yang diunggah
benar-benar merupakan telapak tangan manusia sebelum masuk ke pipeline.
"""

import cv2
import numpy as np
import mediapipe as mp

# ── Singleton MediaPipe Hands ───────────────────────────────────────────
# Model di-cache agar tidak di-load ulang setiap pemanggilan fungsi.
_mp_hands = mp.solutions.hands
_detector = _mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


def validate_hand_image(image_input):
    """
    Memvalidasi apakah citra mengandung struktur tangan manusia.

    Parameters
    ----------
    image_input : str atau np.ndarray
        Path file gambar atau array numpy (BGR/Grayscale).

    Returns
    -------
    (bool, str)
        (True,  pesan sukses) jika tangan terdeteksi.
        (False, pesan error)  jika bukan tangan / gagal membaca file.
    """
    # ── Muat citra ──────────────────────────────────────────────────────
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            return False, "Gagal membaca file gambar."
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image_input.copy()
    else:
        return False, "Format input tidak dikenali."

    # ── Konversi BGR → RGB (MediaPipe memerlukan RGB) ───────────────────
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Deteksi landmark tangan ─────────────────────────────────────────
    results = _detector.process(img_rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        landmarks = results.multi_hand_landmarks[0]
        num_landmarks = len(landmarks.landmark)
        return True, f"Tangan terdeteksi ({num_landmarks} landmarks)."

    return False, "Ini bukan telapak tangan."
