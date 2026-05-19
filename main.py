import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QScrollArea,
    QGridLayout, QFrame, QSizePolicy, QGroupBox
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QSize

from src.preprocessing import preprocess_image, IMG_SIZE
from src.sift_feature import extract_sift_features, draw_sift_keypoints
from src.matcher import match_features, verify_palmprint, MATCH_THRESHOLD
from src.utils import convert_cv_to_qpixmap, resize_to_fit, create_placeholder_pixmap


PREVIEW_W = 280
PREVIEW_H = 280


class ImagePreviewCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #D1D5DB;
            }
        """)
        self.setMinimumSize(320, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1F2937; border: none;")
        layout.addWidget(self.title_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(PREVIEW_W, PREVIEW_H)
        self.image_label.setMaximumSize(PREVIEW_W + 40, PREVIEW_H + 40)
        self.image_label.setStyleSheet("""
            background-color: #F3F4F6;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
        """)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(8)
        layout.addLayout(self.button_layout)

    def set_image(self, pixmap):
        if pixmap and not pixmap.isNull():
            scaled = resize_to_fit(pixmap, PREVIEW_W, PREVIEW_H)
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.clear()

    def add_button(self, text, callback, color="#2563EB"):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(36)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._darken(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken(color, 0.2)};
            }}
        """)
        btn.clicked.connect(callback)
        self.button_layout.addWidget(btn)
        return btn

    def _darken(self, hex_color, amount=0.1):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = max(0, int(r * (1 - amount)))
        g = max(0, int(g * (1 - amount)))
        b = max(0, int(b * (1 - amount)))
        return f"#{r:02x}{g:02x}{b:02x}"


class PreprocessingCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #D1D5DB;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(12)

        header = QLabel("Preprocessing Citra")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #1F2937; border: none;")
        layout.addWidget(header)

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(12)

        self.cards = {}
        stages = [
            ("grayscale", "Grayscale"),
            ("contrast", "Peningkatan Kontras"),
            ("noise", "Reduksi Noise"),
        ]
        for key, title in stages:
            card = QFrame()
            card.setStyleSheet("""
                background-color: #F9FAFB;
                border-radius: 8px;
                border: 1px solid #E5E7EB;
            """)
            vbox = QVBoxLayout(card)
            vbox.setContentsMargins(8, 8, 8, 8)

            lbl_title = QLabel(title)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #4B5563; border: none;")
            vbox.addWidget(lbl_title)

            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setMinimumSize(180, 180)
            img_lbl.setStyleSheet("background-color: #F3F4F6; border-radius: 6px; border: none;")
            vbox.addWidget(img_lbl, alignment=Qt.AlignCenter)

            self.cards[key] = img_lbl
            preview_layout.addWidget(card)

        layout.addLayout(preview_layout)

        self.run_btn = QPushButton("Jalankan Preprocessing")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #1E40AF;
            }
        """)
        layout.addWidget(self.run_btn, alignment=Qt.AlignCenter)

    def set_image(self, key, pixmap):
        if key in self.cards:
            if pixmap and not pixmap.isNull():
                scaled = resize_to_fit(pixmap, 180, 180)
                self.cards[key].setPixmap(scaled)
            else:
                self.cards[key].clear()

    def clear_all(self):
        for key in self.cards:
            self.cards[key].clear()


class SiftCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #D1D5DB;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(12)

        header = QLabel("Hasil Ekstraksi Fitur SIFT")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #1F2937; border: none;")
        layout.addWidget(header)

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(12)

        self.panels = {}
        for key, title in [("ref", "Keypoint Referensi"), ("test", "Keypoint Gambar Uji")]:
            card = QFrame()
            card.setStyleSheet("""
                background-color: #F9FAFB;
                border-radius: 8px;
                border: 1px solid #E5E7EB;
            """)
            vbox = QVBoxLayout(card)
            vbox.setContentsMargins(8, 8, 8, 8)

            lbl_title = QLabel(title)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #4B5563; border: none;")
            vbox.addWidget(lbl_title)

            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setMinimumSize(240, 240)
            img_lbl.setStyleSheet("background-color: #F3F4F6; border-radius: 6px; border: none;")
            vbox.addWidget(img_lbl, alignment=Qt.AlignCenter)

            info_lbl = QLabel("")
            info_lbl.setAlignment(Qt.AlignCenter)
            info_lbl.setStyleSheet("font-size: 11px; color: #6B7280; border: none;")
            vbox.addWidget(info_lbl)

            self.panels[key] = {"img": img_lbl, "info": info_lbl}
            preview_layout.addWidget(card)

        layout.addLayout(preview_layout)

        self.run_btn = QPushButton("Ekstraksi Fitur SIFT")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #1E40AF;
            }
        """)
        layout.addWidget(self.run_btn, alignment=Qt.AlignCenter)

    def set_image(self, key, pixmap):
        if key in self.panels:
            if pixmap and not pixmap.isNull():
                scaled = resize_to_fit(pixmap, 240, 240)
                self.panels[key]["img"].setPixmap(scaled)
            else:
                self.panels[key]["img"].clear()

    def set_info(self, key, text):
        if key in self.panels:
            self.panels[key]["info"].setText(text)

    def clear_all(self):
        for key in self.panels:
            self.panels[key]["img"].clear()
            self.panels[key]["info"].clear()


class ResultCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet("""
            #card {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #D1D5DB;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)

        header = QLabel("Hasil Verifikasi")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #1F2937; border: none;")
        layout.addWidget(header)

        info_layout = QGridLayout()
        info_layout.setSpacing(8)

        labels = [
            ("Status", "status", ""),
            ("Jumlah Match", "match_count", "0"),
            ("Skor Kemiripan", "score", "0%"),
            ("Keputusan", "decision", "-"),
        ]

        self.result_labels = {}
        for i, (label_text, key, default) in enumerate(labels):
            lbl_title = QLabel(f"{label_text} :")
            lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #374151; border: none;")
            lbl_title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            info_layout.addWidget(lbl_title, i, 0)

            lbl_value = QLabel(default)
            lbl_value.setObjectName("value")
            lbl_value.setStyleSheet("font-size: 13px; color: #1F2937; border: none;")
            lbl_value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            info_layout.addWidget(lbl_value, i, 1)
            self.result_labels[key] = lbl_value

        layout.addLayout(info_layout)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.verify_btn = QPushButton("Verifikasi Palmprint")
        self.verify_btn.setCursor(Qt.PointingHandCursor)
        self.verify_btn.setMinimumHeight(40)
        self.verify_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #1E40AF;
            }
        """)
        button_layout.addWidget(self.verify_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #6B7280;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4B5563;
            }
            QPushButton:pressed {
                background-color: #374151;
            }
        """)
        button_layout.addWidget(self.reset_btn)

        layout.addLayout(button_layout)

    def set_result(self, status_text, match_count, score, decision, is_match):
        green = "#16A34A"
        red = "#DC2626"
        color = green if is_match else red

        self.result_labels["status"].setText(status_text)
        self.result_labels["status"].setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color}; border: none;")

        self.result_labels["match_count"].setText(str(match_count))
        self.result_labels["match_count"].setStyleSheet("font-size: 13px; color: #1F2937; border: none;")

        self.result_labels["score"].setText(f"{score:.1f}%")
        self.result_labels["score"].setStyleSheet("font-size: 13px; color: #1F2937; border: none;")

        decision_color = green if is_match else red
        self.result_labels["decision"].setText(decision)
        self.result_labels["decision"].setStyleSheet(f"font-size: 14px; font-weight: bold; color: {decision_color}; border: none;")

    def clear_all(self):
        for key in self.result_labels:
            self.result_labels[key].setText("-" if key in ["status", "decision"] else "0" if key == "match_count" else "0%")
            self.result_labels[key].setStyleSheet("font-size: 13px; color: #1F2937; border: none;")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem Verifikasi Palmprint Menggunakan Metode SIFT")
        self.setMinimumSize(1100, 900)
        self.setStyleSheet("background-color: #F4F7FB;")

        self.ref_image = None
        self.test_image = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.test_keypoints = None
        self.test_descriptors = None

        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        self.setCentralWidget(scroll)

        container = QWidget()
        container.setStyleSheet("background-color: #F4F7FB;")
        scroll.setWidget(container)

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(16)

        self._build_header(main_layout)
        self._build_input_section(main_layout)
        self._build_preprocessing_section(main_layout)
        self._build_sift_section(main_layout)
        self._build_result_section(main_layout)

        main_layout.addStretch()

    def _build_header(self, parent):
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: transparent;")
        header_layout = QVBoxLayout(header_widget)
        header_layout.setSpacing(4)
        header_layout.setContentsMargins(0, 10, 0, 10)

        title = QLabel("Sistem Verifikasi Palmprint\nMenggunakan Metode SIFT")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #1F2937;
            border: none;
            padding: 5px;
        """)
        header_layout.addWidget(title)

        subtitle = QLabel("Ekstraksi Ciri Telapak Tangan untuk Akses Kontrol Keamanan")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 14px;
            color: #6B7280;
            border: none;
            padding: 2px;
        """)
        header_layout.addWidget(subtitle)

        parent.addWidget(header_widget)

    def _build_input_section(self, parent):
        input_layout = QHBoxLayout()
        input_layout.setSpacing(16)

        ref_card = ImagePreviewCard("Gambar Referensi")
        self.ref_preview = ref_card.image_label
        ref_card.add_button("Upload Referensi", self._load_reference)
        self.ref_placeholder = create_placeholder_pixmap(PREVIEW_W, PREVIEW_H)
        ref_card.set_image(self.ref_placeholder)
        input_layout.addWidget(ref_card)

        test_card = ImagePreviewCard("Gambar Uji")
        self.test_preview = test_card.image_label
        test_card.add_button("Upload Gambar Uji", self._load_test)
        test_card.add_button("Ambil dari Kamera", self._camera_placeholder, "#6B7280")
        self.test_placeholder = create_placeholder_pixmap(PREVIEW_W, PREVIEW_H)
        test_card.set_image(self.test_placeholder)
        input_layout.addWidget(test_card)

        parent.addLayout(input_layout)

    def _build_preprocessing_section(self, parent):
        self.preprocessing_card = PreprocessingCard()
        self.preprocessing_card.run_btn.clicked.connect(self._run_preprocessing)
        parent.addWidget(self.preprocessing_card)

    def _build_sift_section(self, parent):
        self.sift_card = SiftCard()
        self.sift_card.run_btn.clicked.connect(self._run_sift)
        parent.addWidget(self.sift_card)

    def _build_result_section(self, parent):
        self.result_card = ResultCard()
        self.result_card.verify_btn.clicked.connect(self._run_verification)
        self.result_card.reset_btn.clicked.connect(self._reset_all)
        parent.addWidget(self.result_card)

    def _load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Gambar Referensi", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.ref_image = img
                pixmap = convert_cv_to_qpixmap(img)
                self.ref_preview.setPixmap(resize_to_fit(pixmap, PREVIEW_W, PREVIEW_H))
                self._log(f"Referensi dimuat: {os.path.basename(path)}")
            else:
                QMessageBox.warning(self, "Error", "Gagal memuat gambar referensi.")

    def _load_test(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Gambar Uji", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.test_image = img
                pixmap = convert_cv_to_qpixmap(img)
                self.test_preview.setPixmap(resize_to_fit(pixmap, PREVIEW_W, PREVIEW_H))
                self._log(f"Gambar uji dimuat: {os.path.basename(path)}")
            else:
                QMessageBox.warning(self, "Error", "Gagal memuat gambar uji.")

    def _camera_placeholder(self):
        QMessageBox.information(
            self, "Informasi",
            "Fitur kamera belum tersedia.\nSilakan upload gambar uji melalui tombol 'Upload Gambar Uji'."
        )

    def _run_preprocessing(self):
        if self.test_image is None:
            QMessageBox.warning(self, "Peringatan", "Silakan upload gambar uji terlebih dahulu.")
            return

        gray, contrast, noise = preprocess_image(self.test_image)

        if gray is not None:
            self.preprocessing_card.set_image("grayscale", convert_cv_to_qpixmap(gray))
        if contrast is not None:
            self.preprocessing_card.set_image("contrast", convert_cv_to_qpixmap(contrast))
        if noise is not None:
            self.preprocessing_card.set_image("noise", convert_cv_to_qpixmap(noise))

        self._log("Preprocessing selesai.")

    def _run_sift(self):
        if self.ref_image is None:
            QMessageBox.warning(self, "Peringatan", "Silakan upload gambar referensi terlebih dahulu.")
            return
        if self.test_image is None:
            QMessageBox.warning(self, "Peringatan", "Silakan upload gambar uji terlebih dahulu.")
            return

        if self.ref_image is not None:
            kp_ref, desc_ref = extract_sift_features(self.ref_image)
            self.ref_keypoints = kp_ref
            self.ref_descriptors = desc_ref
            if kp_ref is not None:
                img_kp = draw_sift_keypoints(self.ref_image, kp_ref)
                self.sift_card.set_image("ref", convert_cv_to_qpixmap(img_kp))
                self.sift_card.set_info("ref", f"Keypoint: {len(kp_ref)}")
            else:
                self.sift_card.set_info("ref", "Tidak ada keypoint")

        if self.test_image is not None:
            kp_test, desc_test = extract_sift_features(self.test_image)
            self.test_keypoints = kp_test
            self.test_descriptors = desc_test
            if kp_test is not None:
                img_kp = draw_sift_keypoints(self.test_image, kp_test)
                self.sift_card.set_image("test", convert_cv_to_qpixmap(img_kp))
                self.sift_card.set_info("test", f"Keypoint: {len(kp_test)}")
            else:
                self.sift_card.set_info("test", "Tidak ada keypoint")

        self._log("Ekstraksi fitur SIFT selesai.")

    def _run_verification(self):
        if self.ref_descriptors is None or self.test_descriptors is None:
            QMessageBox.warning(
                self, "Peringatan",
                "Silakan jalankan 'Ekstraksi Fitur SIFT' terlebih dahulu."
            )
            return

        if self.ref_descriptors is None or len(self.ref_descriptors) == 0:
            QMessageBox.warning(
                self, "Error",
                "Tidak ada descriptor SIFT pada gambar referensi.\n"
                "Gunakan gambar palmprint yang lebih jelas."
            )
            return

        if self.test_descriptors is None or len(self.test_descriptors) == 0:
            QMessageBox.warning(
                self, "Error",
                "Tidak ada descriptor SIFT pada gambar uji.\n"
                "Gunakan gambar palmprint yang lebih jelas."
            )
            return

        num_matches, similarity = match_features(self.ref_descriptors, self.test_descriptors)
        is_match = verify_palmprint(similarity)

        if is_match:
            status_text = "COCOK"
            decision = "Akses Diterima"
        else:
            status_text = "TIDAK COCOK"
            decision = "Akses Ditolak"

        self.result_card.set_result(
            status_text, num_matches, similarity, decision, is_match
        )

        self._log(
            f"Verifikasi: {status_text} | Match: {num_matches} | "
            f"Skor: {similarity:.1f}% | {decision}"
        )

        kp_ref = self.ref_keypoints or []
        kp_test = self.test_keypoints or []
        print("Keypoint Referensi:", len(kp_ref))
        print("Keypoint Uji:", len(kp_test))
        print("Good Matches:", num_matches)
        print("Similarity Score:", similarity)
        print("Status:", status_text)
        print("Keputusan:", decision)

    def _reset_all(self):
        self.ref_image = None
        self.test_image = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.test_keypoints = None
        self.test_descriptors = None

        self.ref_preview.setPixmap(resize_to_fit(self.ref_placeholder, PREVIEW_W, PREVIEW_H))
        self.test_preview.setPixmap(resize_to_fit(self.test_placeholder, PREVIEW_W, PREVIEW_H))

        self.preprocessing_card.clear_all()
        self.sift_card.clear_all()
        self.result_card.clear_all()

        self._log("Semua data telah di-reset.")

    def _log(self, message):
        print(f"[SISTEM] {message}")


def main():
    app = QApplication(sys.argv)

    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
