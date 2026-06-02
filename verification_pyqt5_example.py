import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFrame, QScrollArea, QLabel, QPushButton, QTabWidget,
    QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class Sidebar(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(200)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(QLabel("Menu"))
        for txt in ["Registrasi", "Verifikasi", "Kelola User"]:
            btn = QPushButton(txt)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(btn)
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.setLayout(layout)


class VerificationPage(QWidget):
    """The whole verification page that will be placed inside a QScrollArea."""
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._build_ui()

    def _build_ui(self):
        # Main horizontal layout: sidebar | content
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar (fixed)
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # Content area (will be inside scroll area)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)

        # ---- Identitas & Gambar Uji card ----
        ident_card = self._create_card("Identitas & Gambar Uji",
                                       "Pilih user terdaftar dan upload file gambar telapak tangan.")
        # Placeholder controls
        ident_form = QVBoxLayout()
        ident_form.addWidget(QLabel("Username: "))
        ident_form.addWidget(QLabel("Folder Dataset: [ Browse ]"))
        ident_form.addWidget(QPushButton("Mulai Registrasi"))
        ident_card.layout().addLayout(ident_form)
        content_layout.addWidget(ident_card)

        # ---- Status Hasil Verifikasi card ----
        result_card = self._create_card("Status Hasil Verifikasi",
                                        "Keputusan akhir sistem pencocokan biometrik.")
        self.result_label = QLabel("Menunggu Input Verifikasi")
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #555;")
        result_card.layout().addWidget(self.result_label)
        self.detail_label = QLabel("Pilih data dan klik tombol 'Verifikasi Sekarang' untuk mencocokkan keypoints.")
        self.detail_label.setWordWrap(True)
        result_card.layout().addWidget(self.detail_label)
        content_layout.addWidget(result_card)

        # ---- Preprocessing preview tabs ----
        preview_card = self._create_card("Tahapan Preprocessing Citra Uji",
                                         "Tampilan interaktif hasil setiap langkah algoritma pengolahan citra.")
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        tab_names = ["Asli", "Grayscale", "Gaussian", "Threshold",
                     "Morphology", "Centroid", "ROI Crop", "CLAHE (Final)"]
        for name in tab_names:
            tab = QLabel(f"Placeholder for {name}")
            tab.setAlignment(Qt.AlignCenter)
            tab.setStyleSheet("background: #f0f0f0;")
            self.tabs.addTab(tab, name)
        preview_card.layout().addWidget(self.tabs)
        content_layout.addWidget(preview_card)

        # ---- Visualisasi SIFT card ----
        viz_card = self._create_card("Visualisasi Pencocokan Fitur SIFT",
                                     "Peta relasi inlier descriptor antara citra uji (kiri) dan database (kanan).")
        self.viz_label = QLabel()
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setText("Hasil visualisasi SIFT akan ditampilkan di sini.\n"
                               "Jika gambar lebih tinggi dari area yang terlihat, scrollbar akan muncul.")
        self.viz_label.setWordWrap(True)
        self.viz_label.setStyleSheet("background: #e8f4fc; border: 1px solid #c0d6e8;")
        viz_card.layout().addWidget(self.viz_label)
        content_layout.addWidget(viz_card)

        content_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        content.setLayout(content_layout)
        main_layout.addWidget(content, stretch=1)

        self.setLayout(main_layout)

    def _create_card(self, title, subtitle):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setLineWidth(1)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(6)

        if title:
            title_lbl = QLabel(title)
            title_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
            card_layout.addWidget(title_lbl)
        if subtitle:
            subtitle_lbl = QLabel(subtitle)
            subtitle_lbl.setWordWrap(True)
            subtitle_lbl.setStyleSheet("color: #666;")
            card_layout.addWidget(subtitle_lbl)

        # Horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        card_layout.addWidget(line)

        card.setLayout(card_layout)
        return card


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi Verifikasi Palmprint – PyQt5 Scrollable Verification")
        self.resize(1000, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # Build verification page widget
        self.verification_page = VerificationPage()

        # Wrap it in QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)  # no border
        scroll_area.setStyleSheet("background: transparent;")
        scroll_area.setWidget(self.verification_page)

        h_layout.addWidget(scroll_area, stretch=1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())