# Sistem Verifikasi Palmprint Menggunakan Metode SIFT

Ekstraksi Ciri Telapak Tangan untuk Akses Kontrol Keamanan

## Deskripsi

Aplikasi GUI berbasis Python untuk memverifikasi apakah gambar palmprint (telapak tangan) uji cocok dengan gambar palmprint referensi menggunakan metode **SIFT (Scale-Invariant Feature Transform)**. Sistem melakukan ekstraksi fitur keypoint SIFT dan feature matching untuk menentukan hasil verifikasi akses.

### Metode yang Digunakan

1. **Preprocessing Citra**
   - Resize gambar ke ukuran standar (400x400)
   - Konversi ke grayscale
   - Peningkatan kontras dengan CLAHE
   - Reduksi noise dengan Gaussian Blur

2. **Ekstraksi Fitur SIFT**
   - Deteksi keypoint unik pada pola garis telapak tangan
   - Ekstraksi descriptor 128-dimensi untuk setiap keypoint

3. **Feature Matching**
   - BFMatcher dengan k-NN (k=2)
   - Lowe's ratio test (threshold 0.75)
   - Perhitungan skor kemiripan

## Persyaratan Sistem

- Python 3.8 atau lebih baru
- pip (Python package installer)
- Webcam (opsional, untuk fitur kamera)

## Instalasi

### 1. Clone / Download Project

```bash
cd palmprint_sift_project
```

### 2. Buat Virtual Environment (Direkomendasikan)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
python main.py
```

## Cara Menggunakan

1. **Upload Gambar Referensi**
   - Klik tombol "Upload Referensi"
   - Pilih gambar palmprint dari folder dataset atau folder lain

2. **Upload Gambar Uji**
   - Klik tombol "Upload Gambar Uji"
   - Pilih gambar palmprint yang akan diverifikasi

3. **Preprocessing Citra**
   - Klik tombol "Jalankan Preprocessing"
   - Sistem akan menampilkan hasil grayscale, kontras, dan reduksi noise

4. **Ekstraksi Fitur SIFT**
   - Klik tombol "Ekstraksi Fitur SIFT"
   - Sistem akan menampilkan keypoint pada gambar referensi dan uji

5. **Verifikasi Palmprint**
   - Klik tombol "Verifikasi Palmprint"
   - Sistem akan menampilkan status COCOK / TIDAK COCOK
   - Jumlah match, skor kemiripan, dan keputusan akses

## Dataset

Dataset dapat diunduh dari Kaggle:
[https://www.kaggle.com/code/lucy2233/palmprint-recognition-for-authentication/input?select=session1](https://www.kaggle.com/code/lucy2233/palmprint-recognition-for-authentication/input?select=session1)

Letakkan dataset pada folder `dataset/`:
```
dataset/
├── session1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── session2/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Aplikasi juga mendukung upload gambar manual dari folder manapun.

## Struktur Project

```
palmprint_sift_project/
│
├── main.py                 # GUI utama PyQt5
├── requirements.txt        # Dependensi Python
├── README.md               # Dokumentasi project
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Fungsi preprocessing citra
│   ├── sift_feature.py     # Ekstraksi fitur SIFT
│   ├── matcher.py          # Feature matching & verifikasi
│   └── utils.py            # Helper (cv2 -> QPixmap)
│
├── dataset/
│   ├── session1/           # Dataset sesi 1
│   └── session2/           # Dataset sesi 2
│
└── assets/
    └── placeholder.png     # Placeholder gambar
```

## Konfigurasi Threshold

Ubah nilai threshold di `src/matcher.py`:

```python
MATCH_THRESHOLD = 40    # Minimal good matches untuk status COCOK
RATIO_THRESHOLD = 0.75  # Threshold Lowe's ratio test
```

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-contrib-python
```

### Error: SIFT tidak tersedia
Pastikan menginstall `opencv-contrib-python`, bukan `opencv-python`.

### Error: PyQt5 tidak ditemukan
```bash
pip install PyQt5
```

## Lisensi

Project ini dibuat untuk tujuan edukasi mata kuliah Pengolahan Citra Digital.
