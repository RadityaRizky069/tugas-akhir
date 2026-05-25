import sys
import tkinter as tk

try:
    import cv2
except ImportError:
    print("=" * 60)
    print("  OpenCV tidak terinstall.")
    print("  Jalankan:  pip install opencv-contrib-python")
    print("=" * 60)
    sys.exit(1)

try:
    cv2.SIFT_create()
except AttributeError:
    print("=" * 60)
    print("  OpenCV SIFT tidak tersedia.")
    print("  Install opencv-contrib-python:")
    print("  pip install opencv-contrib-python")
    print("=" * 60)
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("=" * 60)
    print("  Pillow tidak terinstall.")
    print("  Jalankan:  pip install Pillow")
    print("=" * 60)
    sys.exit(1)

from gui.interface import PalmprintGUI


def main():
    root = tk.Tk()
    root.title("Sistem Verifikasi Palmprint Berbasis SIFT")
    app = PalmprintGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
