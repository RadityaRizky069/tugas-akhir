import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from modules.preprocessing import preprocess_and_extract_roi
from modules.feature_extraction import extract_sift_features
from modules.matching import match_features, make_decision
from modules import utils


class PalmprintGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Verifikasi Palmprint Berbasis SIFT")
        self.root.geometry("960x720")
        self.root.minsize(800, 600)

        self.bg = "#1e1e1e"
        self.fg = "#ffffff"
        self.accent = "#0078d4"
        self.secondary = "#2d2d2d"
        self.tertiary = "#3c3c3c"
        self.success = "#4caf50"
        self.error = "#f44336"
        self.warning = "#ff9800"

        self.root.configure(bg=self.bg)

        self._img_tk_ref = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=self.bg, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=self.secondary,
                        foreground=self.fg,
                        padding=[12, 4],
                        font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", self.accent)],
                  foreground=[("selected", self.fg)])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_reg = tk.Frame(self.notebook, bg=self.bg)
        self.notebook.add(self.tab_reg, text="  Registrasi  ")
        self._build_registration_tab()

        self.tab_ver = tk.Frame(self.notebook, bg=self.bg)
        self.notebook.add(self.tab_ver, text="  Verifikasi  ")
        self._build_verification_tab()

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    # ── Widget factory helpers ──────────────────────────────────────────

    def _make_entry(self, parent, width=35):
        return tk.Entry(parent, bg=self.secondary, fg=self.fg,
                        insertbackground=self.fg, relief=tk.FLAT,
                        font=("Segoe UI", 10), width=width)

    def _make_btn(self, parent, text, command, width=None):
        kwargs = dict(bg=self.accent, fg=self.fg, relief=tk.FLAT,
                      font=("Segoe UI", 10, "bold"), padx=16, pady=4,
                      cursor="hand2", command=command)
        if width:
            kwargs["width"] = width
        return tk.Button(parent, **kwargs)

    def _make_label(self, parent, text, **kwargs):
        defaults = dict(bg=self.bg, fg=self.fg, font=("Segoe UI", 10))
        defaults.update(kwargs)
        return tk.Label(parent, text=text, **defaults)

    # ── Registration Tab ────────────────────────────────────────────────

    def _build_registration_tab(self):
        form = tk.Frame(self.tab_reg, bg=self.bg)
        form.pack(fill=tk.X, padx=20, pady=(20, 5))

        self._make_label(form, "Username:").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.entry_username = self._make_entry(form, 35)
        self.entry_username.grid(row=0, column=1, sticky=tk.W, pady=6, padx=10)

        self._make_label(form, "Folder Foto:").grid(row=1, column=0, sticky=tk.W, pady=6)
        fsub = tk.Frame(form, bg=self.bg)
        fsub.grid(row=1, column=1, sticky=tk.W, pady=6, padx=10)
        self.entry_folder = self._make_entry(fsub, 28)
        self.entry_folder.pack(side=tk.LEFT)
        self._make_btn(fsub, "Browse", self._browse_folder).pack(side=tk.LEFT, padx=6)

        hint = self._make_label(form, "* Minimal 5 foto telapak tangan, latar belakang putih",
                                font=("Segoe UI", 8), fg=self.warning)
        hint.grid(row=2, column=1, sticky=tk.W, pady=2, padx=10)

        self.btn_register = self._make_btn(form, "  Daftarkan  ", self._start_registration)
        self.btn_register.grid(row=3, column=1, sticky=tk.W, pady=(12, 4), padx=10)

        sep = tk.Frame(self.tab_reg, bg=self.tertiary, height=1)
        sep.pack(fill=tk.X, padx=20, pady=6)

        log_container = tk.Frame(self.tab_reg, bg=self.bg)
        log_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(4, 10))

        self._make_label(log_container, "Log:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        txt_frame = tk.Frame(log_container, bg=self.secondary)
        txt_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        self.log_text = tk.Text(txt_frame, bg=self.secondary, fg=self.fg,
                                insertbackground=self.fg, relief=tk.FLAT,
                                font=("Consolas", 9), state=tk.DISABLED,
                                borderwidth=0, highlightthickness=0)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = tk.Scrollbar(txt_frame, command=self.log_text.yview, bg=self.tertiary)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb.set)

    # ── Verification Tab ────────────────────────────────────────────────

    def _build_verification_tab(self):
        form = tk.Frame(self.tab_ver, bg=self.bg)
        form.pack(fill=tk.X, padx=20, pady=(20, 5))

        self._make_label(form, "Pilih User:").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.combo_user = ttk.Combobox(form, values=[], state="readonly",
                                       font=("Segoe UI", 10), width=37)
        self.combo_user.grid(row=0, column=1, sticky=tk.W, pady=6, padx=10)

        self._make_label(form, "Gambar Uji:").grid(row=1, column=0, sticky=tk.W, pady=6)
        fsub = tk.Frame(form, bg=self.bg)
        fsub.grid(row=1, column=1, sticky=tk.W, pady=6, padx=10)
        self.entry_image = self._make_entry(fsub, 28)
        self.entry_image.pack(side=tk.LEFT)
        self._make_btn(fsub, "Browse", self._browse_image).pack(side=tk.LEFT, padx=6)

        self.btn_verify = self._make_btn(form, "  Verifikasi  ", self._start_verification)
        self.btn_verify.grid(row=2, column=1, sticky=tk.W, pady=(12, 4), padx=10)

        # Result area
        res_frame = tk.Frame(self.tab_ver, bg=self.bg)
        res_frame.pack(fill=tk.X, padx=20, pady=4)

        self.label_result = tk.Label(res_frame, text="", bg=self.bg, fg=self.fg,
                                     font=("Segoe UI", 16, "bold"))
        self.label_result.pack(anchor=tk.W)

        self.label_detail = tk.Label(res_frame, text="", bg=self.bg, fg=self.fg,
                                     font=("Segoe UI", 10))
        self.label_detail.pack(anchor=tk.W)

        sep = tk.Frame(self.tab_ver, bg=self.tertiary, height=1)
        sep.pack(fill=tk.X, padx=20, pady=6)

        # Canvas for match visualisation
        viz_frame = tk.Frame(self.tab_ver, bg=self.bg)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(4, 10))

        self._make_label(viz_frame, "Visualisasi Pencocokan (kiri: uji, kanan: database):",
                         font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        csub = tk.Frame(viz_frame, bg=self.secondary, relief=tk.FLAT, bd=0)
        csub.pack(fill=tk.BOTH, expand=True, pady=4)

        self.canvas_viz = tk.Canvas(csub, bg=self.secondary,
                                    highlightthickness=0, width=540, height=270)
        self.canvas_viz.pack(fill=tk.BOTH, expand=True)

    # ── Callbacks ───────────────────────────────────────────────────────

    def _on_tab_change(self, event):
        if self.notebook.index("current") == 1:
            self._refresh_user_list()

    def _refresh_user_list(self):
        users = utils.get_all_users()
        self.combo_user['values'] = users
        if users and not self.combo_user.get():
            self.combo_user.set(users[0])

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Pilih folder foto telapak tangan")
        if folder:
            self.entry_folder.delete(0, tk.END)
            self.entry_folder.insert(0, folder)

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar uji",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")])
        if path:
            self.entry_image.delete(0, tk.END)
            self.entry_image.insert(0, path)

    # ── Log (thread-safe via root.after) ────────────────────────────────

    def _log(self, msg):
        self.root.after(0, self._do_log, msg)

    def _do_log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ── Registration Flow ───────────────────────────────────────────────

    def _start_registration(self):
        username = self.entry_username.get().strip()
        folder = self.entry_folder.get().strip()

        if not username:
            messagebox.showerror("Error", "Username harus diisi!")
            return
        if not folder:
            messagebox.showerror("Error", "Folder harus dipilih!")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Folder tidak valid!")
            return

        self.btn_register.config(state=tk.DISABLED, text="Memproses...")
        self._log(f"Memulai registrasi untuk user '{username}' ...")

        threading.Thread(target=self._do_registration,
                         args=(username, folder), daemon=True).start()

    def _do_registration(self, username, folder):
        try:
            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(exts)])

            if len(files) < 5:
                self.root.after(0, self._reg_fail,
                                f"Minimal 5 foto diperlukan, ditemukan {len(files)}")
                return

            entries = []
            ok = fail = 0

            for i, fp in enumerate(files, 1):
                self._log(f"  [{i}/{len(files)}] {os.path.basename(fp)}")

                roi = preprocess_and_extract_roi(fp)
                if roi is None:
                    self._log("    Gagal: ROI tidak terekstraksi")
                    fail += 1
                    continue

                kp, desc = extract_sift_features(roi)
                if desc is None or len(kp) == 0:
                    self._log("    Gagal: tidak ada fitur SIFT")
                    fail += 1
                    continue

                entries.append({
                    'descriptors': desc,
                    'roi_image': roi,
                    'filename': os.path.basename(fp)
                })
                ok += 1
                self._log(f"    OK ({len(kp)} keypoints)")

            if ok < 5:
                self.root.after(0, self._reg_fail,
                                f"Hanya {ok} gambar berhasil (minimal 5)")
                return

            utils.register_user(username, entries)
            self.root.after(0, self._reg_ok, username, ok, fail)

        except Exception as e:
            self.root.after(0, self._reg_fail, str(e))

    def _reg_ok(self, username, ok, fail):
        self._log(f"Registrasi BERHASIL untuk '{username}' ({ok} ok, {fail} gagal)\n")
        self.btn_register.config(state=tk.NORMAL, text="  Daftarkan  ")
        self._refresh_user_list()
        messagebox.showinfo("Sukses",
                            f"User '{username}' terdaftar dengan {ok} data palmprint!")

    def _reg_fail(self, msg):
        self._log(f"Registrasi GAGAL: {msg}\n")
        self.btn_register.config(state=tk.NORMAL, text="  Daftarkan  ")
        messagebox.showerror("Error", msg)

    # ── Verification Flow ───────────────────────────────────────────────

    def _start_verification(self):
        user = self.combo_user.get()
        path = self.entry_image.get().strip()

        if not user:
            messagebox.showerror("Error", "Pilih user terlebih dahulu!")
            return
        if not path:
            messagebox.showerror("Error", "Pilih gambar uji!")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Error", "File gambar tidak ditemukan!")
            return

        self.btn_verify.config(state=tk.DISABLED, text="Memverifikasi...")
        self.label_result.config(text="")
        self.label_detail.config(text="")
        self.canvas_viz.delete("all")
        self._img_tk_ref = None

        threading.Thread(target=self._do_verification,
                         args=(user, path), daemon=True).start()

    def _do_verification(self, username, img_path):
        try:
            roi_test = preprocess_and_extract_roi(img_path)
            if roi_test is None:
                self.root.after(0, self._ver_fail,
                                "Gagal mengekstrak ROI dari gambar uji")
                return

            kp_test, desc_test = extract_sift_features(roi_test)
            if desc_test is None or len(kp_test) == 0:
                self.root.after(0, self._ver_fail,
                                "Tidak ada fitur SIFT pada gambar uji")
                return

            entries = utils.get_user_data(username)
            if not entries:
                self.root.after(0, self._ver_fail,
                                f"Tidak ada data untuk user '{username}'")
                return

            best_inliers = -1
            best_pct = 0.0
            best_inlier_matches = []
            best_kp_db = []
            best_roi_db = None

            for i, entry in enumerate(entries):
                roi_db = entry['roi_image']
                if roi_db is None or roi_db.size == 0:
                    continue

                kp_db, desc_db = extract_sift_features(roi_db)
                if desc_db is None or len(kp_db) == 0:
                    continue

                inliers, pct, inlier_m = match_features(
                    desc_test, desc_db, kp_test, kp_db)

                if inliers > best_inliers:
                    best_inliers = inliers
                    best_pct = pct
                    best_inlier_matches = inlier_m
                    best_kp_db = kp_db
                    best_roi_db = roi_db

            if best_inliers < 0:
                self.root.after(0, self._ver_fail,
                                "Tidak ada kecocokan dengan data tersimpan")
                return

            decision = make_decision(best_pct)

            viz = self._build_match_viz(roi_test, kp_test,
                                        best_roi_db, best_kp_db,
                                        best_inlier_matches)

            self.root.after(0, self._ver_done,
                            decision, best_pct, best_inliers, viz)

        except Exception as e:
            self.root.after(0, self._ver_fail, str(e))

    def _build_match_viz(self, img1, kp1, img2, kp2, matches):
        if img1 is None or img2 is None:
            return None
        if not kp1 or not kp2:
            return None

        i1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        i2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        try:
            result = cv2.drawMatches(
                i1, kp1, i2, kp2,
                matches if matches else [], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            h, w = result_rgb.shape[:2]
            TARGET_W, TARGET_H = 640, 360
            if w < TARGET_W or h < TARGET_H:
                result_rgb = cv2.resize(
                    result_rgb, (TARGET_W, TARGET_H),
                    interpolation=cv2.INTER_CUBIC)
            return result_rgb
        except cv2.error:
            return None

    def _ver_done(self, decision, pct, inliers, viz_rgb):
        if decision:
            self.label_result.config(text="AKSES DITERIMA", fg=self.success)
        else:
            self.label_result.config(text="AKSES DITOLAK", fg=self.error)

        self.label_detail.config(
            text=f"Skor kecocokan: {pct:.1f}%  ({inliers} inlier dari 10 maksimum)",
            fg=self.fg)

        if viz_rgb is not None:
            self._show_image(viz_rgb)

        self.btn_verify.config(state=tk.NORMAL, text="  Verifikasi  ")

    def _ver_fail(self, msg):
        self.label_result.config(text=f"GAGAL: {msg}", fg=self.error)
        self.btn_verify.config(state=tk.NORMAL, text="  Verifikasi  ")
        messagebox.showerror("Verifikasi Gagal", msg)

    # ── Image display on Canvas ─────────────────────────────────────────

    def _show_image(self, rgb):
        h, w = rgb.shape[:2]

        pil_img = Image.fromarray(rgb)
        self._img_tk_ref = ImageTk.PhotoImage(pil_img)
        self.canvas_viz.config(width=pil_img.width, height=pil_img.height)
        self.canvas_viz.create_image(0, 0, anchor=tk.NW, image=self._img_tk_ref)
