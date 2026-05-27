import tkinter as tk
from tkinter import filedialog, ttk
import customtkinter as ctk
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from modules.preprocessing import preprocess_and_extract_roi, preprocess_and_extract_roi_detailed
from modules.feature_extraction import extract_sift_features
from modules.matching import match_features, make_decision
from modules import utils

# ── Aesthetic & Typography Settings ─────────────────────────────────────
COLOR_BG          = "#F9FAFB"  # Light neutral gray background
COLOR_CARD        = "#FFFFFF"  # Pure white cards
COLOR_BORDER      = "#E5E7EB"  # Subtle gray borders
COLOR_TEXT_MAIN   = "#111827"  # Slate-900 for high readability
COLOR_TEXT_MUTED  = "#4B5563"  # Gray-600 for secondary labels
COLOR_ACCENT      = "#4F46E5"  # Indigo-600 primary color
COLOR_ACCENT_HVR  = "#4338CA"  # Indigo-700 hover state
COLOR_SUCCESS     = "#10B981"  # Emerald green success banner
COLOR_ERROR       = "#EF4444"  # Red error banner
COLOR_WARNING     = "#F59E0B"  # Amber warning accent
COLOR_LOG_BG      = "#F3F4F6"  # Light gray log background

# Modern, premium font family selection
FONT_FAMILY       = "Segoe UI"


# ════════════════════════════════════════════════════════════════════════
#  MODERN MESSAGE BOX (CUSTOM POPUP WINDOW)
# ════════════════════════════════════════════════════════════════════════

class ModernMessageBox(tk.Toplevel):
    """Custom popup window styled with the application's modern theme."""

    def __init__(self, parent, title, message, msg_type="info"):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x230")
        self.resizable(False, False)
        self.configure(bg=COLOR_BG)

        # Remove default system icon/styling to look unified
        self.transient(parent)
        self.grab_set()

        # Center placement relative to parent window
        self.update_idletasks()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()

        x = parent_x + (parent_w // 2) - 200
        y = parent_y + (parent_h // 2) - 115
        self.geometry(f"+{x}+{y}")

        # Core container card
        card = ctk.CTkFrame(self, fg_color=COLOR_CARD, corner_radius=12,
                            border_width=1, border_color=COLOR_BORDER)
        card.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Style icon based on popup type
        icon = "✨"
        color = COLOR_ACCENT
        if msg_type == "success":
            icon = "✓"
            color = COLOR_SUCCESS
        elif msg_type == "error":
            icon = "✗"
            color = COLOR_ERROR
        elif msg_type == "warning":
            icon = "⚠"
            color = COLOR_WARNING

        # Circular symbol badge
        icon_lbl = ctk.CTkLabel(card, text=icon, font=(FONT_FAMILY, 24, "bold"),
                                text_color=color, width=54, height=54,
                                corner_radius=27, fg_color=COLOR_LOG_BG)
        icon_lbl.pack(pady=(16, 6))

        # Title
        ctk.CTkLabel(card, text=title, font=(FONT_FAMILY, 14, "bold"),
                     text_color=COLOR_TEXT_MAIN).pack(pady=(4, 2))
                     
        # Body text message
        ctk.CTkLabel(card, text=message, font=(FONT_FAMILY, 11),
                     text_color=COLOR_TEXT_MUTED, wraplength=340, justify="center"
                     ).pack(fill=tk.X, padx=16, pady=(2, 16))

        # Action OK Button
        btn = ctk.CTkButton(card, text="OK", width=120, height=36, corner_radius=8,
                            fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HVR,
                            font=(FONT_FAMILY, 11, "bold"), command=self.destroy)
        btn.pack()


class PalmprintGUI:
    """State-of-the-art desktop GUI for SIFT Palmprint Verification."""

    def __init__(self, root):
        self.root = root
        self.root.title("PalmID — Sistem Verifikasi Telapak Tangan")
        self.root.geometry("1200x820")
        self.root.minsize(1020, 720)
        self.root.configure(bg=COLOR_BG)

        # Force clean light appearance mode matching modern SaaS aesthetics
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self._img_tk_ref = None
        self._preview_ref = None

        self._build_layout()
        self._show_frame("reg")  # Default frame

    # Helper method to trigger custom styled messagebox popup
    def _show_popup(self, title, message, msg_type="info"):
        ModernMessageBox(self.root, title, message, msg_type)

    # ════════════════════════════════════════════════════════════════════
    #  LAYOUT & NAVIGATION
    # ════════════════════════════════════════════════════════════════════

    def _build_layout(self):
        # Configure root grid columns: Column 0 is sidebar (fixed), Column 1 is content (stretches)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ── Column 0: Sidebar Navigation ────────────────────────────────
        self.sidebar = ctk.CTkFrame(self.root, width=250, corner_radius=0,
                                    fg_color="#0F172A", border_width=0) # Deep dark indigo/slate
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # App Logo/Header (Enlarged)
        self.logo_label = ctk.CTkLabel(self.sidebar, text="🖐  PalmID",
                                       font=(FONT_FAMILY, 24, "bold"),
                                       text_color="#FFFFFF")
        self.logo_label.pack(pady=(36, 4), padx=24, anchor="w")
        
        self.sub_logo = ctk.CTkLabel(self.sidebar, text="Biometric Verification System",
                                     font=(FONT_FAMILY, 11),
                                     text_color="#9CA3AF")
        self.sub_logo.pack(pady=(0, 36), padx=24, anchor="w")

        # Nav Buttons (Enlarged padding and text)
        self.btn_nav_reg = ctk.CTkButton(
            self.sidebar, text="  📋  Registrasi User",
            anchor="w", height=46, corner_radius=8,
            fg_color="transparent", text_color="#E5E7EB",
            hover_color="#1E293B", font=(FONT_FAMILY, 13, "bold"),
            command=lambda: self._show_frame("reg"))
        self.btn_nav_reg.pack(fill=tk.X, padx=14, pady=6)

        self.btn_nav_ver = ctk.CTkButton(
            self.sidebar, text="  🔍  Verifikasi Identitas",
            anchor="w", height=46, corner_radius=8,
            fg_color="transparent", text_color="#E5E7EB",
            hover_color="#1E293B", font=(FONT_FAMILY, 13, "bold"),
            command=lambda: self._show_frame("ver"))
        self.btn_nav_ver.pack(fill=tk.X, padx=14, pady=6)

        self.btn_nav_manage = ctk.CTkButton(
            self.sidebar, text="  ⚙  Kelola User",
            anchor="w", height=46, corner_radius=8,
            fg_color="transparent", text_color="#E5E7EB",
            hover_color="#1E293B", font=(FONT_FAMILY, 13, "bold"),
            command=lambda: self._show_frame("manage"))
        self.btn_nav_manage.pack(fill=tk.X, padx=14, pady=6)

        # Footer info
        footer = ctk.CTkLabel(self.sidebar, text="v1.0.0 — SIFT Engine",
                              font=(FONT_FAMILY, 10), text_color="#6B7280")
        footer.pack(side=tk.BOTTOM, pady=24)

        # ── Column 1: Main Content Container ────────────────────────────
        self.main_container = tk.Frame(self.root, bg=COLOR_BG)
        self.main_container.grid(row=0, column=1, sticky="nsew")

        # Build individual frames inside the main container
        self.frame_reg = tk.Frame(self.main_container, bg=COLOR_BG)
        self.frame_ver = tk.Frame(self.main_container, bg=COLOR_BG)
        self.frame_manage = tk.Frame(self.main_container, bg=COLOR_BG)
        
        self._init_registration_frame(self.frame_reg)
        self._init_verification_frame(self.frame_ver)
        self._init_manage_frame(self.frame_manage)

    def _show_frame(self, name):
        """Switches content panel and updates navigation button styling."""
        # Hide all frames first
        self.frame_reg.pack_forget()
        self.frame_ver.pack_forget()
        self.frame_manage.pack_forget()

        # Reset all nav buttons to inactive
        self.btn_nav_reg.configure(fg_color="transparent", text_color="#E5E7EB")
        self.btn_nav_ver.configure(fg_color="transparent", text_color="#E5E7EB")
        self.btn_nav_manage.configure(fg_color="transparent", text_color="#E5E7EB")

        if name == "reg":
            self.frame_reg.pack(fill=tk.BOTH, expand=True)
            self.btn_nav_reg.configure(fg_color=COLOR_ACCENT, text_color="#FFFFFF")
        elif name == "ver":
            self.frame_ver.pack(fill=tk.BOTH, expand=True)
            self.btn_nav_ver.configure(fg_color=COLOR_ACCENT, text_color="#FFFFFF")
            self._refresh_user_list()
        elif name == "manage":
            self.frame_manage.pack(fill=tk.BOTH, expand=True)
            self.btn_nav_manage.configure(fg_color=COLOR_ACCENT, text_color="#FFFFFF")
            self._refresh_manage_user_list()

    # ════════════════════════════════════════════════════════════════════
    #  UI COMPONENT HELPERS (CARDS)
    # ════════════════════════════════════════════════════════════════════

    def _create_card(self, parent, title=None, subtitle=None):
        """Creates a modern card widget with padding, borders, and rounded corners."""
        card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=12,
                            border_width=1, border_color=COLOR_BORDER)
        
        if title:
            header_frame = tk.Frame(card, bg=COLOR_CARD)
            header_frame.pack(fill=tk.X, padx=24, pady=(20, 10))
            
            ctk.CTkLabel(header_frame, text=title, font=(FONT_FAMILY, 15, "bold"),
                         text_color=COLOR_TEXT_MAIN, anchor="w").pack(fill=tk.X)
            if subtitle:
                ctk.CTkLabel(header_frame, text=subtitle, font=(FONT_FAMILY, 11),
                             text_color=COLOR_TEXT_MUTED, anchor="w").pack(fill=tk.X, pady=(4, 0))
            
            # Subtle divider line
            tk.Frame(card, bg=COLOR_BORDER, height=1).pack(fill=tk.X, padx=24, pady=(4, 12))
            
        return card

    def _create_header_banner(self, parent, title, desc):
        """Creates a header banner at the top of content views."""
        banner = tk.Frame(parent, bg=COLOR_BG)
        banner.pack(fill=tk.X, padx=28, pady=(28, 16))
        
        ctk.CTkLabel(banner, text=title, font=(FONT_FAMILY, 22, "bold"),
                     text_color=COLOR_TEXT_MAIN, anchor="w").pack(fill=tk.X)
        ctk.CTkLabel(banner, text=desc, font=(FONT_FAMILY, 12),
                     text_color=COLOR_TEXT_MUTED, anchor="w").pack(fill=tk.X, pady=(4, 0))

    # ════════════════════════════════════════════════════════════════════
    #  REGISTRATION VIEW
    # ════════════════════════════════════════════════════════════════════

    def _init_registration_frame(self, parent):
        self._create_header_banner(
            parent, "Registrasi User Baru",
            "Daftarkan data telapak tangan pengguna baru ke dalam database lokal."
        )

        # 2-Column layout inside registration
        cols = tk.Frame(parent, bg=COLOR_BG)
        cols.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        cols.columnconfigure(0, weight=1) # Left Column: Form
        cols.columnconfigure(1, weight=1) # Right Column: Log

        # Left Column: Form Input
        left_col = tk.Frame(cols, bg=COLOR_BG)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(6, 12))
        
        form_card = self._create_card(left_col, "Formulir Pendaftaran", 
                                      "Masukkan data identitas dan folder foto telapak tangan.")
        form_card.pack(fill=tk.BOTH, expand=True)

        frm = tk.Frame(form_card, bg=COLOR_CARD)
        frm.pack(fill=tk.X, padx=24, pady=12)

        # Username Input
        ctk.CTkLabel(frm, text="Username", font=(FONT_FAMILY, 12, "bold"),
                     text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(8, 6))
        self.entry_username = ctk.CTkEntry(
            frm, placeholder_text="Ketik username...", height=42, corner_radius=8,
            fg_color=COLOR_LOG_BG, border_color=COLOR_BORDER, text_color=COLOR_TEXT_MAIN,
            font=(FONT_FAMILY, 12))
        self.entry_username.pack(fill=tk.X, pady=(0, 16))

        # Folder Input
        ctk.CTkLabel(frm, text="Folder Dataset", font=(FONT_FAMILY, 12, "bold"),
                     text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(8, 6))
        
        fsub = tk.Frame(frm, bg=COLOR_CARD)
        fsub.pack(fill=tk.X, pady=(0, 16))
        
        self.entry_folder = ctk.CTkEntry(
            fsub, placeholder_text="Pilih direktori foto...", height=42, corner_radius=8,
            fg_color=COLOR_LOG_BG, border_color=COLOR_BORDER, text_color=COLOR_TEXT_MAIN,
            font=(FONT_FAMILY, 12))
        self.entry_folder.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ctk.CTkButton(fsub, text="Browse", width=90, height=42,
                      corner_radius=8, fg_color=COLOR_BORDER, text_color=COLOR_TEXT_MAIN,
                      hover_color="#D1D5DB", font=(FONT_FAMILY, 12, "bold"),
                      command=self._browse_folder).pack(side=tk.LEFT)

        # Info Box banner
        info_box = ctk.CTkFrame(frm, fg_color="#EEF2F6", corner_radius=8, border_width=0)
        info_box.pack(fill=tk.X, pady=(10, 24))
        
        ctk.CTkLabel(info_box, text="ℹ  Persyaratan Registrasi:\n• Minimal 5 file foto (.jpg, .png, .bmp)\n• Pastikan background foto bersih / abu-abu 18%",
                     font=(FONT_FAMILY, 11), text_color=COLOR_TEXT_MUTED, justify="left", anchor="w"
                     ).pack(padx=16, pady=12, fill=tk.X)

        # Register Button
        self.btn_register = ctk.CTkButton(
            frm, text="Mulai Registrasi", height=46,
            corner_radius=8, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HVR,
            font=(FONT_FAMILY, 12, "bold"), command=self._start_registration)
        self.btn_register.pack(fill=tk.X)

        # Right Column: Process Logs
        right_col = tk.Frame(cols, bg=COLOR_BG)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(12, 6))

        log_card = self._create_card(right_col, "Log Hasil Proses", 
                                     "Output status eksekusi preprocessing dan ekstraksi fitur.")
        log_card.pack(fill=tk.BOTH, expand=True)

        self.log_text = ctk.CTkTextbox(
            log_card, fg_color=COLOR_LOG_BG, text_color=COLOR_TEXT_MAIN,
            font=("Consolas", 11), state="disabled", corner_radius=8,
            border_width=1, border_color=COLOR_BORDER)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=24, pady=(10, 24))

    # ════════════════════════════════════════════════════════════════════
    #  VERIFICATION VIEW
    # ════════════════════════════════════════════════════════════════════

    def _init_verification_frame(self, parent):
        self._create_header_banner(
            parent, "Verifikasi Telapak Tangan (1:1)",
            "Verifikasi kesesuaian citra telapak tangan dengan data reference di database."
        )

        # Main 2-column grid
        cols = tk.Frame(parent, bg=COLOR_BG)
        cols.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        cols.columnconfigure(0, weight=4) # Left Column: Controls & Result
        cols.columnconfigure(1, weight=6) # Right Column: Visual Preview & Matching

        # ── Left Column: Controls & Banner Result ───────────────────────
        left_col = tk.Frame(cols, bg=COLOR_BG)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(6, 12))

        # Input Card
        inp_card = self._create_card(left_col, "Identitas & Gambar Uji", 
                                     "Pilih user terdaftar dan upload file gambar telapak tangan.")
        inp_card.pack(fill=tk.X, pady=(0, 16))

        frm = tk.Frame(inp_card, bg=COLOR_CARD)
        frm.pack(fill=tk.X, padx=24, pady=12)

        # Select User Combo
        ctk.CTkLabel(frm, text="Pilih User", font=(FONT_FAMILY, 12, "bold"),
                     text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(6, 6))
        
        self.combo_user = ctk.CTkComboBox(
            frm, height=42, corner_radius=8,
            fg_color=COLOR_LOG_BG, border_color=COLOR_BORDER, text_color=COLOR_TEXT_MAIN,
            button_color=COLOR_BORDER, button_hover_color="#D1D5DB",
            dropdown_fg_color=COLOR_CARD, dropdown_text_color=COLOR_TEXT_MAIN,
            dropdown_hover_color=COLOR_LOG_BG,
            font=(FONT_FAMILY, 12), state="readonly", values=[])
        self.combo_user.pack(fill=tk.X, pady=(0, 16))

        # Image Test File Input
        ctk.CTkLabel(frm, text="Gambar Uji", font=(FONT_FAMILY, 12, "bold"),
                     text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(6, 6))
        
        isub = tk.Frame(frm, bg=COLOR_CARD)
        isub.pack(fill=tk.X, pady=(0, 20))
        
        self.entry_image = ctk.CTkEntry(
            isub, placeholder_text="Pilih gambar uji...", height=42, corner_radius=8,
            fg_color=COLOR_LOG_BG, border_color=COLOR_BORDER, text_color=COLOR_TEXT_MAIN,
            font=(FONT_FAMILY, 12))
        self.entry_image.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ctk.CTkButton(isub, text="Browse", width=90, height=42,
                      corner_radius=8, fg_color=COLOR_BORDER, text_color=COLOR_TEXT_MAIN,
                      hover_color="#D1D5DB", font=(FONT_FAMILY, 12, "bold"),
                      command=self._browse_image).pack(side=tk.LEFT)

        # Verify Button
        self.btn_verify = ctk.CTkButton(
            frm, text="Verifikasi Sekarang", height=46,
            corner_radius=8, fg_color=COLOR_ACCENT, hover_color=COLOR_ACCENT_HVR,
            font=(FONT_FAMILY, 12, "bold"), command=self._start_verification)
        self.btn_verify.pack(fill=tk.X, pady=(4, 0))

        # Result Banner Card
        self.result_card = self._create_card(left_col, "Status Hasil Verifikasi",
                                             "Keputusan akhir sistem pencocokan biometrik.")
        self.result_card.pack(fill=tk.BOTH, expand=True)

        res_inner = tk.Frame(self.result_card, bg=COLOR_CARD)
        res_inner.pack(fill=tk.BOTH, expand=True, padx=24, pady=12)

        self.label_result = tk.Label(res_inner, text="Menunggu Input Verifikasi",
                                     bg=COLOR_CARD, fg=COLOR_TEXT_MUTED,
                                     font=(FONT_FAMILY, 20, "bold"), anchor="w")
        self.label_result.pack(fill=tk.X, pady=(4, 6))
        
        self.label_detail = tk.Label(res_inner, text="Pilih data dan klik tombol 'Verifikasi Sekarang' untuk mencocokkan keypoints.", 
                                     bg=COLOR_CARD, fg=COLOR_TEXT_MUTED, 
                                     font=(FONT_FAMILY, 12), anchor="w", justify="left", wraplength=380)
        self.label_detail.pack(fill=tk.X)

        # ── Right Column: Visual Elements ───────────────────────────────
        right_col = tk.Frame(cols, bg=COLOR_BG)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(12, 6))

        # Preview Card (Upgraded to multi-stage Preprocessing visualizer)
        prev_card = self._create_card(right_col, "Tahapan Preprocessing Citra Uji", 
                                     "Tampilan interaktif hasil setiap langkah algoritma pengolahan citra.")
        prev_card.pack(fill=tk.X, pady=(0, 16))

        self.preview_tabs = ctk.CTkTabview(
            prev_card, fg_color=COLOR_CARD, segmented_button_selected_color=COLOR_ACCENT,
            segmented_button_selected_hover_color=COLOR_ACCENT_HVR,
            segmented_button_unselected_color=COLOR_LOG_BG,
            segmented_button_unselected_hover_color=COLOR_BORDER,
            text_color=COLOR_TEXT_MAIN
        )
        self.preview_tabs.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 20))
        try:
            self.preview_tabs._segmented_button.configure(font=(FONT_FAMILY, 10, "bold"))
        except Exception:
            pass

        self.tabs_list = [
            "Asli", "Grayscale", "Gaussian", "Threshold", 
            "Morphology", "Centroid", "ROI Crop", "CLAHE (Final)"
        ]
        self.canvas_dict = {}
        for tab_name in self.tabs_list:
            tab_ref = self.preview_tabs.add(tab_name)
            frame = tk.Frame(tab_ref, bg=COLOR_LOG_BG, bd=1, relief=tk.SOLID, highlightbackground=COLOR_BORDER)
            frame.pack(pady=4)
            canvas = tk.Canvas(frame, bg=COLOR_LOG_BG, highlightthickness=0, width=440, height=330)
            canvas.pack()
            self.canvas_dict[tab_name] = canvas

        self._preview_tk_refs = {}
        self._preview_placeholder()

        # Match Visualization Card
        viz_card = self._create_card(right_col, "Visualisasi Pencocokan Fitur SIFT", 
                                     "Peta relasi inlier descriptor antara citra uji (kiri) dan database (kanan).")
        viz_card.pack(fill=tk.BOTH, expand=True)

        viz_frame = tk.Frame(viz_card, bg=COLOR_LOG_BG, bd=1, relief=tk.SOLID, highlightbackground=COLOR_BORDER)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=(6, 24))

        self.canvas_viz = tk.Canvas(viz_frame, bg=COLOR_LOG_BG,
                                    highlightthickness=0,
                                    width=560, height=280)
        self.canvas_viz.pack(fill=tk.BOTH, expand=True)

    # ════════════════════════════════════════════════════════════════════
    #  PREVIEW PLACEHOLDERS
    # ════════════════════════════════════════════════════════════════════

    def _preview_placeholder(self):
        for name, canvas in self.canvas_dict.items():
            canvas.delete("all")
            canvas.create_text(
                220, 165, text="Menunggu Gambar Uji...",
                fill=COLOR_TEXT_MUTED, font=(FONT_FAMILY, 11))

    def _update_preprocessing_previews(self, path):
        try:
            roi, steps = preprocess_and_extract_roi_detailed(path)
            if steps is None:
                self._preview_placeholder()
                try:
                    img = Image.open(path)
                    img.thumbnail((440, 330), Image.LANCZOS)
                    self._preview_tk_refs["Asli"] = ImageTk.PhotoImage(img)
                    canvas = self.canvas_dict["Asli"]
                    canvas.delete("all")
                    canvas.config(width=img.width, height=img.height)
                    canvas.create_image(img.width // 2, img.height // 2, anchor=tk.CENTER,
                                        image=self._preview_tk_refs["Asli"])
                except Exception:
                    pass
                return

            self._preview_tk_refs.clear()
            mapping = {
                "Asli": "original",
                "Grayscale": "grayscale",
                "Gaussian": "blurred",
                "Threshold": "thresh",
                "Morphology": "morph",
                "Centroid": "centroid",
                "ROI Crop": "roi_crop",
                "CLAHE (Final)": "roi_clahe"
            }

            for tab_name, step_key in mapping.items():
                if step_key in steps:
                    img_data = steps[step_key]
                    img = Image.fromarray(img_data)
                    img.thumbnail((440, 330), Image.LANCZOS)
                    self._preview_tk_refs[tab_name] = ImageTk.PhotoImage(img)
                    canvas = self.canvas_dict[tab_name]
                    canvas.delete("all")
                    canvas.config(width=img.width, height=img.height)
                    canvas.create_image(img.width // 2, img.height // 2, anchor=tk.CENTER,
                                        image=self._preview_tk_refs[tab_name])
        except Exception as e:
            print("Error updating previews:", e)
            self._preview_placeholder()

    # ════════════════════════════════════════════════════════════════════
    #  CALLBACKS (Unchanged backend logic hooks)
    # ════════════════════════════════════════════════════════════════════

    def _on_tab_switch(self):
        if self.tabview.get() == "Verifikasi":
            self._refresh_user_list()

    def _refresh_user_list(self):
        users = utils.get_all_users()
        self.combo_user.configure(values=users)
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
            self._update_preprocessing_previews(path)

    # ════════════════════════════════════════════════════════════════════
    #  THREAD-SAFE LOGGING
    # ════════════════════════════════════════════════════════════════════

    def _log(self, msg):
        self.root.after(0, self._do_log, msg)

    def _do_log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    # ════════════════════════════════════════════════════════════════════
    #  REGISTRATION PIPELINE (Logic untouched)
    # ════════════════════════════════════════════════════════════════════

    def _start_registration(self):
        username = self.entry_username.get().strip()
        folder = self.entry_folder.get().strip()

        if not username:
            self._show_popup("Peringatan", "Username harus diisi!", "warning")
            return
        if not folder:
            self._show_popup("Peringatan", "Folder dataset harus dipilih!", "warning")
            return
        if not os.path.isdir(folder):
            self._show_popup("Error", "Direktori folder yang dipilih tidak valid!", "error")
            return

        self.btn_register.configure(state="disabled", text="Memproses...")
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
                    self._log("    ✗ Gagal: ROI tidak terekstraksi")
                    fail += 1
                    continue

                kp, desc = extract_sift_features(roi)
                if desc is None or len(kp) == 0:
                    self._log("    ✗ Gagal: tidak ada fitur SIFT")
                    fail += 1
                    continue

                entries.append({
                    'descriptors': desc,
                    'roi_image': roi,
                    'filename': os.path.basename(fp)
                })
                ok += 1
                self._log(f"    ✓ OK ({len(kp)} keypoints)")

            if ok < 5:
                self.root.after(0, self._reg_fail,
                                f"Hanya {ok} gambar valid (minimal 5)")
                return

            utils.register_user(username, entries)
            self.root.after(0, self._reg_ok, username, ok, fail)

        except Exception as e:
            self.root.after(0, self._reg_fail, str(e))

    def _reg_ok(self, username, ok, fail):
        self._log(f"✓ Registrasi BERHASIL untuk '{username}' ({ok} ok, {fail} gagal)\n")
        self.btn_register.configure(state="normal", text="Mulai Registrasi")
        self._refresh_user_list()
        self._show_popup("Sukses", f"User '{username}' terdaftar dengan {ok} data palmprint!", "success")

    def _reg_fail(self, msg):
        self._log(f"✗ Registrasi GAGAL: {msg}\n")
        self.btn_register.configure(state="normal", text="Mulai Registrasi")
        self._show_popup("Registrasi Gagal", msg, "error")

    # ════════════════════════════════════════════════════════════════════
    #  VERIFICATION PIPELINE (Logic untouched)
    # ════════════════════════════════════════════════════════════════════

    def _start_verification(self):
        user = self.combo_user.get()
        path = self.entry_image.get().strip()

        if not user:
            self._show_popup("Peringatan", "Pilih user terlebih dahulu!", "warning")
            return
        if not path:
            self._show_popup("Peringatan", "Pilih gambar uji yang akan diverifikasi!", "warning")
            return
        if not os.path.isfile(path):
            self._show_popup("Error", "File gambar uji tidak ditemukan!", "error")
            return

        self.btn_verify.configure(state="disabled", text="Memverifikasi...")
        self.label_result.config(text="Memproses verifikasi...", fg=COLOR_TEXT_MUTED)
        self.label_detail.config(text="")
        self.canvas_viz.delete("all")
        self._img_tk_ref = None

        threading.Thread(target=self._do_verification,
                         args=(user, path), daemon=True).start()

    def _do_verification(self, username, img_path):
        try:
            roi_test, steps_test = preprocess_and_extract_roi_detailed(img_path)
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
            best_filename = ""

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
                    best_filename = entry.get('filename', f"Template {i+1}")

            if best_inliers < 0:
                self.root.after(0, self._ver_fail,
                                "Tidak ada kecocokan dengan data tersimpan")
                return

            decision = make_decision(best_pct)

            viz = self._build_match_viz(roi_test, kp_test,
                                        best_roi_db, best_kp_db,
                                        best_inlier_matches)

            self.root.after(0, self._ver_done,
                            decision, best_pct, best_inliers, viz, best_filename, username)

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
            TARGET_W, TARGET_H = 560, 280
            scale = min(TARGET_W / w, TARGET_H / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            result_rgb = cv2.resize(result_rgb, (new_w, new_h),
                                    interpolation=cv2.INTER_AREA)
            return result_rgb
        except cv2.error:
            return None

    def _ver_done(self, decision, pct, inliers, viz_rgb, matched_filename, username):
        if decision:
            self.label_result.config(text="✓  AKSES DITERIMA", fg=COLOR_SUCCESS)
            detail_text = (
                f"Autentikasi Berhasil!\n"
                f"• Skor Kecocokan: {pct:.1f}% ({inliers} Inlier Terverifikasi RANSAC)\n"
                f"• Database User: '{username}'\n"
                f"• File Referensi Cocok: '{matched_filename}'\n"
                f"• Analisis Teknis: Skor kemiripan geometri ini berhasil melampaui threshold minimal 50.0%."
            )
            self.label_detail.config(text=detail_text, fg=COLOR_TEXT_MAIN)
            self._show_popup("Verifikasi Berhasil", f"Akses diterima untuk user '{username}'!", "success")
        else:
            self.label_result.config(text="✗  AKSES DITOLAK", fg=COLOR_ERROR)
            detail_text = (
                f"Autentikasi Gagal!\n"
                f"• Skor Kecocokan: {pct:.1f}% ({inliers} Inlier Terverifikasi RANSAC)\n"
                f"• Database User: '{username}'\n"
                f"• File Referensi Cocok: '{matched_filename}'\n"
                f"• Analisis Teknis: Skor kemiripan geometri berada di bawah batas minimal threshold keamanan verifikasi 50.0%."
            )
            self.label_detail.config(text=detail_text, fg=COLOR_TEXT_MAIN)
            self._show_popup("Verifikasi Gagal", f"Akses ditolak untuk user '{username}'.", "error")

        if viz_rgb is not None:
            self._show_viz(viz_rgb)

        self.btn_verify.configure(state="normal", text="Verifikasi Sekarang")

    def _ver_fail(self, msg):
        self.label_result.config(text=f"⚠  {msg}", fg=COLOR_ERROR)
        self.label_detail.config(text="Silakan cek kembali kesesuaian gambar input telapak tangan Anda.")
        self.btn_verify.configure(state="normal", text="Verifikasi Sekarang")
        self._show_popup("Verifikasi Gagal", msg, "error")

    # ════════════════════════════════════════════════════════════════════
    #  IMAGE RENDER
    # ════════════════════════════════════════════════════════════════════

    def _show_viz(self, rgb):
        pil_img = Image.fromarray(rgb)
        self._img_tk_ref = ImageTk.PhotoImage(pil_img)
        self.canvas_viz.config(width=pil_img.width, height=pil_img.height)
        self.canvas_viz.delete("all")
        self.canvas_viz.create_image(0, 0, anchor=tk.NW, image=self._img_tk_ref)

    # ════════════════════════════════════════════════════════════════════
    #  MANAGE USER VIEW
    # ════════════════════════════════════════════════════════════════════

    def _init_manage_frame(self, parent):
        self._create_header_banner(
            parent, "Kelola User",
            "Lihat, hapus, atau ubah folder dataset pengguna yang terdaftar."
        )

        # Main content area
        content = tk.Frame(parent, bg=COLOR_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # ── Left side: User table ─────────────────────────────────────
        table_card = self._create_card(content, "Daftar User Terdaftar",
                                       "Pilih user dari tabel di bawah untuk mengelola data.")
        table_card.grid(row=0, column=0, sticky="nsew", padx=6)

        # Toolbar row (refresh + action buttons)
        toolbar = tk.Frame(table_card, bg=COLOR_CARD)
        toolbar.pack(fill=tk.X, padx=24, pady=(4, 10))

        ctk.CTkButton(
            toolbar, text="🔄  Refresh", width=110, height=36, corner_radius=8,
            fg_color=COLOR_LOG_BG, text_color=COLOR_TEXT_MAIN,
            hover_color=COLOR_BORDER, font=(FONT_FAMILY, 11, "bold"),
            command=self._refresh_manage_user_list
        ).pack(side=tk.LEFT, padx=(0, 8))

        ctk.CTkButton(
            toolbar, text="📁  Ubah Folder Dataset", width=180, height=36, corner_radius=8,
            fg_color="#3B82F6", text_color="#FFFFFF",
            hover_color="#2563EB", font=(FONT_FAMILY, 11, "bold"),
            command=self._manage_change_folder
        ).pack(side=tk.LEFT, padx=(0, 8))

        ctk.CTkButton(
            toolbar, text="🗑  Hapus User", width=130, height=36, corner_radius=8,
            fg_color=COLOR_ERROR, text_color="#FFFFFF",
            hover_color="#DC2626", font=(FONT_FAMILY, 11, "bold"),
            command=self._manage_delete_user
        ).pack(side=tk.LEFT)

        # Treeview (user table)
        tree_frame = tk.Frame(table_card, bg=COLOR_CARD)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 24))

        # Style the treeview to match the app theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Manage.Treeview",
                        background=COLOR_CARD,
                        foreground=COLOR_TEXT_MAIN,
                        fieldbackground=COLOR_CARD,
                        font=(FONT_FAMILY, 11),
                        rowheight=38,
                        borderwidth=0)
        style.configure("Manage.Treeview.Heading",
                        background=COLOR_LOG_BG,
                        foreground=COLOR_TEXT_MAIN,
                        font=(FONT_FAMILY, 11, "bold"),
                        borderwidth=1,
                        relief="flat")
        style.map("Manage.Treeview",
                  background=[("selected", "#E0E7FF")],
                  foreground=[("selected", COLOR_ACCENT)])

        columns = ("no", "username", "jumlah_data")
        self.manage_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings",
            style="Manage.Treeview", selectmode="browse")

        self.manage_tree.heading("no", text="No")
        self.manage_tree.heading("username", text="Username")
        self.manage_tree.heading("jumlah_data", text="Jumlah Data Palmprint")

        self.manage_tree.column("no", width=50, anchor="center", stretch=False)
        self.manage_tree.column("username", width=250, anchor="w")
        self.manage_tree.column("jumlah_data", width=180, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                                  command=self.manage_tree.yview)
        self.manage_tree.configure(yscrollcommand=scrollbar.set)

        self.manage_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Log area for manage operations
        log_card = self._create_card(content, "Log Operasi",
                                     "Output status operasi kelola user.")
        log_card.grid(row=1, column=0, sticky="nsew", padx=6, pady=(16, 0))
        content.rowconfigure(1, weight=0)

        self.manage_log_text = ctk.CTkTextbox(
            log_card, fg_color=COLOR_LOG_BG, text_color=COLOR_TEXT_MAIN,
            font=("Consolas", 11), state="disabled", corner_radius=8,
            border_width=1, border_color=COLOR_BORDER, height=150)
        self.manage_log_text.pack(fill=tk.X, padx=24, pady=(10, 24))

    def _manage_log(self, msg):
        self.root.after(0, self._do_manage_log, msg)

    def _do_manage_log(self, msg):
        self.manage_log_text.configure(state="normal")
        self.manage_log_text.insert(tk.END, msg + "\n")
        self.manage_log_text.see(tk.END)
        self.manage_log_text.configure(state="disabled")

    def _refresh_manage_user_list(self):
        """Reload the user table from the database."""
        for item in self.manage_tree.get_children():
            self.manage_tree.delete(item)

        db = utils.load_database()
        for idx, (username, entries) in enumerate(db.items(), 1):
            count = len(entries) if isinstance(entries, list) else 0
            self.manage_tree.insert("", tk.END, values=(idx, username, f"{count} file"))

    def _get_selected_manage_user(self):
        """Return the username of the currently selected row in the treeview."""
        sel = self.manage_tree.selection()
        if not sel:
            self._show_popup("Peringatan", "Pilih user dari tabel terlebih dahulu!", "warning")
            return None
        values = self.manage_tree.item(sel[0], "values")
        return values[1]  # username column

    # ── Delete User ───────────────────────────────────────────────────

    def _manage_delete_user(self):
        username = self._get_selected_manage_user()
        if not username:
            return
        # Show a confirmation dialog
        ConfirmDeleteDialog(self.root, username, self._confirm_delete_user)

    def _confirm_delete_user(self, username):
        success = utils.delete_user(username)
        if success:
            self._manage_log(f"✓ User '{username}' berhasil dihapus dari database.")
            self._refresh_manage_user_list()
            self._show_popup("Sukses", f"User '{username}' telah dihapus.", "success")
        else:
            self._manage_log(f"✗ Gagal menghapus user '{username}' (tidak ditemukan).")
            self._show_popup("Error", f"User '{username}' tidak ditemukan di database.", "error")

    # ── Change Folder Dataset ─────────────────────────────────────────

    def _manage_change_folder(self):
        username = self._get_selected_manage_user()
        if not username:
            return

        folder = filedialog.askdirectory(
            title=f"Pilih folder dataset baru untuk '{username}'")
        if not folder:
            return
        if not os.path.isdir(folder):
            self._show_popup("Error", "Direktori tidak valid!", "error")
            return

        self._manage_log(f"Memulai update dataset untuk user '{username}' dari folder: {folder}")
        self._show_popup("Proses", f"Memulai re-registrasi dataset untuk '{username}'...\nProses berjalan di background.", "info")

        threading.Thread(target=self._do_change_folder,
                         args=(username, folder), daemon=True).start()

    def _do_change_folder(self, username, folder):
        """Re-register user with new folder dataset (runs in background thread)."""
        try:
            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(exts)])

            if len(files) < 5:
                self.root.after(0, self._change_folder_fail, username,
                                f"Minimal 5 foto diperlukan, ditemukan {len(files)}")
                return

            entries = []
            ok = fail = 0

            for i, fp in enumerate(files, 1):
                self._manage_log(f"  [{i}/{len(files)}] {os.path.basename(fp)}")

                roi = preprocess_and_extract_roi(fp)
                if roi is None:
                    self._manage_log("    ✗ Gagal: ROI tidak terekstraksi")
                    fail += 1
                    continue

                kp, desc = extract_sift_features(roi)
                if desc is None or len(kp) == 0:
                    self._manage_log("    ✗ Gagal: tidak ada fitur SIFT")
                    fail += 1
                    continue

                entries.append({
                    'descriptors': desc,
                    'roi_image': roi,
                    'filename': os.path.basename(fp)
                })
                ok += 1
                self._manage_log(f"    ✓ OK ({len(kp)} keypoints)")

            if ok < 5:
                self.root.after(0, self._change_folder_fail, username,
                                f"Hanya {ok} gambar valid (minimal 5)")
                return

            utils.register_user(username, entries)
            self.root.after(0, self._change_folder_ok, username, ok, fail)

        except Exception as e:
            self.root.after(0, self._change_folder_fail, username, str(e))

    def _change_folder_ok(self, username, ok, fail):
        self._manage_log(f"✓ Dataset '{username}' berhasil diperbarui ({ok} ok, {fail} gagal)\n")
        self._refresh_manage_user_list()
        self._show_popup("Sukses", f"Dataset user '{username}' diperbarui dengan {ok} data palmprint!", "success")

    def _change_folder_fail(self, username, msg):
        self._manage_log(f"✗ Update dataset '{username}' GAGAL: {msg}\n")
        self._show_popup("Update Gagal", msg, "error")


# ════════════════════════════════════════════════════════════════════════
#  CONFIRMATION DELETE DIALOG
# ════════════════════════════════════════════════════════════════════════

class ConfirmDeleteDialog(tk.Toplevel):
    """Modern confirmation dialog for deleting a user."""

    def __init__(self, parent, username, on_confirm):
        super().__init__(parent)
        self.title("Konfirmasi Hapus")
        self.geometry("420x260")
        self.resizable(False, False)
        self.configure(bg=COLOR_BG)
        self._username = username
        self._on_confirm = on_confirm

        self.transient(parent)
        self.grab_set()

        # Center on parent
        self.update_idletasks()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        x = px + (pw // 2) - 210
        y = py + (ph // 2) - 130
        self.geometry(f"+{x}+{y}")

        card = ctk.CTkFrame(self, fg_color=COLOR_CARD, corner_radius=12,
                            border_width=1, border_color=COLOR_BORDER)
        card.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Warning icon
        icon_lbl = ctk.CTkLabel(card, text="⚠", font=(FONT_FAMILY, 28, "bold"),
                                text_color=COLOR_ERROR, width=58, height=58,
                                corner_radius=29, fg_color="#FEE2E2")
        icon_lbl.pack(pady=(18, 8))

        ctk.CTkLabel(card, text="Hapus User?",
                     font=(FONT_FAMILY, 15, "bold"),
                     text_color=COLOR_TEXT_MAIN).pack(pady=(2, 4))

        ctk.CTkLabel(card, text=f"Yakin ingin menghapus user '{username}'?\nSemua data palmprint akan dihapus permanen.",
                     font=(FONT_FAMILY, 11), text_color=COLOR_TEXT_MUTED,
                     wraplength=340, justify="center").pack(padx=16, pady=(0, 18))

        btn_row = tk.Frame(card, bg=COLOR_CARD)
        btn_row.pack(pady=(0, 10))

        ctk.CTkButton(btn_row, text="Batal", width=120, height=36, corner_radius=8,
                      fg_color=COLOR_LOG_BG, text_color=COLOR_TEXT_MAIN,
                      hover_color=COLOR_BORDER, font=(FONT_FAMILY, 11, "bold"),
                      command=self.destroy).pack(side=tk.LEFT, padx=(0, 10))

        ctk.CTkButton(btn_row, text="Hapus", width=120, height=36, corner_radius=8,
                      fg_color=COLOR_ERROR, text_color="#FFFFFF",
                      hover_color="#DC2626", font=(FONT_FAMILY, 11, "bold"),
                      command=self._do_confirm).pack(side=tk.LEFT)

    def _do_confirm(self):
        parent = self.master
        self.destroy()
        parent.after(100, lambda: self._on_confirm(self._username))
