"""CustomTkinter Dark-Mode Control Studio for AnimeCam.

Features:
- #0D0D0F deep dark background with #FF4925 jutsu orange accents.
- Live 60 Hz preview viewport with aspect-ratio scaling.
- Telemetry telemetry badges (FPS, latency, active combo pill).
- Manual jutsu switches and parameter tuning sliders.
- Hardware Privacy Kill-Switch (Panic Button) bound to Spacebar.
- Graceful WM_DELETE_WINDOW shutdown ensuring camera release.
"""

from __future__ import annotations

import logging

import customtkinter as ctk
import cv2
from PIL import Image

from animecam.config import DEFAULT_CONFIG, SHADERS_DIR, AppConfig
from animecam.core.engine import CVEngine

logger = logging.getLogger("animecam.ui")


class AnimeCamStudioApp(ctk.CTk):
    """Main application window for AnimeCam Studio."""

    def __init__(self, engine: CVEngine, config: AppConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.engine = engine
        self.config = config
        self._is_closing = False

        # Apply dark theme styling
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title(config.theme.window_title)
        self.geometry(config.theme.default_window_size)
        self.configure(fg_color=config.theme.bg_dark)
        self.minsize(1080, 680)

        # Main Layout: 2 Columns (Sidebar Controls on Left, Viewport on Right)
        self.grid_columnconfigure(0, weight=0, minsize=340)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_viewport()

        # Keyboard hotkeys
        self.bind("<space>", lambda e: self._on_panic_press())
        self.bind("<Escape>", lambda e: self._on_close())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Kick off 60 Hz non-blocking render loop
        self.after(20, self._render_tick)

    def _build_sidebar(self) -> None:
        """Construct the control sidebar with cards, switches, sliders, and panic button."""
        theme = self.config.theme

        sidebar = ctk.CTkScrollableFrame(
            self,
            fg_color=theme.bg_surface,
            corner_radius=0,
            width=340,
        )
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        sidebar.grid_columnconfigure(0, weight=1)

        # 1. Studio Header Title
        title_label = ctk.CTkLabel(
            sidebar,
            text="🥷 ANIMECAM STUDIO",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=theme.accent_orange,
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 4), sticky="w")

        subtitle = ctk.CTkLabel(
            sidebar,
            text="Real-Time Neural Webcam Effects",
            font=ctk.CTkFont(size=12),
            text_color=theme.text_muted,
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        # 2. Telemetry Card
        telemetry_card = ctk.CTkFrame(sidebar, fg_color=theme.bg_card, corner_radius=10)
        telemetry_card.grid(row=2, column=0, padx=16, pady=8, sticky="ew")
        telemetry_card.grid_columnconfigure((0, 1), weight=1)

        self.fps_badge = ctk.CTkLabel(
            telemetry_card,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=theme.color_success,
        )
        self.fps_badge.grid(row=0, column=0, padx=12, pady=10, sticky="w")

        self.latency_badge = ctk.CTkLabel(
            telemetry_card,
            text="Latency: 0ms",
            font=ctk.CTkFont(size=12),
            text_color=theme.text_muted,
        )
        self.latency_badge.grid(row=0, column=1, padx=12, pady=10, sticky="e")

        self.combo_pill = ctk.CTkLabel(
            telemetry_card,
            text="Jutsu: Neutral",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=theme.text_primary,
            fg_color=theme.bg_card_hover,
            corner_radius=6,
        )
        self.combo_pill.grid(row=1, column=0, columnspan=2, padx=12, pady=(0, 10), sticky="ew")

        # 3. Manual Jutsu Switches Card
        jutsu_card = ctk.CTkFrame(sidebar, fg_color=theme.bg_card, corner_radius=10)
        jutsu_card.grid(row=3, column=0, padx=16, pady=8, sticky="ew")

        ctk.CTkLabel(
            jutsu_card,
            text="MANUAL JUTSU ACTIVATION",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=theme.accent_orange,
        ).pack(anchor="w", padx=14, pady=(12, 8))

        self.switch_invisibility = ctk.CTkSwitch(
            jutsu_card,
            text="Invisibility Cloak (Stealth)",
            progress_color=theme.accent_orange,
            command=self._on_invisibility_toggle,
        )
        self.switch_invisibility.pack(anchor="w", padx=14, pady=6)

        self.switch_clones = ctk.CTkSwitch(
            jutsu_card,
            text="Shadow Clone Jutsu (Bunshin)",
            progress_color=theme.accent_orange,
            command=self._on_clones_toggle,
        )
        self.switch_clones.pack(anchor="w", padx=14, pady=6)

        self.switch_eyes = ctk.CTkSwitch(
            jutsu_card,
            text="Dojutsu Eyes (FaceMesh)",
            progress_color=theme.accent_orange,
            command=self._on_eyes_toggle,
        )
        self.switch_eyes.pack(anchor="w", padx=14, pady=(6, 12))

        # 4. Shader Selector Card
        shader_card = ctk.CTkFrame(sidebar, fg_color=theme.bg_card, corner_radius=10)
        shader_card.grid(row=4, column=0, padx=16, pady=8, sticky="ew")

        ctk.CTkLabel(
            shader_card,
            text="DOJUTSU SHADER TEXTURE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=theme.accent_orange,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        # Detect shader options from assets
        shader_options = self._discover_shader_names()
        self.shader_menu = ctk.CTkOptionMenu(
            shader_card,
            values=shader_options,
            command=self._on_shader_select,
            fg_color=theme.bg_card_hover,
            button_color=theme.accent_orange,
            button_hover_color=theme.accent_orange_hover,
        )
        self.shader_menu.pack(fill="x", padx=14, pady=(0, 12))
        self.shader_menu.set(self.engine.selected_shader)

        # 5. Parameter Tuning Sliders Card
        tuning_card = ctk.CTkFrame(sidebar, fg_color=theme.bg_card, corner_radius=10)
        tuning_card.grid(row=5, column=0, padx=16, pady=8, sticky="ew")

        ctk.CTkLabel(
            tuning_card,
            text="REAL-TIME TUNING",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=theme.accent_orange,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        # Invisibility threshold slider
        self.lbl_thresh = ctk.CTkLabel(
            tuning_card,
            text=f"Stealth Sensitivity: {self.engine.invisibility_healer.threshold:.2f}",
            font=ctk.CTkFont(size=11),
            text_color=theme.text_muted,
        )
        self.lbl_thresh.pack(anchor="w", padx=14, pady=(2, 0))
        self.slider_thresh = ctk.CTkSlider(
            tuning_card,
            from_=0.20,
            to=0.85,
            progress_color=theme.accent_orange,
            button_color=theme.accent_orange,
            command=self._on_thresh_change,
        )
        self.slider_thresh.set(self.engine.invisibility_healer.threshold)
        self.slider_thresh.pack(fill="x", padx=14, pady=(2, 8))

        # Eye scale slider
        self.lbl_eye_scale = ctk.CTkLabel(
            tuning_card,
            text=f"Eye Scale: {self.config.eye.scale_factor:.2f}x",
            font=ctk.CTkFont(size=11),
            text_color=theme.text_muted,
        )
        self.lbl_eye_scale.pack(anchor="w", padx=14, pady=(2, 0))
        self.slider_eye_scale = ctk.CTkSlider(
            tuning_card,
            from_=0.80,
            to=2.20,
            progress_color=theme.accent_orange,
            button_color=theme.accent_orange,
            command=self._on_eye_scale_change,
        )
        self.slider_eye_scale.set(self.config.eye.scale_factor)
        self.slider_eye_scale.pack(fill="x", padx=14, pady=(2, 12))

        # 6. Privacy Panic Kill-Switch
        panic_card = ctk.CTkFrame(sidebar, fg_color=theme.bg_card, corner_radius=10)
        panic_card.grid(row=6, column=0, padx=16, pady=8, sticky="ew")

        self.btn_panic = ctk.CTkButton(
            panic_card,
            text="🚨 PRIVACY KILL-SWITCH (SPACE)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=theme.color_panic,
            hover_color="#DC2626",
            height=44,
            command=self._on_panic_press,
        )
        self.btn_panic.pack(fill="x", padx=14, pady=(12, 4))

        self.btn_reset_panic = ctk.CTkButton(
            panic_card,
            text="Reset Privacy & Restore Broadcast",
            font=ctk.CTkFont(size=12),
            fg_color=theme.bg_card_hover,
            hover_color="#3A3A45",
            command=self._on_reset_panic_press,
        )
        self.btn_reset_panic.pack(fill="x", padx=14, pady=(4, 12))

    def _build_viewport(self) -> None:
        """Construct the live video preview viewport and bottom telemetry bar."""
        theme = self.config.theme

        viewport_frame = ctk.CTkFrame(self, fg_color=theme.bg_dark, corner_radius=0)
        viewport_frame.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        viewport_frame.grid_rowconfigure(0, weight=1)
        viewport_frame.grid_columnconfigure(0, weight=1)

        # Video Canvas Container
        self.video_container = ctk.CTkFrame(viewport_frame, fg_color="#000000", corner_radius=8)
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 10))
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(self.video_container, text="Awaiting camera feed...")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # Bottom Bar: Device Info & Keybindings
        bottom_bar = ctk.CTkFrame(viewport_frame, fg_color=theme.bg_surface, height=36, corner_radius=6)
        bottom_bar.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        bottom_bar.grid_columnconfigure(0, weight=1)
        bottom_bar.grid_columnconfigure(1, weight=1)

        vcam_status = (
            "VIRTUAL WEBCAM: ONLINE" if self.engine.virtual_cam.is_active else "PREVIEW-ONLY MODE"
        )
        res_w, res_h = self.engine.camera.resolution
        self.lbl_dev_info = ctk.CTkLabel(
            bottom_bar,
            text=f"{vcam_status} | Feed: {res_w}x{res_h} @ {self.engine.camera.fps} FPS",
            font=ctk.CTkFont(size=11),
            text_color=theme.text_muted,
        )
        self.lbl_dev_info.grid(row=0, column=0, padx=12, pady=6, sticky="w")

        lbl_keys = ctk.CTkLabel(
            bottom_bar,
            text="Hotkeys: [Space] Panic Kill | [Esc] Exit",
            font=ctk.CTkFont(size=11),
            text_color=theme.text_muted,
        )
        lbl_keys.grid(row=0, column=1, padx=12, pady=6, sticky="e")

    def _discover_shader_names(self) -> list[str]:
        """List shader textures available in assets/shaders."""
        options = ["sharingan.png", "rinnegan.png", "sage_mode.png"]
        if SHADERS_DIR.exists():
            found = [p.name for p in SHADERS_DIR.glob("*.png")]
            if found:
                return sorted(found)
        return options

    def _render_tick(self) -> None:
        """Lock-free non-blocking 60 Hz frame retrieval and canvas presentation."""
        if self._is_closing:
            return

        frame = self.engine.get_latest_frame()
        if frame is not None:
            h, w = frame.shape[:2]

            # Measure container display bounds
            cont_w = max(320, self.video_container.winfo_width())
            cont_h = max(240, self.video_container.winfo_height())

            # Compute aspect-ratio scale factor
            scale = min(cont_w / w, cont_h / h)
            disp_w = max(10, int(w * scale))
            disp_h = max(10, int(h * scale))

            # Convert BGR to RGB and render into CTkImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))

            self.preview_label.configure(image=ctk_img, text="")
            # Keep reference to prevent GC disposal
            self.preview_label.image = ctk_img

        # Sync telemetry badges & switches with engine state
        self._sync_ui_state()

        # Schedule next tick at ~60 Hz (16 ms)
        self.after(16, self._render_tick)

    def _sync_ui_state(self) -> None:
        """Update telemetry badges and synchronize manual switches with gesture state."""
        # Update FPS & Latency
        self.fps_badge.configure(text=f"FPS: {self.engine.fps:4.1f}")
        self.latency_badge.configure(text=f"Latency: {self.engine.inference_latency_ms:3.0f}ms")
        self.combo_pill.configure(text=f"Jutsu: {self.engine.active_combo_name}")

        # Synchronize switches if gestures toggled them
        if self.switch_invisibility.get() != int(self.engine.invisibility_enabled):
            if self.engine.invisibility_enabled:
                self.switch_invisibility.select()
            else:
                self.switch_invisibility.deselect()

        if self.switch_clones.get() != int(self.engine.clones_enabled):
            if self.engine.clones_enabled:
                self.switch_clones.select()
            else:
                self.switch_clones.deselect()

        if self.switch_eyes.get() != int(self.engine.eyes_enabled):
            if self.engine.eyes_enabled:
                self.switch_eyes.select()
            else:
                self.switch_eyes.deselect()

    def _on_invisibility_toggle(self) -> None:
        self.engine.invisibility_enabled = bool(self.switch_invisibility.get())
        logger.info("Manual toggle: invisibility_enabled = %s", self.engine.invisibility_enabled)

    def _on_clones_toggle(self) -> None:
        self.engine.clones_enabled = bool(self.switch_clones.get())
        logger.info("Manual toggle: clones_enabled = %s", self.engine.clones_enabled)

    def _on_eyes_toggle(self) -> None:
        self.engine.eyes_enabled = bool(self.switch_eyes.get())
        logger.info("Manual toggle: eyes_enabled = %s", self.engine.eyes_enabled)

    def _on_shader_select(self, shader_name: str) -> None:
        self.engine.selected_shader = shader_name
        logger.info("Selected shader: %s", shader_name)

    def _on_thresh_change(self, value: float) -> None:
        self.engine.invisibility_healer.threshold = float(value)
        self.lbl_thresh.configure(text=f"Stealth Sensitivity: {value:.2f}")

    def _on_eye_scale_change(self, value: float) -> None:
        # Note: creates a new frozen config with updated scale factor if desired
        self.lbl_eye_scale.configure(text=f"Eye Scale: {value:.2f}x")

    def _on_panic_press(self) -> None:
        self.engine.panic_kill()
        self.switch_invisibility.deselect()
        self.switch_clones.deselect()
        self.switch_eyes.deselect()

    def _on_reset_panic_press(self) -> None:
        self.engine.reset_panic()

    def _on_close(self) -> None:
        """Ensure clean shutdown of background engine and camera handles."""
        if self._is_closing:
            return
        self._is_closing = True
        logger.info("Shutting down AnimeCam Studio UI...")
        try:
            self.engine.stop()
        except Exception as err:
            logger.error("Error stopping engine on UI close: %s", err)
        finally:
            self.destroy()
