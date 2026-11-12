"""Iris-Locked Dojutsu Overlay (Sharingan, Rinnegan, Sage Mode).

Tracks iris landmarks (468, 473) using MediaPipe FaceMesh (refine_landmarks=True)
and performs real-time vectorized alpha-blending of rotating 4-channel BGRA textures
dynamically scaled to eye width.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from animecam.config import DEFAULT_CONFIG, SHADERS_DIR, AppConfig

logger = logging.getLogger("animecam.eyes")


class DojutsuOverlay:
    """Iris-tracking and rotating BGRA eye shader compositing engine."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._mp_face_mesh = None
        self._face_mesh = None

        # Texture cache: mapping shader filename to 4-channel uint8 BGRA array
        self._shaders: dict[str, np.ndarray] = {}
        self._current_angle_deg: float = 0.0

        # Pre-load available shader textures
        self._load_shaders(SHADERS_DIR)

    def _load_shaders(self, shaders_dir: Path) -> None:
        """Cache all transparent 4-channel PNG shaders into memory."""
        if not shaders_dir.exists():
            logger.warning("Shaders directory not found at: %s", shaders_dir)
            return

        for png_path in shaders_dir.glob("*.png"):
            # Load with cv2.IMREAD_UNCHANGED to preserve the 4th Alpha channel
            tex = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if tex is not None and len(tex.shape) == 3 and tex.shape[2] == 4:
                self._shaders[png_path.name] = tex
                logger.info("Loaded eye shader: '%s' (shape: %s)", png_path.name, tex.shape)
            else:
                logger.warning("Skipping shader '%s' (missing 4-channel BGRA data)", png_path.name)

    def _ensure_initialized(self) -> None:
        """Lazy initialization of MediaPipe FaceMesh."""
        if self._face_mesh is not None:
            return

        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=self.config.mediapipe.refine_face_landmarks,
            min_detection_confidence=self.config.mediapipe.min_face_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_face_tracking_confidence,
        )
        logger.info("MediaPipe FaceMesh initialized (refine_landmarks=True).")

    def close(self) -> None:
        """Release FaceMesh resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            self._mp_face_mesh = None

    def apply(self, frame_bgr: np.ndarray, shader_name: str | None = None) -> np.ndarray:
        """Track irises and overlay rotating anime eye textures.

        Args:
            frame_bgr: Current uint8 camera frame.
            shader_name: Name of shader file (e.g. 'sharingan.png').

        Returns:
            np.ndarray: Frame with blended eye overlays.
        """
        self._ensure_initialized()

        active_shader_name = shader_name or self.config.eye.default_shader_name
        texture = self._shaders.get(active_shader_name)

        # Fallback to any loaded shader if requested one is not found
        if texture is None and self._shaders:
            texture = next(iter(self._shaders.values()))

        if texture is None:
            return frame_bgr

        # FaceMesh inference
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        face_lms = results.multi_face_landmarks[0].landmark

        # Advance rotation angle continuously for animated iris spin
        self._current_angle_deg = (
            self._current_angle_deg + self.config.eye.rotation_speed_deg
        ) % 360.0

        # Output canvas (mutated in-place on target regions)
        canvas = frame_bgr.copy()

        # Define left and right eye landmark tuples:
        # (center_idx, outer_corner_idx, inner_corner_idx)
        cfg_mp = self.config.mediapipe
        eye_pairs = [
            # Right eye
            (cfg_mp.right_iris_center, cfg_mp.right_eye_outer, cfg_mp.right_eye_inner),
            # Left eye
            (cfg_mp.left_iris_center, cfg_mp.left_eye_outer, cfg_mp.left_eye_inner),
        ]

        for center_idx, outer_idx, inner_idx in eye_pairs:
            center_lm = face_lms[center_idx]
            outer_lm = face_lms[outer_idx]
            inner_lm = face_lms[inner_idx]

            cx = int(center_lm.x * w)
            cy = int(center_lm.y * h)

            # Eye width in pixels
            eye_width = math.hypot(
                (outer_lm.x - inner_lm.x) * w,
                (outer_lm.y - inner_lm.y) * h,
            )

            # Target diameter scaled to eye width
            diameter = max(12, int(eye_width * self.config.eye.scale_factor))
            self._blend_shader_onto_iris(canvas, texture, cx, cy, diameter, self._current_angle_deg)

        return canvas

    def _blend_shader_onto_iris(
        self,
        canvas: np.ndarray,
        texture_bgra: np.ndarray,
        cx: int,
        cy: int,
        diameter: int,
        angle_deg: float,
    ) -> None:
        """Vectorized alpha-blending of rotating BGRA texture with boundary clipping."""
        h, w = canvas.shape[:2]
        radius = diameter // 2

        # Bounding box coordinates on canvas
        x1 = cx - radius
        y1 = cy - radius
        x2 = x1 + diameter
        y2 = y1 + diameter

        # Check bounds: completely off screen
        if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
            return

        # 1. Resize texture to target diameter
        resized_tex = cv2.resize(texture_bgra, (diameter, diameter), interpolation=cv2.INTER_LINEAR)

        # 2. Rotate resized texture around center
        rot_mat = cv2.getRotationMatrix2D((radius, radius), angle_deg, 1.0)
        rotated_tex = cv2.warpAffine(
            resized_tex,
            rot_mat,
            (diameter, diameter),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # 3. Clip intersection between bounding box and canvas
        canvas_x1 = max(0, x1)
        canvas_y1 = max(0, y1)
        canvas_x2 = min(w, x2)
        canvas_y2 = min(h, y2)

        # Corresponding region inside the rotated texture
        tex_x1 = canvas_x1 - x1
        tex_y1 = canvas_y1 - y1
        tex_x2 = tex_x1 + (canvas_x2 - canvas_x1)
        tex_y2 = tex_y1 + (canvas_y2 - canvas_y1)

        tex_roi = rotated_tex[tex_y1:tex_y2, tex_x1:tex_x2]
        canvas_roi = canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2]

        if tex_roi.shape[0] == 0 or tex_roi.shape[1] == 0:
            return

        # 4. Vectorized Alpha Compositing
        # Extract BGR channels and normalized float Alpha channel
        shader_bgr = tex_roi[:, :, :3].astype(np.float32)
        shader_alpha = (tex_roi[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
        bg_bgr = canvas_roi.astype(np.float32)

        # Alpha Blend: out = alpha * fg + (1 - alpha) * bg
        blended = shader_alpha * shader_bgr + (1.0 - shader_alpha) * bg_bgr
        canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = blended.astype(np.uint8)
