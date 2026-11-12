"""Live Background Healing Invisibility Cloak using MediaPipe Selfie Segmentation.

Erases the user from the webcam feed and continuously updates non-person pixels
using Exponential Moving Average (EMA) blending so dynamic room lighting shifts
look natural behind the invisibility cloak.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from animecam.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger("animecam.invisibility")


class LiveBackgroundHealer:
    """Continuous background estimation and vectorized invisibility compositing."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG.pipeline) -> None:
        self.config = config
        self._segmenter = None
        self._mp_seg = None

        # Continuous background model (stored in float32 for precision during EMA blending)
        self.clean_bg_f32: np.ndarray | None = None
        self._warmup_counter: int = 0
        self._warmup_target: int = config.background_warmup_frames
        self.ema_alpha: float = config.ema_alpha
        self.threshold: float = config.segmentation_threshold

    def _ensure_initialized(self) -> None:
        """Lazy initialization of MediaPipe Selfie Segmentation."""
        if self._segmenter is not None:
            return

        import mediapipe as mp

        self._mp_seg = mp.solutions.selfie_segmentation
        self._segmenter = self._mp_seg.SelfieSegmentation(model_selection=1)
        logger.info("MediaPipe Selfie Segmentation initialized (model_selection=1).")

    def close(self) -> None:
        """Release segmentation resources."""
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None
            self._mp_seg = None

    def reset(self) -> None:
        """Flush the learned background plate (e.g. on panic button or scene reset)."""
        logger.info("Resetting background healing model.")
        self.clean_bg_f32 = None
        self._warmup_counter = 0

    def compute_person_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Compute 2D float32 person probability mask in range [0.0, 1.0].

        Args:
            frame_bgr: uint8 BGR image.

        Returns:
            np.ndarray: float32 array of shape (H, W) with values [0.0, 1.0].
        """
        self._ensure_initialized()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._segmenter.process(frame_rgb)
        if results.segmentation_mask is None:
            return np.zeros(frame_bgr.shape[:2], dtype=np.float32)

        return results.segmentation_mask.astype(np.float32)

    def update_background(self, frame_bgr: np.ndarray, person_mask: np.ndarray) -> None:
        """Continuously update non-person background pixels via vectorized EMA blending.

        Formula:
            bg_pixel = (1 - alpha) * bg_pixel + alpha * frame_pixel
        """
        frame_f32 = frame_bgr.astype(np.float32)

        if self.clean_bg_f32 is None:
            self.clean_bg_f32 = frame_f32.copy()
            self._warmup_counter = 1
            return

        if self._warmup_counter < self._warmup_target:
            # During warmup, fast-learn the background
            self.clean_bg_f32 = 0.5 * self.clean_bg_f32 + 0.5 * frame_f32
            self._warmup_counter += 1
            return

        # Continuous healing: update ONLY non-person pixels (where mask < threshold)
        bg_selector = person_mask < self.threshold
        # Vectorized slice assignment (zero Python loops)
        self.clean_bg_f32[bg_selector] = (
            (1.0 - self.ema_alpha) * self.clean_bg_f32[bg_selector]
            + self.ema_alpha * frame_f32[bg_selector]
        )

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply the invisibility cloak effect over the live frame.

        Erase the user silhouette and composite the continuous healed background.
        """
        person_mask = self.compute_person_mask(frame_bgr)
        self.update_background(frame_bgr, person_mask)

        if self.clean_bg_f32 is None:
            return frame_bgr

        # Anti-aliasing feather: soft Gaussian blur over the 2D mask
        feathered_mask = cv2.GaussianBlur(person_mask, (21, 21), 0)
        alpha = np.clip(feathered_mask, 0.0, 1.0)[:, :, np.newaxis]

        # Vectorized Alpha Blending:
        # Where alpha is high (person), substitute healed clean_bg
        # Where alpha is low (background), preserve live frame
        healed_bg = np.clip(self.clean_bg_f32, 0, 255)
        composite = (alpha * healed_bg + (1.0 - alpha) * frame_bgr.astype(np.float32))
        return composite.astype(np.uint8)
