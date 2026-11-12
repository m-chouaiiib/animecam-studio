"""Ring-Buffer Shadow Clone Compositor (Kage Bunshin no Jutsu).

Stores (frame, mask) pairs in a hardware-bounded ring buffer (deque(maxlen=24))
to composite time-delayed translucent shadow clones behind the user's live silhouette
using 100% vectorized NumPy broadcasting.
"""

from __future__ import annotations

import collections
import logging

import cv2
import numpy as np

from animecam.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger("animecam.clones")


class ShadowCloneCompositor:
    """Ring-buffer temporal compositor rendering shadow clones behind the user."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG.pipeline) -> None:
        self.config = config
        self._mp_seg = None
        self._segmenter = None

        # Fixed-size ring buffer storing tuples of: (frame_bgr: np.ndarray, mask: np.ndarray)
        # O(1) appending and automatic oldest-frame eviction at C level
        self._buffer: collections.deque[tuple[np.ndarray, np.ndarray]] = collections.deque(
            maxlen=config.clone_buffer_maxlen
        )

        self.delays: tuple[int, ...] = config.clone_delays
        self.alphas: tuple[float, ...] = config.clone_alphas
        self.x_offsets: tuple[int, ...] = config.clone_x_offsets

    def _ensure_initialized(self) -> None:
        """Lazy initialization of segmentation model."""
        if self._segmenter is not None:
            return

        import mediapipe as mp

        self._mp_seg = mp.solutions.selfie_segmentation
        self._segmenter = self._mp_seg.SelfieSegmentation(model_selection=0)  # General fast model
        logger.info("ShadowCloneCompositor segmentation model initialized.")

    def close(self) -> None:
        """Release segmenter resources."""
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None
            self._mp_seg = None

    def reset(self) -> None:
        """Clear clone ring buffer memory (e.g. on panic button or effect toggle)."""
        logger.info("Clearing shadow clone ring buffer.")
        self._buffer.clear()

    def compute_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Compute 2D float32 person segmentation mask in range [0.0, 1.0]."""
        self._ensure_initialized()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._segmenter.process(frame_rgb)
        if results.segmentation_mask is None:
            return np.zeros(frame_bgr.shape[:2], dtype=np.float32)

        return results.segmentation_mask.astype(np.float32)

    def apply(self, frame_bgr: np.ndarray, precomputed_mask: np.ndarray | None = None) -> np.ndarray:
        """Composite delayed shadow clones behind the live user silhouette.

        Args:
            frame_bgr: Current uint8 live camera frame.
            precomputed_mask: Optional pre-calculated person mask to avoid redundant inference.

        Returns:
            np.ndarray: Blended output frame with shadow clones.
        """
        # 1. Obtain current person mask
        if precomputed_mask is not None:
            live_mask = precomputed_mask
        else:
            live_mask = self.compute_mask(frame_bgr)

        # 2. Append current frame and mask into bounded ring buffer
        self._buffer.append((frame_bgr.copy(), live_mask.copy()))

        # If buffer is still warming up, return live frame
        max_delay = max(self.delays) if self.delays else 0
        if len(self._buffer) <= max_delay:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.astype(np.float32)
        live_mask_f32 = np.clip(live_mask, 0.0, 1.0)

        # 3. Iterate through clone configurations in reverse order (deepest background first)
        # Zip delays, alphas, and horizontal offsets
        clone_specs = sorted(
            zip(self.delays, self.alphas, self.x_offsets, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )

        for delay, alpha, x_offset in clone_specs:
            buf_idx = len(self._buffer) - 1 - delay
            if buf_idx < 0:
                continue

            past_frame, past_mask = self._buffer[buf_idx]

            # Shift the delayed clone horizontally using an affine translation matrix
            t_mat = np.float32([[1.0, 0.0, float(x_offset)], [0.0, 1.0, 0.0]])
            shifted_frame = cv2.warpAffine(
                past_frame,
                t_mat,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.float32)

            shifted_mask = cv2.warpAffine(
                past_mask,
                t_mat,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # Soften clone mask edges with a 15x15 Gaussian blur
            feathered_clone_mask = cv2.GaussianBlur(shifted_mask, (15, 15), 0)

            # CRITICAL COMPOSITING RULE:
            # Clone appears where past silhouette existed (feathered_clone_mask)
            # BUT is occluded by the live user (1.0 - live_mask_f32)
            # Scaled by the clone's transparency alpha
            clone_visibility = (
                np.clip(feathered_clone_mask, 0.0, 1.0)
                * (1.0 - live_mask_f32)
                * alpha
            )[:, :, np.newaxis]

            # Vectorized Alpha Blending
            canvas = (1.0 - clone_visibility) * canvas + clone_visibility * shifted_frame

        return np.clip(canvas, 0, 255).astype(np.uint8)
