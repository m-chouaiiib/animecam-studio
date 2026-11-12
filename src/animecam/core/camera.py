"""Hardware-safe camera context managers for physical acquisition and virtual broadcasting.

Provides:
- SafeCamera: Context-managed cv2.VideoCapture with DirectShow/V4L2 auto-backend,
  hardware release guarantees, and synthetic test-pattern fallback.
- SafeVirtualCamera: Context-managed pyvirtualcam wrapper with driver auto-detection
  and graceful no-op fallback when virtual webcam drivers are missing.
"""

from __future__ import annotations

import logging
import math
import sys
import time

import cv2
import numpy as np

from animecam.config import DEFAULT_CONFIG, CameraConfig

logger = logging.getLogger("animecam.camera")


class SafeCamera:
    """Context-managed wrapper around cv2.VideoCapture.

    Guarantees hardware release (__exit__) to prevent locked OS video handles.
    Falls back to a synthetic procedural test pattern when no physical webcam is attached.
    """

    def __init__(self, config: CameraConfig = DEFAULT_CONFIG.camera) -> None:
        self.config = config
        self.cap: cv2.VideoCapture | None = None
        self.is_synthetic: bool = False
        self._frame_count: int = 0
        self._actual_width: int = config.width
        self._actual_height: int = config.height
        self._actual_fps: int = config.target_fps

    def __enter__(self) -> SafeCamera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def open(self) -> None:
        """Initialize the video capture device with platform-optimized backends."""
        if self.cap is not None and self.cap.isOpened():
            return

        # Platform-specific backend selection for fast negotiation
        backend = cv2.CAP_ANY
        if sys.platform.startswith("win"):
            # cv2.CAP_DSHOW prevents long DirectShow enumeration delays on Windows
            backend = cv2.CAP_DSHOW
        elif sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2

        logger.info(
            "Opening camera index %d with backend %s...",
            self.config.device_index,
            backend,
        )

        try:
            self.cap = cv2.VideoCapture(self.config.device_index, backend)
        except Exception as err:
            logger.warning("Failed to invoke VideoCapture: %s. Enabling synthetic fallback.", err)
            self.cap = None

        if self.cap is None or not self.cap.isOpened():
            logger.warning(
                "Physical webcam at index %d unavailable. Entering SYNTHETIC test mode.",
                self.config.device_index,
            )
            self.is_synthetic = True
            return

        # Configure hardware resolution and frame rate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        # Attempt MJPG FourCC for higher USB 2.0/3.0 bandwidth throughput if available
        fourcc = cv2.VideoWriter.fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Query negotiated properties
        ret_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ret_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        if ret_w > 0 and ret_h > 0:
            self._actual_width = ret_w
            self._actual_height = ret_h
        if ret_fps > 0:
            self._actual_fps = ret_fps

        logger.info(
            "Camera opened: %dx%d @ %d FPS (requested %dx%d @ %d)",
            self._actual_width,
            self._actual_height,
            self._actual_fps,
            self.config.width,
            self.config.height,
            self.config.target_fps,
        )

        # Warmup grab
        success, _ = self.cap.read()
        if not success:
            logger.warning(
                "Camera opened but failed initial frame read. Switching to synthetic mode."
            )
            self.is_synthetic = True

    def read(self) -> tuple[bool, np.ndarray]:
        """Read a BGR frame from the hardware or generate a synthetic test pattern.

        Returns:
            tuple[bool, np.ndarray]: (Success flag, uint8 BGR image of shape (H, W, 3))
        """
        if self.is_synthetic or self.cap is None or not self.cap.isOpened():
            return True, self._generate_synthetic_frame()

        success, frame = self.cap.read()
        if not success or frame is None:
            logger.warning("Hardware frame read failed. Yielding synthetic fallback frame.")
            return True, self._generate_synthetic_frame()

        self._frame_count += 1
        return True, frame

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Vectorized procedural test-pattern generator for headless/camera-less environments."""
        self._frame_count += 1
        h, w = self._actual_height, self._actual_width

        # Create dark anime studio backdrop
        frame = np.full((h, w, 3), (20, 18, 25), dtype=np.uint8)

        # Draw a moving synth grid line
        y_pos = int((self._frame_count * 4) % h)
        frame[y_pos : min(y_pos + 3, h), :, :] = (40, 200, 255)

        # Animated center target circle
        cx = int(w / 2 + math.sin(self._frame_count * 0.05) * (w * 0.15))
        cy = int(h / 2 + math.cos(self._frame_count * 0.05) * (h * 0.15))
        cv2.circle(frame, (cx, cy), 65, (37, 73, 255), -1)  # Jutsu orange core
        cv2.circle(frame, (cx, cy), 75, (255, 255, 255), 2)  # Halo outline

        # Synthetic telemetry watermark
        ts = time.strftime("%H:%M:%S")
        cv2.putText(
            frame,
            f"SYNTHETIC TEST FEED | FRAME: {self._frame_count:05d} | {ts}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (16, 185, 129),
            2,
            cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        """Explicitly release the camera hardware handle."""
        if self.cap is not None:
            logger.info("Releasing camera hardware handle...")
            try:
                self.cap.release()
            except Exception as err:
                logger.error("Error releasing VideoCapture handle: %s", err)
            finally:
                self.cap = None

    @property
    def resolution(self) -> tuple[int, int]:
        return self._actual_width, self._actual_height

    @property
    def fps(self) -> int:
        return self._actual_fps


class SafeVirtualCamera:
    """Context-managed wrapper around pyvirtualcam.

    Broadcasts processed video to system applications (OBS, Zoom, Discord).
    If no virtual camera driver is installed (e.g. OBS Virtualcam or v4l2loopback missing),
    it enters a harmless no-op mode without crashing the studio.
    """

    def __init__(
        self,
        width: int = DEFAULT_CONFIG.camera.width,
        height: int = DEFAULT_CONFIG.camera.height,
        fps: int = DEFAULT_CONFIG.camera.target_fps,
        enabled: bool = DEFAULT_CONFIG.pipeline.enable_virtual_cam,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.enabled = enabled
        self._cam = None
        self.is_active: bool = False

    def __enter__(self) -> SafeVirtualCamera:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def start(self) -> None:
        """Initialize pyvirtualcam if drivers are available."""
        if not self.enabled:
            logger.info("Virtual camera disabled by configuration.")
            return

        try:
            import pyvirtualcam

            self._cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            self.is_active = True
            logger.info(
                "Virtual camera online: device '%s' (%dx%d @ %d FPS)",
                self._cam.device,
                self.width,
                self.height,
                self.fps,
            )
        except Exception as err:
            logger.warning(
                "Could not initialize pyvirtualcam (%s). Running in PREVIEW-ONLY mode.",
                err,
            )
            self._cam = None
            self.is_active = False

    def send(self, frame_bgr: np.ndarray) -> None:
        """Transmit a BGR frame to the virtual camera device.

        Args:
            frame_bgr: uint8 NumPy array of shape (height, width, 3)
        """
        if not self.is_active or self._cam is None:
            return

        # Ensure correct resolution matches virtual camera initialization
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = cv2.resize(
                frame_bgr,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )

        try:
            self._cam.send(frame_bgr)
        except Exception as err:
            logger.error("Error sending frame to pyvirtualcam: %s", err)

    def close(self) -> None:
        """Close the virtual camera device node."""
        if self._cam is not None:
            logger.info("Closing virtual camera device...")
            try:
                self._cam.close()
            except Exception as err:
                logger.error("Error closing pyvirtualcam: %s", err)
            finally:
                self._cam = None
                self.is_active = False
