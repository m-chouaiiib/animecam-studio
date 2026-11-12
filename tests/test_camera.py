"""Unit tests for SafeCamera and SafeVirtualCamera context managers."""

import numpy as np

from animecam.config import CameraConfig
from animecam.core.camera import SafeCamera, SafeVirtualCamera


def test_safe_camera_synthetic_generation():
    """Verify that SafeCamera provides valid synthetic frames when device is unavailable."""
    cfg = CameraConfig(device_index=-999, width=640, height=360, target_fps=30)
    with SafeCamera(cfg) as cam:
        assert cam.is_synthetic is True
        success, frame = cam.read()
        assert success is True
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (360, 640, 3)
        assert frame.dtype == np.uint8
        assert cam.resolution == (640, 360)
        assert cam.fps == 30


def test_safe_camera_idempotent_release():
    """Verify that multiple release calls on SafeCamera are safe and idempotent."""
    cam = SafeCamera()
    cam.release()
    cam.release()
    assert cam.cap is None


def test_safe_virtual_camera_fallback():
    """Verify that SafeVirtualCamera handles environments without virtual webcam drivers gracefully."""
    with SafeVirtualCamera(width=640, height=480, fps=30, enabled=False) as vcam:
        assert vcam.is_active is False
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # send() must not raise any exceptions even when inactive
        vcam.send(dummy_frame)
