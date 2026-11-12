"""Unit tests for CVEngine orchestrator, threading, and panic kill-switch."""

import time

import numpy as np

from animecam.config import AppConfig, CameraConfig, PipelineConfig
from animecam.core.engine import CVEngine


def test_engine_lifecycle_and_atomic_swap():
    """Verify that CVEngine runs in a daemon thread, paces frames, and provides atomic pointer swaps."""
    # Use synthetic camera and virtual cam disabled for testing
    cfg = AppConfig(
        camera=CameraConfig(device_index=-999, width=320, height=240, target_fps=30),
        pipeline=PipelineConfig(enable_virtual_cam=False),
    )
    engine = CVEngine(cfg)
    try:
        engine.start()
        assert engine._worker_thread is not None
        assert engine._worker_thread.is_alive()
        assert engine._worker_thread.daemon is True

        # Wait for at least one frame to be produced
        timeout = 2.0
        start = time.time()
        frame = None
        while time.time() - start < timeout:
            frame = engine.get_latest_frame()
            if frame is not None:
                break
            time.sleep(0.02)

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    finally:
        engine.stop()
        assert engine._worker_thread is None
        assert engine.camera.cap is None


def test_engine_panic_kill_switch():
    """Verify that panic_kill immediately cuts jutsu toggles and displays kill banner."""
    cfg = AppConfig(
        camera=CameraConfig(device_index=-999, width=320, height=240, target_fps=30),
        pipeline=PipelineConfig(enable_virtual_cam=False),
    )
    engine = CVEngine(cfg)
    try:
        engine.start()
        # Enable jutsu states
        engine.clones_enabled = True
        engine.eyes_enabled = True
        engine.invisibility_enabled = True

        # Trigger panic
        engine.panic_kill()
        assert engine.panic_active is True
        assert engine.clones_enabled is False
        assert engine.eyes_enabled is False
        assert engine.invisibility_enabled is False
        assert "PANIC" in engine.active_combo_name

        # Reset panic
        engine.reset_panic()
        assert engine.panic_active is False
        assert engine.active_combo_name == "Neutral"

    finally:
        engine.stop()
