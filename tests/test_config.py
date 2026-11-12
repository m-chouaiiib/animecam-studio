"""Unit tests for AnimeCam Studio configuration module."""

from dataclasses import FrozenInstanceError

import pytest

from animecam.config import DEFAULT_CONFIG, AppConfig, CameraConfig


def test_default_config_types_and_defaults():
    """Verify that DEFAULT_CONFIG loads valid default instances."""
    assert isinstance(DEFAULT_CONFIG, AppConfig)
    assert DEFAULT_CONFIG.camera.width == 1280
    assert DEFAULT_CONFIG.camera.height == 720
    assert DEFAULT_CONFIG.camera.target_fps == 30
    assert DEFAULT_CONFIG.pipeline.gesture_cadence == 2
    assert DEFAULT_CONFIG.pipeline.clone_buffer_maxlen == 24


def test_config_immutability():
    """Ensure that config dataclasses are frozen and prevent accidental runtime mutations."""
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.camera.target_fps = 60  # type: ignore[misc]


def test_custom_camera_config():
    """Verify custom overriding of configuration instances."""
    custom_cam = CameraConfig(device_index=1, width=640, height=480, target_fps=60)
    assert custom_cam.device_index == 1
    assert custom_cam.width == 640
    assert custom_cam.height == 480
    assert custom_cam.target_fps == 60
