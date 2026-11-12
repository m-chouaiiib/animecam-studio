"""Configuration module for AnimeCam Studio.

Provides validated dataclasses and constants governing camera acquisition,
frame scheduling, MediaPipe thresholds, vectorized effect parameters, and
CustomTkinter UI styling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------------------
# Filesystem Paths
# ------------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR: Path = BASE_DIR / "assets"
SHADERS_DIR: Path = ASSETS_DIR / "shaders"


@dataclass(frozen=True)
class CameraConfig:
    """Hardware acquisition and resolution parameters."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    fallback_width: int = 640
    fallback_height: int = 480
    api_preference: int = 0  # cv2.CAP_ANY by default


@dataclass(frozen=True)
class PipelineConfig:
    """Frame cadence, throttling, and memory bounds."""

    # Run heavy Hand gesture detection every N frames (15 Hz at 30 FPS video)
    # Cuts CPU utilization by ~40% while preserving responsive interaction
    gesture_cadence: int = 2

    # Ring buffer bounds for Shadow Clone Jutsu (O(1) memory bound)
    clone_buffer_maxlen: int = 24
    clone_delays: tuple[int, ...] = (8, 16)
    clone_alphas: tuple[float, ...] = (0.45, 0.65)
    clone_x_offsets: tuple[int, ...] = (-160, 160)

    # Invisibility / Stealth Mode (MediaPipe Selfie Segmentation)
    segmentation_threshold: float = 0.50
    ema_alpha: float = 0.05  # Background healing rate for non-person pixels
    background_warmup_frames: int = 20

    # Privacy & Safety
    clean_virtual_feed: bool = True  # Never send HUD/landmarks to virtual feed
    enable_virtual_cam: bool = True  # Can be disabled for standalone preview testing


@dataclass(frozen=True)
class MediaPipeConfig:
    """MediaPipe landmark detection confidence and topology."""

    min_hand_detection_confidence: float = 0.65
    min_hand_tracking_confidence: float = 0.65
    max_num_hands: int = 2

    # FaceMesh (only initialized lazily when Dojutsu / Eye effects are active)
    refine_face_landmarks: bool = True
    min_face_detection_confidence: float = 0.60
    min_face_tracking_confidence: float = 0.60

    # Key Landmark Indices for MediaPipe FaceMesh Iris tracking
    # Right eye
    right_iris_center: int = 468
    right_eye_inner: int = 133
    right_eye_outer: int = 33

    # Left eye
    left_iris_center: int = 473
    left_eye_inner: int = 362
    left_eye_outer: int = 263


@dataclass(frozen=True)
class EyeEffectConfig:
    """Dojutsu overlay scaling, rotation, and blend mode."""

    scale_factor: float = 1.35
    rotation_speed_deg: float = 3.5
    default_shader_name: str = "sharingan.png"


@dataclass(frozen=True)
class UIThemeConfig:
    """CustomTkinter dark-mode styling constants."""

    bg_dark: str = "#0D0D0F"
    bg_surface: str = "#16161B"
    bg_card: str = "#1E1E24"
    bg_card_hover: str = "#2A2A33"
    accent_orange: str = "#FF4925"
    accent_orange_hover: str = "#E03A18"
    text_primary: str = "#FFFFFF"
    text_muted: str = "#8E8E93"
    color_success: str = "#10B981"
    color_panic: str = "#EF4444"
    window_title: str = "AnimeCam Studio - Real-Time Neural Effects"
    default_window_size: str = "1280x820"


@dataclass(frozen=True)
class AppConfig:
    """Unified application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    eye: EyeEffectConfig = field(default_factory=EyeEffectConfig)
    theme: UIThemeConfig = field(default_factory=UIThemeConfig)


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()
