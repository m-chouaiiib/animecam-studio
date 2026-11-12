"""Unit tests for vectorized visual effects: Invisibility, Clones, and Dojutsu Eyes."""

import numpy as np

from animecam.config import DEFAULT_CONFIG
from animecam.effects.clones import ShadowCloneCompositor
from animecam.effects.eyes import DojutsuOverlay
from animecam.effects.invisibility import LiveBackgroundHealer


def test_background_healer_ema_and_mask_gating():
    """Verify that background healer updates background ONLY where person is absent."""
    healer = LiveBackgroundHealer(DEFAULT_CONFIG.pipeline)
    try:
        h, w = 120, 160
        # Frame 1: initial background plate (gray: 100)
        bg_frame = np.full((h, w, 3), 100, dtype=np.uint8)
        mask_empty = np.zeros((h, w), dtype=np.float32)

        # Initialize background plate
        healer.update_background(bg_frame, mask_empty)
        assert healer.clean_bg_f32 is not None

        # Warm up
        for _ in range(healer._warmup_target + 1):
            healer.update_background(bg_frame, mask_empty)

        # Frame 2: Person (bright white 255) enters center square [40:80, 60:100]
        # Background changes slightly from 100 to 120 (lighting change)
        new_frame = np.full((h, w, 3), 120, dtype=np.uint8)
        new_frame[40:80, 60:100] = 255

        person_mask = np.zeros((h, w), dtype=np.float32)
        person_mask[40:80, 60:100] = 1.0  # Person present in center

        # Update background
        healer.update_background(new_frame, person_mask)

        # Verification:
        # In person region [40:80, 60:100], background must NOT have absorbed the 255 white person pixels!
        assert np.all(healer.clean_bg_f32[40:80, 60:100] < 125)

        # In non-person region [0:10, 0:10], background must have adapted towards 120
        assert healer.clean_bg_f32[0, 0, 0] > 100.0

        # Invisibility application test
        output = healer.apply(new_frame)
        assert isinstance(output, np.ndarray)
        assert output.shape == (h, w, 3)
        assert output.dtype == np.uint8

    finally:
        healer.close()


def test_shadow_clone_ring_buffer_bounds_and_occlusion():
    """Verify ring-buffer O(1) memory bounding and occlusion behind live silhouette."""
    compositor = ShadowCloneCompositor(DEFAULT_CONFIG.pipeline)
    try:
        h, w = 120, 160
        # Feed 35 frames into ring buffer (maxlen = 24)
        for i in range(35):
            frame = np.full((h, w, 3), i, dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.float32)
            mask[40:80, 60:100] = 1.0  # Center silhouette
            compositor.apply(frame, precomputed_mask=mask)

        # Verify ring buffer never exceeds maxlen (zero memory leak guarantee)
        assert len(compositor._buffer) == compositor.config.clone_buffer_maxlen

        # Test clone compositing
        live_frame = np.full((h, w, 3), 50, dtype=np.uint8)
        live_mask = np.zeros((h, w), dtype=np.float32)
        live_mask[40:80, 60:100] = 1.0  # User is in center

        out = compositor.apply(live_frame, precomputed_mask=live_mask)
        assert isinstance(out, np.ndarray)
        assert out.shape == (h, w, 3)
        assert out.dtype == np.uint8

        # Live user center region must remain crisp
        assert np.all(out[50, 70] == 50)

    finally:
        compositor.close()


def test_dojutsu_overlay_shader_blending_and_clipping():
    """Verify iris shader rotation and boundary clipping."""
    overlay = DojutsuOverlay(DEFAULT_CONFIG)
    try:
        # Check that shaders are loaded from assets
        assert len(overlay._shaders) > 0
        assert "sharingan.png" in overlay._shaders

        # Test boundary clipping when iris is close to the image edge
        h, w = 200, 200
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        shader = overlay._shaders["sharingan.png"]

        # Center close to top-left corner (-10, -10) -> partial clipping
        overlay._blend_shader_onto_iris(canvas, shader, cx=-10, cy=-10, diameter=60, angle_deg=45.0)

        # Center inside canvas (100, 100) -> full render
        overlay._blend_shader_onto_iris(canvas, shader, cx=100, cy=100, diameter=50, angle_deg=90.0)

        # Check that canvas has red pixels from Sharingan iris
        assert canvas[:, :, 2].max() > 180  # Crimson red channel active

        # Test full apply on synthetic frame (no faces detected)
        out = overlay.apply(canvas, shader_name="sharingan.png")
        assert out.shape == (h, w, 3)

    finally:
        overlay.close()
