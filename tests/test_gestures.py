"""Unit tests for scale-invariant Hand Gesture recognition and Combo FSM."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from animecam.core.gestures import HandGestureFSM, HandSign, SingleHandData


def create_mock_landmark(x: float, y: float, z: float = 0.0):
    """Helper to mock a MediaPipe NormalizedLandmark object."""
    mock_lm = MagicMock()
    mock_lm.x = x
    mock_lm.y = y
    mock_lm.z = z
    return mock_lm


def create_mock_hand_landmarks(scale: float = 1.0, is_tiger: bool = False, is_fist: bool = False):
    """Generate 21 mock landmarks with variable scale to test scale-invariance."""
    # Base wrist at (0.5, 0.8)
    wrist_x, wrist_y = 0.5, 0.8
    # Middle MCP at (0.5, 0.5) => palm_scale = 0.3 * scale
    palm_len = 0.3 * scale

    lms = []
    # 0: Wrist
    lms.append(create_mock_landmark(wrist_x, wrist_y))

    # 1..4: Thumb
    lms.append(create_mock_landmark(wrist_x - 0.05 * scale, wrist_y - 0.08 * scale))
    lms.append(create_mock_landmark(wrist_x - 0.09 * scale, wrist_y - 0.15 * scale))
    lms.append(create_mock_landmark(wrist_x - 0.15 * scale, wrist_y - 0.20 * scale))
    if is_fist:
        lms.append(create_mock_landmark(wrist_x - 0.04 * scale, wrist_y - 0.18 * scale))  # 4: Tip curled
    else:
        lms.append(create_mock_landmark(wrist_x - 0.22 * scale, wrist_y - 0.22 * scale))  # 4: Tip extended

    # 5..8: Index
    idx_mcp_y = wrist_y - palm_len * 0.95
    lms.append(create_mock_landmark(wrist_x - 0.04 * scale, idx_mcp_y))  # 5: MCP
    lms.append(create_mock_landmark(wrist_x - 0.04 * scale, idx_mcp_y - 0.08 * scale))  # 6: PIP
    lms.append(create_mock_landmark(wrist_x - 0.04 * scale, idx_mcp_y - 0.15 * scale))  # 7: DIP
    if is_fist:
        # Curled back toward palm
        lms.append(create_mock_landmark(wrist_x - 0.04 * scale, idx_mcp_y - 0.03 * scale))  # 8: Tip curled
    else:
        # Extended upwards
        lms.append(create_mock_landmark(wrist_x - 0.04 * scale, idx_mcp_y - 0.25 * scale))  # 8: Tip extended

    # 9..12: Middle
    mid_mcp_y = wrist_y - palm_len
    lms.append(create_mock_landmark(wrist_x, mid_mcp_y))  # 9: MCP
    lms.append(create_mock_landmark(wrist_x, mid_mcp_y - 0.09 * scale))  # 10: PIP
    lms.append(create_mock_landmark(wrist_x, mid_mcp_y - 0.17 * scale))  # 11: DIP
    if is_fist:
        lms.append(create_mock_landmark(wrist_x, mid_mcp_y - 0.04 * scale))  # 12: Tip curled
    else:
        lms.append(create_mock_landmark(wrist_x, mid_mcp_y - 0.28 * scale))  # 12: Tip extended

    # 13..16: Ring
    rng_mcp_y = wrist_y - palm_len * 0.92
    lms.append(create_mock_landmark(wrist_x + 0.04 * scale, rng_mcp_y))  # 13: MCP
    lms.append(create_mock_landmark(wrist_x + 0.04 * scale, rng_mcp_y - 0.07 * scale))  # 14: PIP
    lms.append(create_mock_landmark(wrist_x + 0.04 * scale, rng_mcp_y - 0.13 * scale))  # 15: DIP
    if is_tiger or is_fist:
        lms.append(create_mock_landmark(wrist_x + 0.04 * scale, rng_mcp_y - 0.02 * scale))  # 16: Tip curled
    else:
        lms.append(create_mock_landmark(wrist_x + 0.04 * scale, rng_mcp_y - 0.23 * scale))  # 16: Tip extended

    # 17..20: Pinky
    pnk_mcp_y = wrist_y - palm_len * 0.82
    lms.append(create_mock_landmark(wrist_x + 0.08 * scale, pnk_mcp_y))  # 17: MCP
    lms.append(create_mock_landmark(wrist_x + 0.08 * scale, pnk_mcp_y - 0.06 * scale))  # 18: PIP
    lms.append(create_mock_landmark(wrist_x + 0.08 * scale, pnk_mcp_y - 0.11 * scale))  # 19: DIP
    if is_tiger or is_fist:
        lms.append(create_mock_landmark(wrist_x + 0.08 * scale, pnk_mcp_y - 0.02 * scale))  # 20: Tip curled
    else:
        lms.append(create_mock_landmark(wrist_x + 0.08 * scale, pnk_mcp_y - 0.22 * scale))  # 20: Tip extended

    return lms


def test_scale_invariance_finger_extension():
    """Verify that finger extension detection is scale-invariant across 0.5x, 1.0x, and 2.0x scales."""
    fsm = HandGestureFSM()

    # Test scales: hand far away (0.5x), medium (1.0x), hand close to camera (2.0x)
    for scale in [0.5, 1.0, 2.0]:
        tiger_lms = create_mock_hand_landmarks(scale=scale, is_tiger=True)
        hand_data = fsm._analyze_hand_geometry(tiger_lms)

        # Scale must scale palm_scale proportionally
        assert pytest.approx(hand_data.palm_scale, rel=1e-2) == 0.3 * scale

        # Finger extension classifications must be invariant to scale
        assert hand_data.is_index_extended is True
        assert hand_data.is_middle_extended is True
        assert hand_data.is_ring_extended is False
        assert hand_data.is_pinky_extended is False


def test_single_hand_sign_classification():
    """Test classification of atomic TIGER, FIST/BOAR, and OPEN_PALM signs."""
    fsm = HandGestureFSM()

    # 1. Tiger seal
    tiger_lms = create_mock_hand_landmarks(scale=1.0, is_tiger=True)
    tiger_data = fsm._analyze_hand_geometry(tiger_lms)
    assert fsm._classify_single_hand(tiger_data) == HandSign.TIGER

    # 2. Fist / Boar seal
    fist_lms = create_mock_hand_landmarks(scale=1.0, is_fist=True)
    fist_data = fsm._analyze_hand_geometry(fist_lms)
    assert fsm._classify_single_hand(fist_data) == HandSign.FIST_BOAR

    # 3. Open palm
    open_lms = create_mock_hand_landmarks(scale=1.0, is_tiger=False, is_fist=False)
    open_data = fsm._analyze_hand_geometry(open_lms)
    assert fsm._classify_single_hand(open_data) == HandSign.OPEN_PALM


def test_two_handed_clone_cross_seal():
    """Verify orthogonal index finger intersection detection for Shadow Clone Jutsu."""
    fsm = HandGestureFSM()

    # Hand 1: vertical index finger pointing up
    h1_lm = np.zeros((21, 3), dtype=np.float32)
    h1_lm[0] = [0.5, 0.6, 0.0]  # wrist
    h1_lm[9] = [0.5, 0.4, 0.0]  # middle mcp (scale = 0.2)
    h1_lm[5] = [0.48, 0.4, 0.0]  # index mcp
    h1_lm[8] = [0.48, 0.2, 0.0]  # index tip pointing UP (dy = -0.2)

    hand1 = SingleHandData(
        landmarks=h1_lm,
        palm_scale=0.2,
        wrist=h1_lm[0],
        is_index_extended=True,
        is_middle_extended=True,
        is_ring_extended=False,
        is_pinky_extended=False,
        is_thumb_extended=False,
    )

    # Hand 2: horizontal index finger pointing right, crossing near (0.48, 0.2)
    h2_lm = np.zeros((21, 3), dtype=np.float32)
    h2_lm[0] = [0.35, 0.2, 0.0]  # wrist
    h2_lm[9] = [0.42, 0.2, 0.0]  # middle mcp (scale = 0.07 -> adjust to 0.2)
    h2_lm[9] = [0.35, 0.4, 0.0]  # palm scale 0.2
    h2_lm[5] = [0.40, 0.2, 0.0]  # index mcp
    h2_lm[8] = [0.50, 0.2, 0.0]  # index tip crossing index 1 at (0.48, 0.2)

    hand2 = SingleHandData(
        landmarks=h2_lm,
        palm_scale=0.2,
        wrist=h2_lm[0],
        is_index_extended=True,
        is_middle_extended=True,
        is_ring_extended=False,
        is_pinky_extended=False,
        is_thumb_extended=False,
    )

    # Orthogonal cross seal should evaluate to True
    assert fsm._is_clone_cross_seal(hand1, hand2) is True


def test_combo_fsm_sequence_and_cooldown():
    """Verify temporal sliding-window combo trigger and cooldown."""
    fsm = HandGestureFSM()
    fsm.min_consecutive_frames = 1  # Fast test debounce

    # Feed TIGER
    fsm._record_sign(HandSign.TIGER)
    assert fsm._evaluate_combos() is None

    # Feed RAM -> combo [TIGER, RAM] activates INVISIBILITY_CLOAK
    fsm._record_sign(HandSign.RAM)
    combo = fsm._evaluate_combos()
    assert combo == "INVISIBILITY_CLOAK"

    # Immediate next evaluation must be throttled by cooldown
    fsm._record_sign(HandSign.CLONE_CROSS)
    assert fsm._evaluate_combos() is None


def test_process_blank_frame_graceful():
    """Verify that processing a blank frame without hands returns None without crashing."""
    fsm = HandGestureFSM()
    try:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        result = fsm.process(blank, frame_index=0)
        assert result is None
    finally:
        fsm.close()
