"""Multi-seal Hand Gesture Recognition & Combo FSM.

Computes scale-invariant Euclidean ratios from MediaPipe Hands 21 landmarks
and runs a temporal sliding-window state machine to trigger ninja jutsu combos.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from animecam.config import DEFAULT_CONFIG, MediaPipeConfig

logger = logging.getLogger("animecam.gestures")


class HandSign(str, Enum):
    """Recognized atomic hand signs."""

    NONE = "NONE"
    OPEN_PALM = "OPEN_PALM"
    FIST_BOAR = "BOAR"
    TIGER = "TIGER"
    RAM = "RAM"
    CLONE_CROSS = "CLONE_CROSS"


@dataclass
class SingleHandData:
    """Pre-processed hand geometry and landmark metrics."""

    landmarks: np.ndarray  # Shape: (21, 3) normalized [0, 1]
    palm_scale: float
    wrist: np.ndarray
    is_index_extended: bool
    is_middle_extended: bool
    is_ring_extended: bool
    is_pinky_extended: bool
    is_thumb_extended: bool


class HandGestureFSM:
    """Scale-invariant hand sign classifier and sliding-window combo state machine."""

    def __init__(self, config: MediaPipeConfig = DEFAULT_CONFIG.mediapipe) -> None:
        self.config = config
        self._mp_hands = None
        self._hands = None

        # Temporal sliding window for combo recognition
        # Stores tuples: (timestamp: float, sign: HandSign)
        self._history: collections.deque[tuple[float, HandSign]] = collections.deque(maxlen=45)

        # Combo debounce and cooldown management
        self._last_detected_sign: HandSign = HandSign.NONE
        self._sign_consecutive_count: int = 0
        self._last_combo_trigger_time: float = 0.0
        self._combo_cooldown_sec: float = 1.25

        # Debounce threshold: minimum consecutive frames to register a stable sign
        self.min_consecutive_frames: int = 2

        # Holds latest detected landmarks for preview HUD rendering
        self.last_results = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of MediaPipe Hands."""
        if self._hands is not None:
            return

        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_hand_detection_confidence,
            min_tracking_confidence=self.config.min_hand_tracking_confidence,
        )
        logger.info("MediaPipe Hands initialized with max_num_hands=%d.", self.config.max_num_hands)

    def close(self) -> None:
        """Release MediaPipe Hands resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
            self._mp_hands = None

    def process(self, frame_bgr: np.ndarray, frame_index: int = 0) -> str | None:
        """Detect hand landmarks, classify current signs, and evaluate jutsu combos.

        Args:
            frame_bgr: uint8 BGR image from camera.
            frame_index: Running frame counter.

        Returns:
            Optional[str]: Triggered jutsu combo name (e.g. 'CLONE_JUTSU'), or None.
        """
        self._ensure_initialized()

        # Convert BGR to RGB without making unnecessary copies
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        self.last_results = results

        if not results.multi_hand_landmarks:
            self._record_sign(HandSign.NONE)
            return None

        hands_data = [
            self._analyze_hand_geometry(hand.landmark)
            for hand in results.multi_hand_landmarks
        ]

        # 1. Check for Two-Handed CLONE_CROSS seal first
        current_sign = HandSign.NONE
        if len(hands_data) >= 2:
            if self._is_clone_cross_seal(hands_data[0], hands_data[1]):
                current_sign = HandSign.CLONE_CROSS

        # 2. If no two-hand seal, classify single-hand sign (using the most prominent hand)
        if current_sign == HandSign.NONE and len(hands_data) > 0:
            current_sign = self._classify_single_hand(hands_data[0])

        # 3. Update temporal state and check combo trigger
        self._record_sign(current_sign)
        return self._evaluate_combos()

    def _analyze_hand_geometry(self, landmarks_proto) -> SingleHandData:
        """Extract scale-invariant geometric metrics from 21 MediaPipe landmarks."""
        # Convert landmarks to (21, 3) NumPy array
        lm = np.empty((21, 3), dtype=np.float32)
        for i in range(21):
            lm[i, 0] = landmarks_proto[i].x
            lm[i, 1] = landmarks_proto[i].y
            lm[i, 2] = landmarks_proto[i].z

        wrist = lm[0]
        middle_mcp = lm[9]

        # Reference anatomical palm vector: distance from wrist (0) to middle MCP (9)
        # This rigid bone structure is invariant to finger folding
        palm_scale = float(np.linalg.norm(middle_mcp - wrist))
        if palm_scale < 1e-4:
            palm_scale = 1.0

        # Finger extension tests using normalized wrist-to-tip vs wrist-to-PIP ratios
        is_index = self._is_finger_extended(lm, tip_idx=8, pip_idx=6, mcp_idx=5, palm_scale=palm_scale)
        is_middle = self._is_finger_extended(lm, tip_idx=12, pip_idx=10, mcp_idx=9, palm_scale=palm_scale)
        is_ring = self._is_finger_extended(lm, tip_idx=16, pip_idx=14, mcp_idx=13, palm_scale=palm_scale)
        is_pinky = self._is_finger_extended(lm, tip_idx=20, pip_idx=18, mcp_idx=17, palm_scale=palm_scale)

        # Thumb extension: tip (4) distance from index MCP (5) and wrist (0)
        d_thumb_idx = float(np.linalg.norm(lm[4] - lm[5])) / palm_scale
        d_thumb_wrist = float(np.linalg.norm(lm[4] - wrist)) / palm_scale
        is_thumb = d_thumb_idx > 0.40 and d_thumb_wrist > 0.50

        return SingleHandData(
            landmarks=lm,
            palm_scale=palm_scale,
            wrist=wrist,
            is_index_extended=is_index,
            is_middle_extended=is_middle,
            is_ring_extended=is_ring,
            is_pinky_extended=is_pinky,
            is_thumb_extended=is_thumb,
        )

    def _is_finger_extended(
        self,
        lm: np.ndarray,
        tip_idx: int,
        pip_idx: int,
        mcp_idx: int,
        palm_scale: float,
    ) -> bool:
        """Check if finger is extended using scale-invariant Euclidean distance from wrist."""
        wrist = lm[0]
        d_tip_wrist = np.linalg.norm(lm[tip_idx] - wrist)
        d_pip_wrist = np.linalg.norm(lm[pip_idx] - wrist)
        d_tip_mcp = np.linalg.norm(lm[tip_idx] - lm[mcp_idx])

        # Extended if tip is significantly further from wrist than PIP joint
        return bool(d_tip_wrist > d_pip_wrist + (0.20 * palm_scale) and d_tip_mcp > (0.65 * palm_scale))

    def _classify_single_hand(self, hand: SingleHandData) -> HandSign:
        """Classify atomic hand sign from single-hand landmark metrics."""
        lm = hand.landmarks
        scale = hand.palm_scale

        # 1. TIGER Seal (寅):
        # Index + Middle extended together, Ring + Pinky curled, Index & Middle tips close
        if hand.is_index_extended and hand.is_middle_extended:
            if not hand.is_ring_extended and not hand.is_pinky_extended:
                d_index_middle = float(np.linalg.norm(lm[8] - lm[12])) / scale
                if d_index_middle < 0.40:
                    return HandSign.TIGER

        # 2. RAM Seal (未):
        # Index extended, Middle extended, Ring and Pinky curled, thumb tucked into palm
        if hand.is_index_extended and hand.is_middle_extended:
            if not hand.is_ring_extended and not hand.is_pinky_extended:
                d_thumb_middle_mcp = float(np.linalg.norm(lm[4] - lm[9])) / scale
                if d_thumb_middle_mcp < 0.50:
                    return HandSign.RAM

        # 3. BOAR Seal (亥) / Fist:
        # All fingers curled tightly
        if (
            not hand.is_index_extended
            and not hand.is_middle_extended
            and not hand.is_ring_extended
            and not hand.is_pinky_extended
        ):
            return HandSign.FIST_BOAR

        # 4. OPEN_PALM:
        # All 5 digits extended
        if (
            hand.is_index_extended
            and hand.is_middle_extended
            and hand.is_ring_extended
            and hand.is_pinky_extended
            and hand.is_thumb_extended
        ):
            return HandSign.OPEN_PALM

        return HandSign.NONE

    def _is_clone_cross_seal(self, hand1: SingleHandData, hand2: SingleHandData) -> bool:
        """Check for Kage Bunshin Cross Seal (two hands forming a '+' cross)."""
        # Both hands must have index fingers extended
        if not (hand1.is_index_extended and hand2.is_index_extended):
            return False

        # Distance between index finger midpoints of both hands
        avg_scale = (hand1.palm_scale + hand2.palm_scale) / 2.0
        left_index_tip = hand1.landmarks[8]
        right_index_tip = hand2.landmarks[8]

        d_tips = float(np.linalg.norm(left_index_tip - right_index_tip)) / avg_scale
        if d_tips > 0.65:
            return False

        # Orientation: check angle between left and right index finger vectors
        v1 = hand1.landmarks[8] - hand1.landmarks[5]
        v2 = hand2.landmarks[8] - hand2.landmarks[5]
        cos_theta = np.dot(v1[:2], v2[:2]) / (np.linalg.norm(v1[:2]) * np.linalg.norm(v2[:2]) + 1e-6)
        angle_deg = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))

        # Approximately perpendicular (between 45 and 135 degrees)
        return 45.0 <= angle_deg <= 135.0

    def _record_sign(self, sign: HandSign) -> None:
        """Append classified sign to sliding window with debouncing."""
        now = time.perf_counter()

        if sign == self._last_detected_sign:
            self._sign_consecutive_count += 1
        else:
            self._last_detected_sign = sign
            self._sign_consecutive_count = 1

        # Only register stable signs that survive debouncing
        if self._sign_consecutive_count >= self.min_consecutive_frames:
            self._history.append((now, sign))

    def _evaluate_combos(self) -> str | None:
        """Examine temporal sign history to trigger jutsu combos."""
        now = time.perf_counter()
        if now - self._last_combo_trigger_time < self._combo_cooldown_sec:
            return None

        # Clean old history entries (> 2.0 seconds old)
        while self._history and (now - self._history[0][0] > 2.0):
            self._history.popleft()

        if not self._history:
            return None

        # Extract sequence of unique consecutive signs
        sign_sequence: list[HandSign] = []
        for _, s in self._history:
            if s != HandSign.NONE and (not sign_sequence or sign_sequence[-1] != s):
                sign_sequence.append(s)

        # 1. Immediate Shadow Clone Cross Seal
        if HandSign.CLONE_CROSS in sign_sequence:
            self._trigger_combo("CLONE_JUTSU", now)
            return "CLONE_JUTSU"

        # 2. Invisibility Cloak combo: [TIGER -> RAM] or [RAM -> TIGER]
        if len(sign_sequence) >= 2:
            last_two = (sign_sequence[-2], sign_sequence[-1])
            if last_two in [(HandSign.TIGER, HandSign.RAM), (HandSign.RAM, HandSign.TIGER)]:
                self._trigger_combo("INVISIBILITY_CLOAK", now)
                return "INVISIBILITY_CLOAK"

        # 3. Dojutsu Sharingan Eye combo: [BOAR -> TIGER] or Holding TIGER for > 15 frames
        tiger_count = sum(1 for _, s in self._history if s == HandSign.TIGER)
        if tiger_count >= 14:
            self._trigger_combo("DOJUTSU_EYES", now)
            return "DOJUTSU_EYES"

        if len(sign_sequence) >= 2 and (sign_sequence[-2], sign_sequence[-1]) == (HandSign.FIST_BOAR, HandSign.TIGER):
            self._trigger_combo("DOJUTSU_EYES", now)
            return "DOJUTSU_EYES"

        return None

    def _trigger_combo(self, combo_name: str, now: float) -> None:
        """Trigger combo, flush history, and engage cooldown."""
        logger.info("[JUTSU TRIGGERED] %s!", combo_name)
        self._last_combo_trigger_time = now
        self._history.clear()
        self._sign_consecutive_count = 0

    def draw_debug_landmarks(self, frame_bgr: np.ndarray) -> None:
        """Render glowing ninja hand landmarks onto the local preview frame."""
        if self.last_results is None or not self.last_results.multi_hand_landmarks:
            return

        h, w = frame_bgr.shape[:2]
        for hand_landmarks in self.last_results.multi_hand_landmarks:
            # Draw glowing green joints
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (cx, cy), 4, (16, 185, 129), -1)

            # Draw connector bones
            if self._mp_hands is not None:
                for connection in self._mp_hands.HAND_CONNECTIONS:
                    pt1 = hand_landmarks.landmark[connection[0]]
                    pt2 = hand_landmarks.landmark[connection[1]]
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(frame_bgr, (x1, y1), (x2, y2), (255, 73, 37), 2)
