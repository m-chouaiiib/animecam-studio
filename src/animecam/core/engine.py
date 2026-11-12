"""Thread-safe CV Orchestrator and frame scheduler for AnimeCam Studio.

Coordinates:
- Camera ingestion via SafeCamera in a dedicated daemon thread.
- Lazy model execution (running hands at 15 Hz cadence, FaceMesh only when active).
- Dual-buffer output splitting: Clean frame for pyvirtualcam, telemetry HUD for local UI.
- Lock-free atomic frame swapping and hardware privacy panic kill-switch.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from animecam.config import DEFAULT_CONFIG, AppConfig
from animecam.core.camera import SafeCamera, SafeVirtualCamera
from animecam.core.gestures import HandGestureFSM

logger = logging.getLogger("animecam.engine")


class CVEngine:
    """Core computer vision processing orchestrator."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self.config = config

        # Hardware wrappers
        self.camera = SafeCamera(config.camera)
        self.virtual_cam = SafeVirtualCamera(
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.target_fps,
            enabled=config.pipeline.enable_virtual_cam,
        )

        # Effect toggle states (thread-safe boolean flags)
        self.invisibility_enabled: bool = False
        self.clones_enabled: bool = False
        self.eyes_enabled: bool = False
        self.selected_shader: str = config.eye.default_shader_name
        self.show_debug_hud: bool = True

        # Privacy panic switch
        self.panic_active: bool = False

        # Concurrency & Lock-Free Threading primitives
        self._stop_event = threading.Event()
        self._frame_ready_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # Lock-free atomic latest-frame pointers
        self._latest_preview_frame: np.ndarray | None = None
        self._latest_virtual_frame: np.ndarray | None = None

        # Telemetry & Performance counters
        self._frame_index: int = 0
        self.fps: float = 0.0
        self.inference_latency_ms: float = 0.0
        self._last_fps_calc_time: float = time.perf_counter()
        self._fps_frame_counter: int = 0

        # Gesture Recognition FSM
        self.gesture_fsm = HandGestureFSM(config.mediapipe)
        self.gesture_detector: Callable[[np.ndarray, int], str | None] | None = (
            self.gesture_fsm.process
        )
        self.invisibility_filter: Callable[[np.ndarray], np.ndarray] | None = None
        self.clone_compositor: Callable[[np.ndarray], np.ndarray] | None = None
        self.eye_overlay_filter: Callable[[np.ndarray], np.ndarray] | None = None

        # Current detected gesture combo
        self.active_combo_name: str = "Neutral"

        # Signal handling
        self._register_signals()

    def _register_signals(self) -> None:
        """Register OS interrupt signals to ensure hardware camera handles release."""
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._handle_os_signal)
                signal.signal(signal.SIGTERM, self._handle_os_signal)
        except (ValueError, AttributeError) as err:
            logger.debug("Signal registration skipped (non-main thread or unsupported platform): %s", err)

    def _handle_os_signal(self, signum: int, frame) -> None:
        logger.warning("Received termination signal %d. Triggering safe shutdown...", signum)
        self.stop()
        sys.exit(0)

    def start(self) -> None:
        """Start camera hardware and spawn the dedicated daemon worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("CVEngine worker thread already running.")
            return

        self._stop_event.clear()
        self.camera.open()
        self.virtual_cam.start()

        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="CVEngineWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("CVEngine worker thread started successfully.")

    def stop(self) -> None:
        """Signal the worker loop to stop, wait for thread exit, and release hardware."""
        logger.info("Stopping CVEngine...")
        self._stop_event.set()

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        # Guarantee hardware and ML resource release
        self.camera.release()
        self.virtual_cam.close()
        self.gesture_fsm.close()
        logger.info("CVEngine stopped cleanly; hardware released.")

    def panic_kill(self) -> None:
        """Hardware/UI privacy kill-switch: cut virtual broadcast to black and clear buffers."""
        logger.warning("[PANIC BUTTON] Privacy kill-switch activated!")
        self.panic_active = True

        # Disable all active jutsu
        self.invisibility_enabled = False
        self.clones_enabled = False
        self.eyes_enabled = False
        self.active_combo_name = "KILLED (PANIC)"

        # Immediately send blank black frame to virtual camera to mask feed
        w, h = self.camera.resolution
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
        self.virtual_cam.send(black_frame)

    def reset_panic(self) -> None:
        """Resume normal virtual camera broadcast after panic activation."""
        logger.info("[PANIC BUTTON] Privacy kill-switch reset.")
        self.panic_active = False
        self.active_combo_name = "Neutral"

    def get_latest_frame(self) -> np.ndarray | None:
        """Lock-free non-blocking retrieval of the latest preview frame for the UI.

        Returns:
            Optional[np.ndarray]: Latest preview frame reference, or None.
        """
        return self._latest_preview_frame

    def _run_loop(self) -> None:
        """Core CV pipeline execution loop running inside the daemon thread."""
        target_frame_time = 1.0 / max(1, self.config.camera.target_fps)

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            # 1. Grab raw camera frame
            success, raw_frame = self.camera.read()
            if not success or raw_frame is None:
                time.sleep(0.005)
                continue

            self._frame_index += 1
            t_process_start = time.perf_counter()

            # 2. Check Panic Mode
            if self.panic_active:
                # In panic mode, blank both feeds
                h, w = raw_frame.shape[:2]
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(
                    blank_frame,
                    "PRIVACY KILL-SWITCH ENGAGED",
                    (w // 2 - 250, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                self._latest_preview_frame = blank_frame
                self.virtual_cam.send(np.zeros((h, w, 3), dtype=np.uint8))
                self._frame_ready_event.set()
                self._pace_frame(loop_start, target_frame_time)
                continue

            # 3. Working copy for clean virtual camera output
            clean_output = raw_frame.copy()

            # 4. Lazy Hand Gesture Detection (Run at 15 Hz cadence, every 2nd frame)
            if (
                self.gesture_detector is not None
                and self._frame_index % self.config.pipeline.gesture_cadence == 0
            ):
                new_combo = self.gesture_detector(raw_frame, self._frame_index)
                if new_combo:
                    self._handle_combo_trigger(new_combo)

            # 5. Pipeline Effects (Vectorized NumPy Operations)
            if self.invisibility_enabled and self.invisibility_filter is not None:
                clean_output = self.invisibility_filter(clean_output)

            if self.clones_enabled and self.clone_compositor is not None:
                clean_output = self.clone_compositor(clean_output)

            if self.eyes_enabled and self.eye_overlay_filter is not None:
                clean_output = self.eye_overlay_filter(clean_output)

            # Measure processing latency
            self.inference_latency_ms = (time.perf_counter() - t_process_start) * 1000.0

            # 6. Transmit clean frame to pyvirtualcam (Zero HUD/Landmarks guarantee)
            self.virtual_cam.send(clean_output)

            # 7. Local UI Preview Generation (with Telemetry HUD)
            preview_frame = clean_output.copy()
            if self.show_debug_hud:
                self._render_telemetry_hud(preview_frame)

            # 8. Lock-free atomic reference swap
            self._latest_preview_frame = preview_frame
            self._latest_virtual_frame = clean_output
            self._frame_ready_event.set()

            # 9. Update FPS Telemetry
            self._update_fps_telemetry()

            # 10. Frame Pacing (Target 30 FPS sleep adjustment)
            self._pace_frame(loop_start, target_frame_time)

    def _handle_combo_trigger(self, combo: str) -> None:
        """Map recognized jutsu hand signs to engine effect states."""
        self.active_combo_name = combo
        if combo == "CLONE_JUTSU":
            self.clones_enabled = not self.clones_enabled
            logger.info("Clone Jutsu toggled: %s", self.clones_enabled)
        elif combo == "INVISIBILITY_CLOAK":
            self.invisibility_enabled = not self.invisibility_enabled
            logger.info("Invisibility toggled: %s", self.invisibility_enabled)
        elif combo == "DOJUTSU_EYES":
            self.eyes_enabled = not self.eyes_enabled
            logger.info("Dojutsu Eyes toggled: %s", self.eyes_enabled)

    def _render_telemetry_hud(self, frame: np.ndarray) -> None:
        """Render debug telemetry onto the local preview frame only."""
        # 1. Render detected hand landmarks
        self.gesture_fsm.draw_debug_landmarks(frame)

        h, w = frame.shape[:2]

        # Top telemetry bar overlay
        cv2.rectangle(frame, (10, 10), (380, 115), (15, 15, 20), -1)
        cv2.rectangle(frame, (10, 10), (380, 115), (37, 73, 255), 1)

        # Telemetry text lines
        fps_color = (16, 185, 129) if self.fps >= 25.0 else (0, 165, 255)
        cv2.putText(
            frame,
            f"FPS: {self.fps:4.1f} | Latency: {self.inference_latency_ms:4.1f}ms",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            fps_color,
            1,
            cv2.LINE_AA,
        )

        status_txt = f"Active Jutsu: {self.active_combo_name}"
        cv2.putText(
            frame,
            status_txt,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        toggles = (
            f"Eyes: {'ON' if self.eyes_enabled else 'OFF'} | "
            f"Clones: {'ON' if self.clones_enabled else 'OFF'} | "
            f"Stealth: {'ON' if self.invisibility_enabled else 'OFF'}"
        )
        cv2.putText(
            frame,
            toggles,
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (147, 142, 142),
            1,
            cv2.LINE_AA,
        )

    def _update_fps_telemetry(self) -> None:
        """Calculate running FPS over a 1.0-second rolling window."""
        self._fps_frame_counter += 1
        now = time.perf_counter()
        delta = now - self._last_fps_calc_time
        if delta >= 1.0:
            self.fps = self._fps_frame_counter / delta
            self._fps_frame_counter = 0
            self._last_fps_calc_time = now

    def _pace_frame(self, loop_start: float, target_frame_time: float) -> None:
        """Throttle the worker thread to maintain the target FPS and avoid 100% CPU pinning."""
        elapsed = time.perf_counter() - loop_start
        sleep_dur = target_frame_time - elapsed
        if sleep_dur > 0.001:
            time.sleep(sleep_dur)
