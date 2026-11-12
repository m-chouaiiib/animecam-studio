# AnimeCam Studio - Architecture & Engineering Specification

## 1. High-Level System Architecture

AnimeCam Studio is engineered for real-time video processing targeting a stable **30 FPS on standard consumer CPUs** (<15% RAM overhead). It decouples frame acquisition, high-throughput computer vision pipeline execution, virtual camera broadcasting, and UI telemetry.

```
                           +---------------------------+
                           |  Physical Webcam (V4L2 /  |
                           |  DirectShow / AVFound.)   |
                           +-------------+-------------+
                                         |
                                         v
                      +-------------------------------------+
                      | SafeCamera Context Manager          |
                      | - Hardware release guarantees       |
                      | - Zero unmanaged capture handles    |
                      +------------------+------------------+
                                         | Raw BGR Frame (720p/30fps)
                                         v
+---------------------------------------------------------------------------------+
|                       CV Engine Thread (Daemon Thread)                         |
|                                                                                 |
|  [Frame Scheduler]                                                              |
|   -> Evaluates frame index: runs Heavy ML (Hands) at 15 Hz cadence (odd frames) |
|   -> Evaluates active effects: FaceMesh executed lazily (only if Dojutsu on)    |
|                                                                                 |
|  [Vectorized Effects Pipeline]                                                  |
|   1. Hand Combo FSM (Scale-invariant Euclidean ratios & sliding-window deque)   |
|   2. Invisibility / Stealth Mode (Selfie Segmentation + EMA Background Healing) |
|   3. Shadow Clone Jutsu (Ring-buffer deque(maxlen=24) + vectorized alpha decay) |
|   4. Dojutsu / Eye Overlay (Iris landmark geometry + rotating BGRA blit)        |
|                                                                                 |
|  [Output Splitting]                                                             |
|   ├── Clean Output Frame -> pyvirtualcam (Virtual Camera Feed)                  |
|   └── Debug/Telemetry Frame (HUD, FPS, Landmarks) -> Atomic Pointer Swap       |
+----------------------------------------+----------------------------------------+
                                         |
                                         | Atomic Pointer Update
                                         v
                      +-------------------------------------+
                      | Lock-Free UI Presentation           |
                      | - CustomTkinter 60 Hz event loop    |
                      | - Non-blocking frame retrieval      |
                      | - Hardware Panic Button kill-switch |
                      +-------------------------------------+
```

## 2. Concurrency Model: Decoupled UI & Lock-Free Atomic Swapping

Traditional Python GUI apps freeze or stutter if OpenCV inference runs directly on the UI thread (`Tk.mainloop()`). Conversely, using a heavy locking mechanism (`threading.Lock` across 30 FPS loops) induces contention and latency jitter.

### Architectural Solution:
1. **Daemon Engine Worker:** The camera acquisition and CV processing pipeline run inside an isolated Python daemon thread.
2. **Atomic Pointer Swapping:** In Python (CPython), assigning an object reference (`self._latest_display_frame = frame`) is atomic under the GIL. The UI reads this reference during its periodic tick (e.g. `after(16, self._render_loop)`) without locking the CV loop.
3. **Threading Event Signaling:** A `threading.Event` wakes up the consumer when a new frame is ready, while ensuring frame drops occur gracefully if the UI is busy, preventing buffer accumulation.

## 3. Zero-Copy Vectorization & Memory Hygiene

To eliminate Python garbage collection pauses and CPU overhead:
- **No Python For-Loops over Pixels:** Image transformations are strictly computed using NumPy vectorized C-intrinsics (`np.where`, 3-channel broadcasting, and direct slice mutations).
- **Ring Buffers (`collections.deque(maxlen=N)`):** Shadow clones leverage a hardware-bounded double-ended queue. When frame count exceeds 24, oldest items are dropped at the C level with $O(1)$ complexity, preventing memory leaks.
- **Pre-allocated Healing Buffer:** The background model (`clean_bg`) is allocated once upon camera initialization and updated continuously via Exponential Moving Average (EMA):
  $$\text{bg}_{t} = (1 - \alpha) \cdot \text{bg}_{t-1} + \alpha \cdot \text{frame}_t$$
  only where the person mask is zero.

## 4. Hardware Release Guarantees & Privacy Kill-Switch

- **Context Management:** Both `cv2.VideoCapture` and `pyvirtualcam.Camera` are wrapped in context managers ensuring explicit C-library device releases (`cap.release()`, `cam.close()`) even during unhandled exceptions.
- **OS Signal Interception:** Signal handlers for `SIGINT` and `SIGTERM` invoke the engine's graceful `stop()` method to prevent orphaned hardware locks on video nodes (`/dev/video*`).
- **Panic Button:** Pressing `Space` or the UI Privacy Switch instantly blanks the virtual feed, clears the background cache and clone ring buffers from memory, and disables all effects.
