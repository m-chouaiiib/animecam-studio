"""Application entry point and graceful execution harness for AnimeCam Studio."""

from __future__ import annotations

import argparse
import logging
import sys

from animecam.config import AppConfig, CameraConfig, PipelineConfig
from animecam.core.engine import CVEngine
from animecam.ui.studio_app import AnimeCamStudioApp


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging for AnimeCam Studio."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, datefmt="%H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for hardware configuration."""
    parser = argparse.ArgumentParser(
        prog="animecam",
        description="AnimeCam Studio: Real-Time Virtual Webcam Effects Studio",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Physical webcam device index (default: 0). Use -1 for synthetic mode.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target video width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target video height in pixels (default: 720).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate (default: 30).",
    )
    parser.add_argument(
        "--no-virtual-cam",
        action="store_true",
        help="Run in preview-only mode without broadcasting to pyvirtualcam.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic procedural test pattern feed without using physical camera.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("animecam.main")

    dev_index = -1 if args.synthetic else args.device
    enable_vcam = not args.no_virtual_cam

    # Build customized configuration from CLI args
    app_config = AppConfig(
        camera=CameraConfig(
            device_index=dev_index,
            width=args.width,
            height=args.height,
            target_fps=args.fps,
        ),
        pipeline=PipelineConfig(
            enable_virtual_cam=enable_vcam,
        ),
    )

    logger.info("Initializing AnimeCam Studio...")
    logger.info(
        "Camera: device=%d, resolution=%dx%d @ %d FPS, virtual_cam=%s",
        dev_index,
        args.width,
        args.height,
        args.fps,
        enable_vcam,
    )

    engine = CVEngine(app_config)

    try:
        # Start background CV ingestion and processing thread
        engine.start()

        # Start CustomTkinter GUI on main thread
        app = AnimeCamStudioApp(engine, app_config)
        app.mainloop()

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    except Exception as err:
        logger.exception("Fatal error during execution: %s", err)
        return 1
    finally:
        logger.info("Executing safe hardware release...")
        engine.stop()

    logger.info("AnimeCam Studio closed cleanly. Sayonara! 🥷")
    return 0


if __name__ == "__main__":
    sys.exit(main())
