"""Unit tests for AnimeCam Studio CLI parsing and UI helpers."""

import sys
from unittest.mock import patch

from animecam.config import DEFAULT_CONFIG
from animecam.main import parse_args
from animecam.ui.studio_app import AnimeCamStudioApp


def test_cli_argument_parsing():
    """Verify CLI parsing maps flags to parameters."""
    test_args = [
        "animecam",
        "--device",
        "2",
        "--width",
        "640",
        "--height",
        "480",
        "--fps",
        "60",
        "--no-virtual-cam",
        "--synthetic",
        "--verbose",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.device == 2
        assert args.width == 640
        assert args.height == 480
        assert args.fps == 60
        assert args.no_virtual_cam is True
        assert args.synthetic is True
        assert args.verbose is True


def test_shader_discovery_helper():
    """Verify that AnimeCamStudioApp discovers shaders from assets/shaders."""
    # Test standalone helper logic without opening a window
    shaders = AnimeCamStudioApp._discover_shader_names(None)  # type: ignore[arg-type]
    assert len(shaders) >= 3
    assert "sharingan.png" in shaders
    assert "rinnegan.png" in shaders
    assert "sage_mode.png" in shaders


def test_ui_instantiation_and_sync():
    """Verify that UI instantiates, binds controls, and synchronizes with CVEngine."""
    from animecam.core.engine import CVEngine

    engine = CVEngine(DEFAULT_CONFIG)
    try:
        app = AnimeCamStudioApp(engine, DEFAULT_CONFIG)
        # Verify initial states
        assert app.engine is engine
        assert app.switch_invisibility.get() == 0
        assert app.switch_clones.get() == 0
        assert app.switch_eyes.get() == 0

        # Test manual toggle
        app.switch_invisibility.select()
        app._on_invisibility_toggle()
        assert engine.invisibility_enabled is True

        # Test panic button press
        app._on_panic_press()
        assert engine.panic_active is True
        assert engine.invisibility_enabled is False
        assert app.switch_invisibility.get() == 0

        # Clean close
        app._on_close()
        assert app._is_closing is True

    except Exception as err:
        # In headless Linux CI without X11, Tkinter raises TclError
        if "no display name" in str(err) or "TclError" in str(type(err)):
            pass
        else:
            raise
    finally:
        engine.stop()
