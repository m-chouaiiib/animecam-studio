"""Asset generator script to create procedural BGRA shader PNGs.

Generates:
1. sharingan.png - Crimson glowing iris with three tomoe marks and outer ring
2. rinnegan.png - Ripple pattern concentric rings with mystical lavender hue
3. sage_mode.png - Golden toad sage iris with horizontal bar pupil
"""

from __future__ import annotations

import math
from pathlib import Path

# Create directory if needed
OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "shaders"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def create_png(width: int, height: int, pixels_rgba: bytes, filepath: Path) -> None:
    """Minimal standalone pure-Python PNG writer (zero third-party dependencies)."""
    import struct
    import zlib

    # PNG Signature
    png_header = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk: width (4), height (4), bit depth (1), color type (6 = RGBA), compression (0), filter (0), interlace (0)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    # Raw image data with filter byte (0 = None) at start of each row
    raw_lines = []
    stride = width * 4
    for y in range(height):
        raw_lines.append(b"\x00" + pixels_rgba[y * stride : (y + 1) * stride])
    raw_data = b"".join(raw_lines)

    # IDAT chunk
    compressed_data = zlib.compress(raw_data, 9)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed_data))
    idat_chunk = struct.pack(">I", len(compressed_data)) + b"IDAT" + compressed_data + idat_crc

    # IEND chunk
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND"))
    iend_chunk = struct.pack(">I", 0) + b"IEND" + iend_crc

    with open(filepath, "wb") as f:
        f.write(png_header + ihdr_chunk + idat_chunk + iend_chunk)


def generate_sharingan(size: int = 256) -> None:
    """Generate 3-Tomoe Sharingan texture."""
    center = size / 2.0
    radius = size * 0.46
    buf = bytearray(size * size * 4)

    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = math.hypot(dx, dy)
            idx = (y * size + x) * 4

            if dist > radius:
                # Fully transparent outside
                buf[idx : idx + 4] = b"\x00\x00\x00\x00"
                continue

            # Antialiasing on outer boundary
            edge_alpha = 1.0
            if dist > radius - 2.0:
                edge_alpha = max(0.0, (radius - dist) / 2.0)

            # Iris base color: vibrant crimson gradient
            r_norm = dist / radius
            r = int(min(255, 230 - 40 * r_norm))
            g = int(20 * (1.0 - r_norm))
            b = int(25 * (1.0 - r_norm))
            a = int(255 * edge_alpha)

            # Outer black boundary ring
            if dist > radius * 0.88:
                r, g, b = 20, 20, 25

            # Inner thin black ring connecting tomoe
            if abs(dist - radius * 0.55) < 3.5:
                r, g, b = 15, 15, 18

            # Center pupil
            if dist < radius * 0.22:
                r, g, b = 10, 10, 12

            # 3 Tomoe marks
            angle = math.atan2(dy, dx)
            for i in range(3):
                tomoe_angle = i * (2 * math.pi / 3)
                # Position of tomoe head
                tx = center + radius * 0.55 * math.cos(tomoe_angle)
                ty = center + radius * 0.55 * math.sin(tomoe_angle)
                d_tomoe = math.hypot(x - tx, y - ty)

                # Head circle
                if d_tomoe < radius * 0.12:
                    r, g, b = 12, 12, 15

                # Curved tail
                tail_angle = tomoe_angle + 0.35
                ttx = center + radius * 0.48 * math.cos(tail_angle)
                tty = center + radius * 0.48 * math.sin(tail_angle)
                d_tail = math.hypot(x - ttx, y - tty)
                if d_tail < radius * 0.08:
                    r, g, b = 12, 12, 15

            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b
            buf[idx + 3] = a

    create_png(size, size, bytes(buf), OUT_DIR / "sharingan.png")
    print(f"[✓] Generated {OUT_DIR / 'sharingan.png'}")


def generate_rinnegan(size: int = 256) -> None:
    """Generate Rinnegan ripple pattern."""
    center = size / 2.0
    radius = size * 0.46
    buf = bytearray(size * size * 4)

    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = math.hypot(dx, dy)
            idx = (y * size + x) * 4

            if dist > radius:
                buf[idx : idx + 4] = b"\x00\x00\x00\x00"
                continue

            edge_alpha = 1.0
            if dist > radius - 2.0:
                edge_alpha = max(0.0, (radius - dist) / 2.0)

            # Mystical lavender base
            r = 150
            g = 135
            b = 185
            a = int(250 * edge_alpha)

            # Center pupil
            if dist < radius * 0.16:
                r, g, b = 25, 20, 35

            # Concentric ripple rings (3 rings)
            ring_radii = [radius * 0.38, radius * 0.60, radius * 0.82]
            for r_dist in ring_radii:
                if abs(dist - r_dist) < 3.2:
                    r, g, b = 45, 35, 60

            # Outer boundary
            if dist > radius * 0.92:
                r, g, b = 40, 30, 55

            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b
            buf[idx + 3] = a

    create_png(size, size, bytes(buf), OUT_DIR / "rinnegan.png")
    print(f"[✓] Generated {OUT_DIR / 'rinnegan.png'}")


def generate_sage_mode(size: int = 256) -> None:
    """Generate Sage Mode toad eye texture."""
    center = size / 2.0
    radius = size * 0.46
    buf = bytearray(size * size * 4)

    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = math.hypot(dx, dy)
            idx = (y * size + x) * 4

            if dist > radius:
                buf[idx : idx + 4] = b"\x00\x00\x00\x00"
                continue

            edge_alpha = 1.0
            if dist > radius - 2.0:
                edge_alpha = max(0.0, (radius - dist) / 2.0)

            # Golden amber iris
            r = 235
            g = 175
            b = 30
            a = int(250 * edge_alpha)

            # Outer boundary
            if dist > radius * 0.88:
                r, g, b = 30, 20, 10

            # Horizontal slit pupil (Toad sage)
            if abs(dy) < radius * 0.16 and abs(dx) < radius * 0.65:
                r, g, b = 15, 12, 10

            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b
            buf[idx + 3] = a

    create_png(size, size, bytes(buf), OUT_DIR / "sage_mode.png")
    print(f"[✓] Generated {OUT_DIR / 'sage_mode.png'}")


if __name__ == "__main__":
    generate_sharingan()
    generate_rinnegan()
    generate_sage_mode()
