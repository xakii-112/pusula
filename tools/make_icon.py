"""Draw the application icon (a compass needle) and write src/pusula/web/pusula.ico. Run once; the
result is checked in. PNG payloads in an ICO container, standard library and numpy only.

    uv run python tools/make_icon.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src" / "pusula" / "web" / "pusula.ico"
SIZES = (16, 32, 48, 64, 256)
RING = (240, 180, 41)
NORTH = (240, 180, 41)
SOUTH = (139, 147, 167)
DISC = (15, 18, 24)
HUB = (11, 14, 19)
OVERSAMPLE = 4


def _triangle(px: np.ndarray, py: np.ndarray, a: tuple, b: tuple, c: tuple) -> np.ndarray:
    def side(p: tuple, q: tuple) -> np.ndarray:
        return (q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0])

    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))


def draw(size: int) -> np.ndarray:
    n = size * OVERSAMPLE
    axis = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    px, py = np.meshgrid(axis, axis)
    r = np.hypot(px, py)
    img = np.zeros((n, n, 4))
    disc = r <= 0.97
    img[disc] = (*DISC, 255)
    img[(r > 0.80) & disc] = (*RING, 255)
    # the needle, tilted a little off north
    t = np.radians(28.0)
    ux, uy = np.sin(t), -np.cos(t)
    vx, vy = np.cos(t), np.sin(t)
    tip = (0.70 * ux, 0.70 * uy)
    tail = (-0.70 * ux, -0.70 * uy)
    left = (0.20 * vx, 0.20 * vy)
    right = (-0.20 * vx, -0.20 * vy)
    img[_triangle(px, py, tail, left, right)] = (*SOUTH, 255)
    img[_triangle(px, py, tip, left, right)] = (*NORTH, 255)
    img[r <= 0.09] = (*HUB, 255)
    # box filter down to the real size
    out = img.reshape(size, OVERSAMPLE, size, OVERSAMPLE, 4)
    alpha = out[..., 3:] / 255.0
    weight = alpha.mean(axis=(1, 3))
    rgb = (out[..., :3] * alpha).mean(axis=(1, 3)) / np.maximum(weight, 1e-9)
    return np.concatenate([rgb, weight * 255.0], axis=-1).round().astype(np.uint8)


def png(rgba: np.ndarray) -> bytes:
    h, w = rgba.shape[:2]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    header = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    images = [png(draw(s)) for s in SIZES]
    directory = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    body = b""
    for size, data in zip(SIZES, images, strict=True):
        side = 0 if size == 256 else size
        directory += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
        body += data
    OUT.write_bytes(directory + body)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
