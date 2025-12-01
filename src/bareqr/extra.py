from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

from .util import chunked

if TYPE_CHECKING:
    from .qr import Matrix


def transform(matrix: Matrix, *, border: int, scale=1, colors: tuple[int, int] | None = None):
    if not border and scale == 1 and not colors:
        return matrix.rows

    if not colors:
        colors = (0, 1)

    bg = [colors[0]]

    width = matrix.order * scale + border * 2
    v_border = bg * border
    h_borders = [bg * width] * border

    out: list[list[int]] = []

    out.extend(h_borders)
    for row in matrix.rows:
        transformed: list[int] = []
        transformed.extend(v_border)
        for mod in row:
            color = colors[mod]
            transformed.extend([color] * scale)
        transformed.extend(v_border)
        out.extend([transformed] * scale)
    out.extend(h_borders)

    return out


def as_ascii(matrix: Matrix, *, invert=False, border=0, scale=1):

    codes: dict[tuple[int, ...], str] = {
        (0, 0): "\u00A0",
        (0, 1): "\u2584",
        (1, 0): "\u2580",
        (1, 1): "\u2588",
        (0,): "\u00A0",
        (1,): "\u2580",
    }

    rows = transform(matrix, border=border, colors=(1, 0) if invert else (0, 1), scale=scale)

    out: list[str] = []
    for row_pair in chunked(rows, 2):
        out.append("".join(codes[mod_pair] for mod_pair in zip(*row_pair, strict=False)))
    return out


def _png_block(name: bytes, data: bytes):
    content = name + data
    return [len(data).to_bytes(4), content, zlib.crc32(content).to_bytes(4)]


def as_png(matrix: Matrix, *, invert=False, border: int | None = None, scale=4):

    pixels = bytearray()

    rows = transform(matrix, border=border or 0, scale=scale, colors=(0, 255) if invert else (255, 0))

    for row in rows:
        pixels.append(0)
        pixels.extend(row)

    pixels = zlib.compress(pixels)

    size = len(rows)
    size_word = size.to_bytes(4)

    return b"".join(
        [
            b"\x89PNG\r\n\x1A\n",
            *_png_block(b"IHDR", size_word + size_word + b"\x08\0\0\0\0"),
            *_png_block(b"IDAT", pixels),
            *_png_block(b"IEND", b""),
        ]
    )
