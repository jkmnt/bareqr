from __future__ import annotations

import functools
import math
import re
from enum import IntFlag
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from . import Matrix

from . import base, exceptions
from .base import VERSIONS, Correction, RSBlock, chunked, rs_blocks

MASK_PATTERNS = range(0, 8)


# QR encoding modes.
class Mode(IntFlag):
    NUMBER = 1 << 0
    ALPHA_NUM = 1 << 1
    BYTE = 1 << 2
    KANJI = 1 << 3


# Encoding mode sizes.
MODE_SIZE_SMALL = {
    Mode.NUMBER: 10,
    Mode.ALPHA_NUM: 9,
    Mode.BYTE: 8,
    Mode.KANJI: 8,
}
MODE_SIZE_MEDIUM = {
    Mode.NUMBER: 12,
    Mode.ALPHA_NUM: 11,
    Mode.BYTE: 16,
    Mode.KANJI: 10,
}
MODE_SIZE_LARGE = {
    Mode.NUMBER: 14,
    Mode.ALPHA_NUM: 13,
    Mode.BYTE: 16,
    Mode.KANJI: 12,
}

ALPHA_NUM = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
RE_ALPHA_NUM = re.compile(b"^[" + re.escape(ALPHA_NUM) + rb"]*\Z")

# The number of bits for numeric delimited data lengths.
NUMBER_LENGTH = {3: 10, 2: 7, 1: 4}


ADJUST_PATTERN_TABLE: tuple[tuple[int, ...], ...] = (
    (),
    (6, 18),
    (6, 22),
    (6, 26),
    (6, 30),
    (6, 34),
    (6, 22, 38),
    (6, 24, 42),
    (6, 26, 46),
    (6, 28, 50),
    (6, 30, 54),
    (6, 32, 58),
    (6, 34, 62),
    (6, 26, 46, 66),
    (6, 26, 48, 70),
    (6, 26, 50, 74),
    (6, 30, 54, 78),
    (6, 30, 56, 82),
    (6, 30, 58, 86),
    (6, 34, 62, 90),
    (6, 28, 50, 72, 94),
    (6, 26, 50, 74, 98),
    (6, 30, 54, 78, 102),
    (6, 28, 54, 80, 106),
    (6, 32, 58, 84, 110),
    (6, 30, 58, 86, 114),
    (6, 34, 62, 90, 118),
    (6, 26, 50, 74, 98, 122),
    (6, 30, 54, 78, 102, 126),
    (6, 26, 52, 78, 104, 130),
    (6, 30, 56, 82, 108, 134),
    (6, 34, 60, 86, 112, 138),
    (6, 30, 58, 86, 114, 142),
    (6, 34, 62, 90, 118, 146),
    (6, 30, 54, 78, 102, 126, 150),
    (6, 24, 50, 76, 102, 128, 154),
    (6, 28, 54, 80, 106, 132, 158),
    (6, 32, 58, 84, 110, 136, 162),
    (6, 26, 54, 82, 110, 138, 166),
    (6, 30, 58, 86, 114, 142, 170),
)

G15 = (1 << 10) | (1 << 8) | (1 << 5) | (1 << 4) | (1 << 2) | (1 << 1) | (1 << 0)
G18 = (1 << 12) | (1 << 11) | (1 << 10) | (1 << 9) | (1 << 8) | (1 << 5) | (1 << 2) | (1 << 0)
G15_MASK = (1 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | (1 << 1)

PAD0 = 0xEC
PAD1 = 0x11


# def create_bytes(rs_blocks):
#     for r in range(len(rs_blocks)):
#         dcCount = rs_blocks[r].data_count
#         ecCount = rs_blocks[r].total_count - dcCount
#         rsPoly = base.Polynomial([1], 0)
#         for i in range(ecCount):
#             rsPoly = rsPoly * base.Polynomial([1, base.gexp(i)], 0)
#         return ecCount, rsPoly

# rsPoly_LUT = {}
# for version in range(1,41):
#     for error_correction in range(4):
#         rs_blocks_list = base.rs_blocks(version, error_correction)
#         ecCount, rsPoly = create_bytes(rs_blocks_list)
#         rsPoly_LUT[ecCount]=rsPoly.num
# print(rsPoly_LUT)

# Result. Usage: input: ecCount, output: Polynomial.num
# e.g. rsPoly = base.Polynomial(LUT.rsPoly_LUT[ecCount], 0)

# fmt: off
RS_POLY_LUT: dict[int, tuple[int, ...]] = {
    7:  (1, 127, 122, 154, 164, 11, 68, 117),
    10: (1, 216, 194, 159, 111, 199, 94, 95, 113, 157, 193),
    13: (1, 137, 73, 227, 17, 177, 17, 52, 13, 46, 43, 83, 132, 120),
    15: (1, 29, 196, 111, 163, 112, 74, 10, 105, 105, 139, 132, 151, 32, 134, 26),
    16: (1, 59, 13, 104, 189, 68, 209, 30, 8, 163, 65, 41, 229, 98, 50, 36, 59),
    17: (1, 119, 66, 83, 120, 119, 22, 197, 83, 249, 41, 143, 134, 85, 53, 125, 99, 79),
    18: (1, 239, 251, 183, 113, 149, 175, 199, 215, 240, 220, 73, 82, 173, 75, 32, 67, 217, 146),
    20: (1, 152, 185, 240, 5, 111, 99, 6, 220, 112, 150, 69, 36, 187, 22, 228, 198, 121, 121, 165, 174),
    22: (1, 89, 179, 131, 176, 182, 244, 19, 189, 69, 40, 28, 137, 29, 123, 67, 253, 86, 218, 230, 26, 145, 245),
    24: (1, 122, 118, 169, 70, 178, 237, 216, 102, 115, 150, 229, 73, 130, 72, 61, 43, 206, 1, 237, 247, 127, 217, 144, 117),
    26: (1, 246, 51, 183, 4, 136, 98, 199, 152, 77, 56, 206, 24, 145, 40, 209, 117, 233, 42, 135, 68, 70, 144, 146, 77, 43, 94),
    28: (1, 252, 9, 28, 13, 18, 251, 208, 150, 103, 174, 100, 41, 167, 12, 247, 56, 117, 119, 233, 127, 181, 100, 121, 147, 176, 74, 58, 197),
    30: (1, 212, 246, 77, 73, 195, 192, 75, 98, 5, 70, 103, 177, 22, 217, 138, 51, 181, 246, 72, 25, 18, 46, 228, 74, 216, 195, 11, 106, 130, 150),
}
# fmt: on


## indexed from version - 1


def _capacity_of_rs_blocks(blocks: Iterable[RSBlock]):
    return sum(block.data_count * 8 for block in blocks)


@functools.lru_cache(maxsize=len(Correction))
def get_bit_capacity(correction: Correction):
    caps = [_capacity_of_rs_blocks(rs_blocks(version, correction)) for version in VERSIONS]
    return 0, *caps


def bch_type_info(data: int):
    d = data << 10
    while bch_digit(d) - bch_digit(G15) >= 0:
        d ^= G15 << (bch_digit(d) - bch_digit(G15))

    return ((data << 10) | d) ^ G15_MASK


def bch_type_number(data: int):
    d = data << 12
    while bch_digit(d) - bch_digit(G18) >= 0:
        d ^= G18 << (bch_digit(d) - bch_digit(G18))
    return (data << 12) | d


def bch_digit(data: int):
    digit = 0
    while data != 0:
        digit += 1
        data >>= 1
    return digit


def get_adjust_pattern(version: int):
    return ADJUST_PATTERN_TABLE[version - 1]


MASK_FUNCS: tuple[Callable[[int, int], bool], ...] = (
    lambda i, j: (i + j) % 2 == 0,
    lambda i, j: i % 2 == 0,
    lambda i, j: j % 3 == 0,
    lambda i, j: (i + j) % 3 == 0,
    lambda i, j: (math.floor(i / 2) + math.floor(j / 3)) % 2 == 0,
    lambda i, j: (i * j) % 2 + (i * j) % 3 == 0,
    lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0,
    lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0,
)


def get_mask_func(pattern: int) -> Callable[[int, int], bool]:
    """
    Return the mask function for the given mask pattern.
    """
    return MASK_FUNCS[pattern]


def get_mode_sizes_for_version(version: int):
    if version < 10:
        return MODE_SIZE_SMALL
    elif version < 27:
        return MODE_SIZE_MEDIUM
    else:
        return MODE_SIZE_LARGE


def get_lost_point(matrix: Matrix):
    return (
        _lost_point_level1(matrix)
        + _lost_point_level2(matrix)
        + _lost_point_level3(matrix)
        + _lost_point_level4(matrix)
    )


def _lost_point_level1(matrix: Matrix):
    lost_point = 0

    rows = matrix._rows
    rows_count = matrix.order

    rows_range = range(rows_count)
    container = [0] * (rows_count + 1)

    for row in rows_range:
        this_row = rows[row]
        previous_color = this_row[0]
        length = 0
        for col in rows_range:
            if this_row[col] == previous_color:
                length += 1
            else:
                if length >= 5:
                    container[length] += 1
                length = 1
                previous_color = this_row[col]
        if length >= 5:
            container[length] += 1

    for col in rows_range:
        previous_color = rows[0][col]
        length = 0
        for row in rows_range:
            if rows[row][col] == previous_color:
                length += 1
            else:
                if length >= 5:
                    container[length] += 1
                length = 1
                previous_color = rows[row][col]
        if length >= 5:
            container[length] += 1

    lost_point += sum(container[each_length] * (each_length - 2) for each_length in range(5, rows_count + 1))

    return lost_point


def _lost_point_level2(matrix: Matrix):

    rows = matrix._rows
    rows_count = matrix.order
    rows_range = range(rows_count - 1)

    lost_point = 0

    for row in rows_range:
        this_row = rows[row]
        next_row = rows[row + 1]
        # use iter() and next() to skip next four-block. e.g.
        # d a f   if top-right a != b bottom-right,
        # c b e   then both abcd and abef won't lost any point.
        col_range_iter = iter(rows_range)
        for col in col_range_iter:
            top_right = this_row[col + 1]
            if top_right != next_row[col + 1]:
                # reduce 33.3% of runtime via next().
                # None: raise nothing if there is no next item.
                next(col_range_iter, None)
            elif top_right != this_row[col]:
                continue
            elif top_right != next_row[col]:
                continue
            else:
                lost_point += 3

    return lost_point


def _lost_point_level3(matrix: Matrix):
    # 1 : 1 : 3 : 1 : 1 ratio (dark:light:dark:light:dark) pattern in
    # row/column, preceded or followed by light area 4 modules wide. From ISOIEC.
    # pattern1:     10111010000
    # pattern2: 00001011101

    rows = matrix._rows
    rows_count = matrix.order

    rows_range = range(rows_count)
    rows_range_short = range(rows_count - 10)

    lost_point = 0

    for row in rows_range:
        this_row = rows[row]
        col_range_short_iter = iter(rows_range_short)
        col = 0
        for col in col_range_short_iter:
            if (
                not this_row[col + 1]
                and this_row[col + 4]
                and not this_row[col + 5]
                and this_row[col + 6]
                and not this_row[col + 9]
                and (
                    this_row[col + 0]
                    and this_row[col + 2]
                    and this_row[col + 3]
                    and not this_row[col + 7]
                    and not this_row[col + 8]
                    and not this_row[col + 10]
                    or not this_row[col + 0]
                    and not this_row[col + 2]
                    and not this_row[col + 3]
                    and this_row[col + 7]
                    and this_row[col + 8]
                    and this_row[col + 10]
                )
            ):
                lost_point += 40
            # horspool algorithm.
            # if this_row[col + 10]:
            #   pattern1 shift 4, pattern2 shift 2. So min=2.
            # else:
            #   pattern1 shift 1, pattern2 shift 1. So min=1.
            if this_row[col + 10]:
                next(col_range_short_iter, None)

    for col in rows_range:
        col_range_short_iter = iter(rows_range_short)
        row = 0
        for row in col_range_short_iter:
            if (
                not rows[row + 1][col]
                and rows[row + 4][col]
                and not rows[row + 5][col]
                and rows[row + 6][col]
                and not rows[row + 9][col]
                and (
                    rows[row + 0][col]
                    and rows[row + 2][col]
                    and rows[row + 3][col]
                    and not rows[row + 7][col]
                    and not rows[row + 8][col]
                    and not rows[row + 10][col]
                    or not rows[row + 0][col]
                    and not rows[row + 2][col]
                    and not rows[row + 3][col]
                    and rows[row + 7][col]
                    and rows[row + 8][col]
                    and rows[row + 10][col]
                )
            ):
                lost_point += 40
            if rows[row + 10][col]:
                next(col_range_short_iter, None)

    return lost_point


def _lost_point_level4(matrix: Matrix):
    rows = matrix._rows
    rows_count = matrix.order

    dark_count = sum(module for row in rows for module in row if module)
    percent = float(dark_count) / (rows_count**2)
    # Every 5% departure from 50%, rating++
    rating = int(abs(percent * 100 - 50) / 5)
    return rating * 10


def _optimal_split(data: bytes, pattern: re.Pattern[bytes]):
    while data:
        match = re.search(pattern, data)
        if not match:
            break
        start, end = match.start(), match.end()
        if start:
            yield False, data[:start]
        yield True, data[start:end]
        data = data[end:]
    if data:
        yield False, data


def to_bytestring(data: str | bytes):
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def optimal_mode(data: bytes):
    """
    Calculate the optimal mode for this chunk of data.
    """
    if data.isdigit():
        return Mode.NUMBER
    if RE_ALPHA_NUM.match(data):
        return Mode.ALPHA_NUM
    return Mode.BYTE


class QRData:
    """
    Data held in a QR compatible format.

    Doesn't currently handle KANJI.
    """

    __slots__ = ("mode", "data")

    def __init__(self, data: str | bytes, *, mode: Mode | None = None, check_data=True):
        """
        If ``mode`` isn't provided, the most compact QR data type possible is
        chosen.
        """
        data = to_bytestring(data)

        if mode is None:
            self.mode = optimal_mode(data)
        else:
            self.mode = mode
            if check_data and mode < optimal_mode(data):
                raise ValueError(f"Provided data can not be represented in mode {mode}")

        self.data = data

    def write(self, buffer: BitBuffer, version: int):
        buffer.put(self.mode, 4)
        buffer.put(len(self.data), get_mode_sizes_for_version(version)[self.mode])

        if self.mode == Mode.NUMBER:
            for chars in chunked(self.data, 3):
                bit_length = NUMBER_LENGTH[len(chars)]
                buffer.put(int(chars), bit_length)
        elif self.mode == Mode.ALPHA_NUM:
            for chars in chunked(self.data, 2):
                if len(chars) > 1:
                    buffer.put(ALPHA_NUM.find(chars[0]) * 45 + ALPHA_NUM.find(chars[1]), 11)
                else:
                    buffer.put(ALPHA_NUM.find(chars), 6)
        else:
            for c in self.data:
                buffer.put(c, 8)

    def __repr__(self):
        return repr(self.data)

    @staticmethod
    def in_optimal_data_chunks(data: str | bytes, min_chunk=4):
        """
        An iterator returning QRData chunks optimized to the data content.

        :param min_chunk: The minimum number of bytes in a row to split as a chunk.
        """
        data = to_bytestring(data)
        num_pattern = rb"\d"
        alpha_pattern = b"[" + re.escape(ALPHA_NUM) + b"]"
        if len(data) <= min_chunk:
            num_pattern = re.compile(b"^" + num_pattern + b"+$")
            alpha_pattern = re.compile(b"^" + alpha_pattern + b"+$")
        else:
            re_repeat = b"{" + str(min_chunk).encode("ascii") + b",}"
            num_pattern = re.compile(num_pattern + re_repeat)
            alpha_pattern = re.compile(alpha_pattern + re_repeat)
        num_bits = _optimal_split(data, num_pattern)
        for is_num, chunk in num_bits:
            if is_num:
                yield QRData(chunk, mode=Mode.NUMBER, check_data=False)
            else:
                for is_alpha, sub_chunk in _optimal_split(chunk, alpha_pattern):
                    mode = Mode.ALPHA_NUM if is_alpha else Mode.BYTE
                    yield QRData(sub_chunk, mode=mode, check_data=False)


# NOTE: this really should be backed by Python bigint, not bytearray
class BitBuffer:

    __slots__ = ("buffer", "length")

    def __init__(self):
        self.buffer = bytearray()
        self.length = 0

    def __repr__(self):
        return self.buffer.hex()

    def put(self, num: int, length: int):
        for i in range(length):
            self.put_bit((num >> (length - i - 1)) & 1)

    def put_bit(self, bit: int):
        buf_index = self.length // 8
        if len(self.buffer) <= buf_index:
            self.buffer.append(0)
        if bit:
            self.buffer[buf_index] |= 0x80 >> (self.length % 8)
        self.length += 1


def create_bytes(buffer: BitBuffer, rs_blocks: Iterable[RSBlock]):
    offset = 0

    max_dc_count = 0
    max_ec_count = 0

    dc_data: list[list[int]] = []
    ec_data: list[list[int]] = []

    for block in rs_blocks:
        dc_count = block.data_count
        ec_count = block.total_count - dc_count

        max_dc_count = max(max_dc_count, dc_count)
        max_ec_count = max(max_ec_count, ec_count)

        current_dc = [buffer.buffer[i + offset] for i in range(dc_count)]
        offset += dc_count

        # Get error correction polynomial.
        if ec_count in RS_POLY_LUT:
            poly = base.Polynomial(RS_POLY_LUT[ec_count], 0)
        else:
            poly = base.Polynomial([1], 0)
            for i in range(ec_count):
                poly = poly * base.Polynomial([1, base.gexp(i)], 0)

        raw_poly = base.Polynomial(current_dc, len(poly) - 1)

        mod_poly = raw_poly % poly
        current_ec: list[int] = []
        mod_offset = len(mod_poly) - ec_count
        for i in range(ec_count):
            mod_index = i + mod_offset
            current_ec.append(mod_poly[mod_index] if (mod_index >= 0) else 0)

        dc_data.append(current_dc)
        ec_data.append(current_ec)

    data = bytearray()
    for i in range(max_dc_count):
        for dc in dc_data:
            if i < len(dc):
                data.append(dc[i])
    for i in range(max_ec_count):
        for ec in ec_data:
            if i < len(ec):
                data.append(ec[i])

    return data


def create_data(
    *data_list: QRData,
    version: int,
    correction: Correction,
):
    buffer = BitBuffer()
    for data in data_list:
        data.write(buffer, version)

    # Calculate the maximum number of bits for the given version.
    rs_blocks = base.rs_blocks(version=version, correction=correction)
    bit_limit = _capacity_of_rs_blocks(rs_blocks)

    if buffer.length > bit_limit:
        raise exceptions.DataOverflowError()

    # Terminate the bits (add up to four 0s).
    for _ in range(min(bit_limit - buffer.length, 4)):
        buffer.put_bit(0)

    # Delimit the string into 8-bit words, padding with 0s if necessary.
    delimit = buffer.length % 8
    if delimit:
        for _ in range(8 - delimit):
            buffer.put_bit(1)

    # Add special alternating padding bitstrings until buffer is full.
    bytes_to_fill = (bit_limit - buffer.length) // 8
    for i in range(bytes_to_fill):
        if i % 2 == 0:
            buffer.put(PAD0, 8)
        else:
            buffer.put(PAD1, 8)

    return create_bytes(buffer, rs_blocks)
