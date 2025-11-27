"""Bare headless QR code generator"""

from __future__ import annotations

__version__ = "0.1.0"


from bisect import bisect_left
from typing import Final, Sequence, cast

from . import exceptions, util
from .base import VERSIONS, Correction


def _choose_version(*data_list: util.QRData, start_version: int, limits_by_version: Sequence[int]) -> int:
    """
    Find the minimum size required to fit in the data.
    """

    # Corresponds to the code in util.create_data, except we don't yet know
    # version, so optimistically assume start and check later
    mode_sizes = util.get_mode_sizes_for_version(start_version)

    buffer = util.BitBuffer()
    for data in data_list:
        data.write(buffer, start_version)

    need_bits = buffer.length
    version = bisect_left(limits_by_version, need_bits, start_version)
    if version not in util.VERSIONS:
        raise exceptions.DataOverflowError()

    # Now check whether we need more bits for the mode sizes, recursing if
    # our guess was too low
    # XXX: identity compare ?
    if mode_sizes is not util.get_mode_sizes_for_version(version):
        version = _choose_version(*data_list, start_version=version, limits_by_version=limits_by_version)
    return version


def _choose_mask_pattern(data_bytes: bytes, version: int, cache: BlanksCache, correction: Correction):
    """
    Find the most efficient mask pattern.
    """
    min_lost_point = 0
    pattern = 0

    for pattern in util.MASK_PATTERNS:
        matrix = _compile(
            data_bytes,
            test=True,
            version=version,
            pattern=pattern,
            cache=cache,
            correction=correction,
        )

        lost_point = util.get_lost_point(matrix)

        if pattern == 0 or min_lost_point > lost_point:
            min_lost_point = lost_point
            pattern = pattern

    return pattern


class Matrix:
    __slots__ = ("version", "order", "_rows")

    def __init__(self, version: int, data: list[list[int | None]] | None = None):
        order = version * 4 + 17
        self.version: Final = version
        self.order: Final = order
        self._rows: Final[list[list[int | None]]]
        if data:
            self._rows = data
        else:
            self._rows = [[None] * order for _ in range(order)]

    def copy(self):
        return Matrix(self.version, [list(row) for row in self._rows])

    def _put_probe_pattern(self, row: int, col: int):
        for r in range(-1, 8):
            if row + r <= -1 or self.order <= row + r:
                continue

            for c in range(-1, 8):
                if col + c <= -1 or self.order <= col + c:
                    continue

                self._rows[row + r][col + c] = int(
                    (0 <= r <= 6 and c in (0, 6)) or (0 <= c <= 6 and r in (0, 6)) or (2 <= r <= 4 and 2 <= c <= 4)
                )

    def _put_all_probe_patterns(self):
        self._put_probe_pattern(0, 0)
        self._put_probe_pattern(self.order - 7, 0)
        self._put_probe_pattern(0, self.order - 7)

    def _put_adjust_pattern(self):
        pos = util.get_adjust_pattern(self.version)

        for row in pos:
            for col in pos:
                if self._rows[row][col] is not None:
                    continue
                for r in range(-2, 3):
                    for c in range(-2, 3):
                        self._rows[row + r][col + c] = int(
                            r == -2 or r == 2 or c == -2 or c == 2 or (r == 0 and c == 0)
                        )

    def _put_timing_pattern(self):
        for r in range(8, self.order - 8):
            if self._rows[r][6] is not None:
                continue
            self._rows[r][6] = ~r & 1

        for c in range(8, self.order - 8):
            if self._rows[6][c] is not None:
                continue
            self._rows[6][c] = ~c & 1

    def _put_type_info(self, *, test: bool, pattern: int, correction: Correction):
        data = (correction << 3) | pattern
        bits = util.bch_type_info(data)

        # vertical
        for i in range(15):
            mod = 0 if test else ((bits >> i) & 1)

            if i < 6:
                self._rows[i][8] = mod
            elif i < 8:
                self._rows[i + 1][8] = mod
            else:
                self._rows[self.order - 15 + i][8] = mod

        # horizontal
        for i in range(15):
            mod = ((bits >> i) & 1) if test else 0

            if i < 8:
                self._rows[8][self.order - i - 1] = mod
            elif i < 9:
                self._rows[8][15 - i - 1 + 1] = mod
            else:
                self._rows[8][15 - i - 1] = mod

        # fixed module
        self._rows[self.order - 8][8] = 0 if test else 1

    def _put_type_number(self, *, test: bool):
        bits = util.bch_type_number(self.version)
        order = self.order

        for i in range(18):
            mod = 0 if test else ((bits >> i) & 1)
            self._rows[i // 3][i % 3 + order - 8 - 3] = mod

        for i in range(18):
            mod = 0 if test else ((bits >> i) & 1)
            self._rows[i % 3 + order - 8 - 3][i // 3] = mod

    def _put_data(self, data: bytes, *, pattern: int):
        inc = -1
        row = self.order - 1
        bit_index = 7
        byte_index = 0

        mask_func = util.get_mask_func(pattern)

        data_len = len(data)

        for col in range(self.order - 1, 0, -2):
            if col <= 6:
                col -= 1

            col_range = (col, col - 1)

            while True:
                for c in col_range:
                    if self._rows[row][c] is None:
                        bit = 0

                        if byte_index < data_len:
                            bit = (data[byte_index] >> bit_index) & 1

                        if mask_func(row, c):
                            bit = bit ^ 1

                        self._rows[row][c] = bit
                        bit_index -= 1

                        if bit_index == -1:
                            byte_index += 1
                            bit_index = 7

                row += inc

                if row < 0 or self.order <= row:
                    row -= inc
                    inc = -inc
                    break

    def as_matrix(self, border=4):
        if not border:
            return self._rows

        width = len(self._rows) + border * 2
        x_border = [0] * border

        code = [[0] * width] * border
        for row in self._rows:
            code.append(x_border + cast(list[int], row) + x_border)
        code += [[0] * width] * border

        return code

    def as_strings(self):
        return ["".join(str(m) for m in row) for row in self._rows]

    @property
    def rows(self):
        return cast(list[list[int]], self._rows)

    @classmethod
    def blank(cls, version: int):
        matrix = cls(version)
        matrix._put_all_probe_patterns()
        matrix._put_adjust_pattern()
        matrix._put_timing_pattern()
        return matrix


def _compile(data_bytes: bytes, *, test: bool, pattern: int, version: int, cache: BlanksCache, correction: Correction):
    if version in cache:
        matrix = cache[version].copy()
    else:
        matrix = Matrix.blank(version)
        cache[version] = matrix.copy()

    matrix._put_type_info(test=test, pattern=pattern, correction=correction)

    if version >= 7:
        matrix._put_type_number(test=test)

    matrix._put_data(data_bytes, pattern=pattern)
    return matrix


def qrcode(
    *data: str | bytes | util.QRData,
    version: int | None = None,
    error_correction=Correction.M,
    mask_pattern: int | None = None,
    min_chunk: int = 20,
    blanks_cache: BlanksCache | None = None,
):
    data_list: list[util.QRData] = []
    for item in data:
        if isinstance(item, util.QRData):
            data_list.append(item)
        else:
            if min_chunk:
                data_list.extend(util.QRData.in_optimal_data_chunks(item, min_chunk=min_chunk))
            else:
                data_list.append(util.QRData(item))

    if version is None:
        version = _choose_version(
            *data_list, start_version=1, limits_by_version=util.get_bit_capacity(error_correction)
        )
    assert version in VERSIONS

    data_bytes = util.create_data(
        *data_list,
        version=version,
        correction=error_correction,
    )

    if blanks_cache is None:
        blanks_cache = {}

    if mask_pattern is not None:
        assert mask_pattern in util.MASK_PATTERNS
    else:
        mask_pattern = _choose_mask_pattern(
            data_bytes, version=version, cache=blanks_cache, correction=error_correction
        )

    return _compile(
        data_bytes,
        test=False,
        pattern=mask_pattern,
        version=version,
        correction=error_correction,
        cache=blanks_cache,
    )


BlanksCache = dict[int, Matrix]
