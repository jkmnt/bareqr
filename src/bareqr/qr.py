from __future__ import annotations

from bisect import bisect_left
from typing import Final, cast

from .corr import Corr, CorrM
from .data import BitBuffer, DataOverflowError, QRData, create_data, make_qrdata
from .mask import Mask, choose_mask_pattern
from .util import bch_type_info, bch_type_number, get_adjust_pattern_pos

CorrectionType = type[Corr]

VERSIONS = range(1, 41)


def choose_version(*data_list: QRData, start_version: int, corr: CorrectionType) -> int:
    """
    Find the minimum size required to fit in the data.
    """

    start_version_mode_sizes = [item.get_mode_size(start_version) for item in data_list]

    buffer = BitBuffer()
    for data in data_list:
        data.write(buffer, start_version)

    need_bits = buffer.length

    # NOTE: Python 3.13+ docs says the bisect is not threadsafe if used on a same sequence.
    version = bisect_left(
        range(VERSIONS.stop), need_bits, lo=start_version, key=lambda ver: corr.get_bit_capacity_for_version(ver)
    )
    if version not in VERSIONS:
        raise DataOverflowError()

    # Now check whether we need more bits for the mode sizes, recursing if
    # our guess was too low
    # XXX: dislike this comparison
    version_mode_sizes = [item.get_mode_size(version) for item in data_list]

    if start_version_mode_sizes != version_mode_sizes:
        version = choose_version(*data_list, start_version=version, corr=corr)
    return version


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
        pos = get_adjust_pattern_pos(self.version)

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

    def _put_type_info(self, *, test: bool, mask: Mask, corr: Corr):
        bits = bch_type_info((corr.code << 3) | mask.code)

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
            mod = 0 if test else ((bits >> i) & 1)

            if i < 8:
                self._rows[8][self.order - i - 1] = mod
            elif i < 9:
                self._rows[8][15 - i - 1 + 1] = mod
            else:
                self._rows[8][15 - i - 1] = mod

        # fixed module
        self._rows[self.order - 8][8] = 0 if test else 1

    def _put_type_number(self, *, test: bool):
        bits = bch_type_number(self.version)
        order = self.order

        for i in range(18):
            mod = 0 if test else ((bits >> i) & 1)
            self._rows[i // 3][i % 3 + order - 8 - 3] = mod

        for i in range(18):
            mod = 0 if test else ((bits >> i) & 1)
            self._rows[i % 3 + order - 8 - 3][i // 3] = mod

    def _put_data(self, data: bytes, *, mask: Mask):
        inc = -1
        row = self.order - 1
        bit_index = 7
        byte_index = 0

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

                        bit = bit ^ mask(row, c)

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


def compile(data_bytes: bytes, *, test: bool, mask: Mask, cache: BlanksCache, version: int, corr: Corr):

    if version in cache:
        matrix = cache[version].copy()
    else:
        matrix = Matrix.blank(version)
        cache[version] = matrix.copy()

    matrix._put_type_info(test=test, mask=mask, corr=corr)

    if version >= 7:
        matrix._put_type_number(test=test)

    matrix._put_data(data_bytes, mask=mask)
    return matrix


def qrcode(
    *data: str | bytes | QRData,
    version: int | None = None,
    error_correction: CorrectionType | None = None,
    mask_pattern: Mask | None = None,
    blanks_cache: BlanksCache | None = None,
):
    data_list = [make_qrdata(item) if isinstance(item, (str, bytes)) else item for item in data]

    if error_correction is None:
        error_correction = CorrM

    if version is None:
        version = choose_version(*data_list, start_version=1, corr=error_correction)
    else:
        assert version in VERSIONS

    corr = error_correction(version)

    data_bytes = create_data(
        *data_list,
        version=version,
        corr=corr,
    )

    if blanks_cache is None:
        blanks_cache = {}

    if mask_pattern is None:

        def _make_test_matrix(mask: Mask):
            return compile(data_bytes, test=True, mask=mask, cache=blanks_cache, version=version, corr=corr)

        mask_pattern = choose_mask_pattern(_make_test_matrix)

    return compile(data_bytes, test=False, mask=mask_pattern, corr=corr, version=version, cache=blanks_cache)


BlanksCache = dict[int, Matrix]
