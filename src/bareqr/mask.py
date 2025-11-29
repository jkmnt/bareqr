from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, cast

if TYPE_CHECKING:
    from .qr import Matrix

# keep it dead simple = let the masks be the function with extra 'code' attribute


class Mask(Protocol):
    code: int

    @classmethod
    def __call__(cls, /, i: int, j: int) -> bool: ...


def _make_mask(code: int, f: Callable[[int, int], bool]) -> Mask:
    f.code = code
    return cast(Mask, f)


MASK0 = _make_mask(0, lambda i, j: (i + j) % 2 == 0)
MASK1 = _make_mask(1, lambda i, j: i % 2 == 0)
MASK2 = _make_mask(2, lambda i, j: j % 3 == 0)
MASK3 = _make_mask(3, lambda i, j: (i + j) % 3 == 0)
MASK4 = _make_mask(4, lambda i, j: ((i // 2) + (j // 3)) % 2 == 0)
MASK5 = _make_mask(5, lambda i, j: (i * j) % 2 + (i * j) % 3 == 0)
MASK6 = _make_mask(6, lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0)
MASK7 = _make_mask(7, lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0)


# ---


def choose_mask_pattern(get_matrix: Callable[[Mask], Matrix]):
    candidates: list[tuple[int, int, Mask]] = []

    for mask in MASK0, MASK1, MASK2, MASK3, MASK4, MASK5, MASK6, MASK7:
        matrix = get_matrix(mask)

        penalty = (
            lost_point_level1(matrix)
            + lost_point_level2(matrix)
            + lost_point_level3(matrix)
            + lost_point_level4(matrix)
        )

        # mask.code is here to break ties if penalty is same
        candidates.append((penalty, mask.code, mask))

    return min(candidates)[2]


def lost_point_level1(matrix: Matrix):
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


def lost_point_level2(matrix: Matrix):

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


def lost_point_level3(matrix: Matrix):
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


def lost_point_level4(matrix: Matrix):
    rows = matrix._rows
    rows_count = matrix.order

    dark_count = sum(module for row in rows for module in row if module)
    percent = float(dark_count) / (rows_count**2)
    # Every 5% departure from 50%, rating++
    rating = int(abs(percent * 100 - 50) / 5)
    return rating * 10
