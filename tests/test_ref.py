# Q&D check against the reference implementation of 'qrcode'

from random import randbytes, randint

import pytest
import qrcode
import qrcode.util
from qrcode import ERROR_CORRECT_H, ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, QRCode

import bareqr
from bareqr.extra import as_ascii
from bareqr.mask import MASK0, MASK1, MASK2, MASK3, MASK4, MASK5, MASK6, MASK7, Mask
from bareqr.qr import CorrectionType, Matrix


def to_bare(qr: QRCode):
    return [[1 if mod else 0 for mod in row] for row in qr.get_matrix()]


def print_ascii(mx: Matrix):
    for row in as_ascii(mx, invert=True):
        print(row)


MASKMAP = [
    (0, MASK0),
    (1, MASK1),
    (2, MASK2),
    (3, MASK3),
    (4, MASK4),
    (5, MASK5),
    (6, MASK6),
    (7, MASK7),
]

CORRMAP = [
    (ERROR_CORRECT_L, bareqr.CORRECTION_L),
    (ERROR_CORRECT_M, bareqr.CORRECTION_M),
    (ERROR_CORRECT_Q, bareqr.CORRECTION_Q),
    (ERROR_CORRECT_H, bareqr.CORRECTION_H),
]

VERSIONS = range(1, 41)


@pytest.mark.parametrize("mask", MASKMAP)
@pytest.mark.parametrize("corr", CORRMAP)
@pytest.mark.parametrize("version", VERSIONS)
def test_hello(*, mask: tuple[int, Mask], corr: tuple[int, CorrectionType], version: int):
    """permutation of mask/version/correction for the simple hello bytes"""
    golden = qrcode.QRCode(error_correction=corr[0], border=0, mask_pattern=mask[0], version=version)
    golden.add_data("hello", optimize=0)
    golden.make(fit=False)
    mx = to_bare(golden)

    dut = bareqr.qrcode("hello", error_correction=corr[1], mask_pattern=mask[1], version=version)

    assert mx == dut.rows


@pytest.mark.parametrize("corr", CORRMAP)
@pytest.mark.parametrize("version", VERSIONS)
def test_choose_mask(*, corr: tuple[int, CorrectionType], version: int):
    golden = qrcode.QRCode(error_correction=corr[0], border=0, version=version)
    golden.add_data("hello", optimize=0)
    golden.make(fit=False)
    mx = to_bare(golden)

    dut = bareqr.qrcode("hello", error_correction=corr[1], version=version)

    assert mx == dut.rows


@pytest.mark.parametrize("corr", CORRMAP)
@pytest.mark.parametrize(
    "data",
    [
        "111",
        "AAA",
        "bbb",
        "",
    ],
)
@pytest.mark.parametrize("version", (1, 9, 10, 26, 27, 28))
def test_data_type(*, corr: tuple[int, CorrectionType], data: str, version: int):
    golden = qrcode.QRCode(error_correction=corr[0], border=0, version=version)
    golden.add_data(data, optimize=0)
    golden.make(fit=False)
    mx = to_bare(golden)

    dut = bareqr.qrcode(data, error_correction=corr[1], version=version)

    assert mx == dut.rows


@pytest.mark.parametrize("corr", CORRMAP)
@pytest.mark.parametrize("data", [randbytes(randint(0, 1024)) for i_ in range(32)])
def test_data_type_random(*, corr: tuple[int, CorrectionType], data: str):
    golden = qrcode.QRCode(
        error_correction=corr[0],
        border=0,
    )
    golden.add_data(data, optimize=0)
    golden.make(fit=False)
    mx = to_bare(golden)

    dut = bareqr.qrcode(data, error_correction=corr[1])

    assert mx == dut.rows


@pytest.mark.parametrize("corr", CORRMAP)
@pytest.mark.parametrize("data", [randbytes(randint(0, 128)) for i_ in range(32)])
@pytest.mark.parametrize("opt", [randint(1, 32) for i_ in range(32)])
def test_data_type_random_opt(*, corr: tuple[int, CorrectionType], data: str, opt: int):
    golden = qrcode.QRCode(error_correction=corr[0], border=0)
    golden.add_data(data, optimize=opt)
    golden.make(fit=False)
    mx = to_bare(golden)

    dut = bareqr.qrcode(*bareqr.optimal_chunks(data, min_chunk=opt), error_correction=corr[1])

    assert mx == dut.rows
