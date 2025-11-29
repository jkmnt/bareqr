from __future__ import annotations

import re
from typing import ClassVar, Final, Protocol

from .corr import Corr
from .util import chunked


class DataOverflowError(Exception):
    pass


def to_bytestring(data: str | bytes):
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def optimal_mode_cls(data: bytes):
    for cls in QRDataNumbers, QRDataAlnums, QRDataBytes:
        if cls.may_represent(data):
            return cls
    raise AssertionError("Can't choose the QRData class")


class QRData(Protocol):
    def write(self, buffer: BitBuffer, version: int): ...

    @classmethod
    def get_mode_size(cls, version: int) -> int: ...

    @classmethod
    def may_represent(cls, data: bytes) -> bool: ...


class QRDataNumbers:
    def __init__(self, data: bytes):
        self.data: Final = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    def write(self, buffer: BitBuffer, version: int):
        buffer.put(1 << 0, 4)
        buffer.put(len(self.data), self.get_mode_size(version))

        bit_lengths = (-1, 4, 7, 10)
        for chars in chunked(self.data, 3):
            buffer.put(int(chars), bit_lengths[len(chars)])

    @classmethod
    def get_mode_size(cls, version: int):
        if version < 10:
            return 10
        elif version < 27:
            return 12
        else:
            return 14

    @classmethod
    def may_represent(cls, data: bytes) -> bool:
        return data.isdigit()


class QRDataAlnums:

    CHARS: ClassVar = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    RE: ClassVar = re.compile(b"^[" + re.escape(CHARS) + rb"]*\Z")

    def __init__(self, data: bytes):
        self.data: Final = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    def write(self, buffer: BitBuffer, version: int):
        buffer.put(1 << 1, 4)
        buffer.put(len(self.data), self.get_mode_size(version))

        lut = self.CHARS

        for chars in chunked(self.data, 2):
            if len(chars) > 1:
                buffer.put(lut.index(chars[0]) * 45 + lut.index(chars[1]), 11)
            else:
                buffer.put(lut.index(chars), 6)

    @classmethod
    def get_mode_size(cls, version: int):
        if version < 10:
            return 9
        elif version < 27:
            return 11
        else:
            return 13

    @classmethod
    def may_represent(cls, data: bytes) -> bool:
        return cls.RE.match(data) is not None


class QRDataBytes:
    def __init__(self, data: bytes):
        self.data: Final = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    def write(self, buffer: BitBuffer, version: int):
        buffer.put(1 << 2, 4)
        buffer.put(len(self.data), self.get_mode_size(version))

        for c in self.data:
            buffer.put(c, 8)

    @classmethod
    def get_mode_size(cls, version: int):
        if version < 10:
            return 8
        else:
            return 16

    @classmethod
    def may_represent(cls, data: bytes) -> bool:
        return True


# NOTE: this really should be backed by Python bigint, not bytearray
class BitBuffer:

    __slots__ = ("buffer", "length")

    def __init__(self):
        self.buffer = bytearray()
        self.length = 0

    def __repr__(self):
        return self.buffer.hex()

    def put(self, num: int, length: int):
        for shift in range(length - 1, -1, -1):
            self.put_bit((num >> shift) & 1)

    def put_bit(self, bit: int):
        buf_index = self.length // 8
        if len(self.buffer) <= buf_index:
            self.buffer.append(0)
        if bit:
            self.buffer[buf_index] |= 0x80 >> (self.length % 8)
        self.length += 1


def create_data(
    *data_list: QRData,
    version: int,
    corr: Corr,
):
    buffer = BitBuffer()
    for data in data_list:
        data.write(buffer, version)

    bit_capacity = corr.get_bit_capacity()

    if buffer.length > bit_capacity:
        raise DataOverflowError()

    # Terminate the bits (add up to four 0s).
    for _ in range(min(bit_capacity - buffer.length, 4)):
        buffer.put_bit(0)

    # Delimit the string into 8-bit words, padding with 0s if necessary.
    delimit = buffer.length % 8
    if delimit:
        for _ in range(8 - delimit):
            buffer.put_bit(0)

    # Add special alternating padding bitstrings until buffer is full.
    bytes_to_fill = (bit_capacity - buffer.length) // 8
    for i in range(bytes_to_fill):
        buffer.put(0x11 if i % 2 else 0xEC, 8)

    return corr.apply_to(buffer.buffer)


def iter_optimal_splits(data: bytes, pattern: re.Pattern[bytes]):
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


def make_qrdata(data: str | bytes, cls: type[QRDataAlnums | QRDataNumbers | QRDataBytes] | None = None):
    data = to_bytestring(data)

    if cls is None:
        cls = optimal_mode_cls(data)
    else:
        if not cls.may_represent(data):
            raise ValueError(f"Provided data can not be represented in mode {cls}")
    return cls(data)


def optimal_chunks(data: str | bytes, *, min_chunk=4):
    """
    Returns list of QRData chunks optimized to the data content.

    :param min_chunk: The minimum number of bytes in a row to split as a chunk.
    """

    if min_chunk <= 0:
        raise ValueError("min chunk must be > 0")

    data = to_bytestring(data)
    num_re = rb"\d"
    alnum_re = b"[" + re.escape(QRDataAlnums.CHARS) + b"]"
    if len(data) <= min_chunk:
        num_re = re.compile(b"^" + num_re + b"+$")
        alnum_re = re.compile(b"^" + alnum_re + b"+$")
    else:
        re_repeat = b"{" + str(min_chunk).encode("ascii") + b",}"
        num_re = re.compile(num_re + re_repeat)
        alnum_re = re.compile(alnum_re + re_repeat)

    out: list[QRData] = []
    for is_num, chunk in iter_optimal_splits(data, num_re):
        if is_num:
            out.append(QRDataNumbers(chunk))
        else:
            for is_alnum, sub_chunk in iter_optimal_splits(chunk, alnum_re):
                out.append(QRDataAlnums(sub_chunk) if is_alnum else QRDataBytes(sub_chunk))
    return out
