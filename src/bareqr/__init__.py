"""Bare headless QR code generator"""

__version__ = "0.1.1"

from .corr import (
    CorrH as CORRECTION_H,
)
from .corr import (
    CorrL as CORRECTION_L,
)
from .corr import (
    CorrM as CORRECTION_M,
)
from .corr import (
    CorrQ as CORRECTION_Q,
)
from .data import (
    optimal_chunks,
)
from .extra import (
    as_ascii,
    as_png,
)
from .mask import MASK0, MASK1, MASK2, MASK3, MASK4, MASK5, MASK6, MASK7
from .qr import (
    BlanksCache,
    qrcode,
)

__all__ = [
    "CORRECTION_L",
    "CORRECTION_M",
    "CORRECTION_Q",
    "CORRECTION_H",
    "MASK0",
    "MASK1",
    "MASK2",
    "MASK3",
    "MASK4",
    "MASK5",
    "MASK6",
    "MASK7",
    "as_ascii",
    "as_png",
    "qrcode",
    "optimal_chunks",
    "BlanksCache",
]
