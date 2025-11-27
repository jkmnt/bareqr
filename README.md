# bareqr - Bare headless QR code generator

**bareqr** was derived from [pyqrcode](https://github.com/lincolnloop/python-qrcode). I removed all dependencies, console scripts, renderers and made it completely headless.

The only thing it could do is to produce the bit matrix.

```python
from pprint import pprint
from bareqr import qrcode

qr = qrcode("1234")

pprint(qr.rows)
>>>
[[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
 [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
 [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
 ...
 [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]]

>>> pprint(qr.as_strings())

 ['111111100100101111111',
 '100000100000001000001',
 '101110100001101011101',
 ...
 '111111101100010100100']
```

Also **bareqr** is:

-   stateless
-   threadsafe
-   optimized there and here
-   PEP-8 compliant
-   fully typed
