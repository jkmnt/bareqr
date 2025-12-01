# bareqr - Bare headless QR code generator

**bareqr** was derived from [python-qrcode](https://github.com/lincolnloop/python-qrcode). I removed all dependencies, console scripts, renderers,
and made it headless.

**bareqr** is:

-   stateless
-   threadsafe
-   PEP-8 compliant
-   fully typed

## Main features

There is just one feature: produce the matrix.

```python
from bareqr import qrcode
from pprint import pprint

qr = qrcode("1234")

pprint(qr.rows)
>>>
[[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
 [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
 [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
 ...
 [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]]

```

**bareqr** won't automatically split data to chunks of different types.
You may split data manually:

```python
qr = qrcode("1234", "HELLO", "world")
```

For each chunk **bareqr** will choose the correct data type: number, alphanumeric or bytes.

The automatic splitting is available as a separate utility:

```python
from bareqr import optimal_chunks

chunks = optimal_chunks("1234HELLOworld", min_chunk=4)
qr = qrcode(*chunks)
```

`qrcode` has an internal global cache of precomputed blank QR codes.
In **bareqr** this cache is external and opt-in:

```python
cache = {}
qr1 = qrcode("1234", blanks_cache=cache)
qr2 = qrcode("5678", blanks_cache=cache)
```

## Extra features

Ok, in fact **bareqr** is not completely headless.
There are two utils to render QR as ascii or png. Treat them as handy extras.

```python
from bareqr import qrcode, as_ascii, as_png
qr = qrcode("1234")

asc = as_ascii(qr, border=2, invert=True)
for row in asc:
    print(row)

png = as_png(qr, border=2, scale=4)
with open("qr.png", "wb") as f:
    f.write(png)

```

The png encoder is custom to avoid extra dependencies.
