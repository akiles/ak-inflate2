
import zlib
import os


def deflate_zlib(x):
    # The 2:-4 is because zlib stream has a header+crc, and we just want the raw deflate stream.
    # https://stackoverflow.com/questions/1089662/python-inflate-and-deflate-implementations
    return zlib.compress(x, level=9)[2:-4]


with open('app.bin', 'rb') as f:
    data = f.read()
data = deflate_zlib(data)
with open('app.bin.deflate', 'wb') as f:
    f.write(data)
