
import zlib
import os


def deflate_zlib(x):
    # The 2:-4 is because zlib stream has a header+crc, and we just want the raw deflate stream.
    # https://stackoverflow.com/questions/1089662/python-inflate-and-deflate-implementations
    return zlib.compress(x, level=9)[2:-4]


def test(raw):
    deflated = deflate_zlib(raw)
    print(f'(&hex!("{deflated.hex()}"), &hex!("{raw.hex()}")),')


print('use hex_literal::hex;')
print('pub static TEST_CASES: &[(&[u8], &[u8])] = &[')

test(b'')
test(b'\x00')
test(b'\x01')
test(b'\xff')
test(b'\x00\xff')
test(b'\xff\x00')
test(b'\x01\x02\x03\x04\x05\x06')
test(b'\x01\x02\x03\x04\x05\x06\x07\x08')

for i in range(100):
    test(b'a'*i)

for i in range(100):
    test(b'ab'*i)
    test(b'ab'*i + b'a')

for i in range(100):
    test(os.urandom(i))

print("];")
