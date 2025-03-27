#![cfg_attr(not(any(test, feature = "std")), no_std)]

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    UnexpectedEof,
    Malformed,
}

pub trait Reader {
    fn read(&mut self) -> u8;
}

pub trait Writer {
    fn write(&mut self, byte: u8);
    fn get(&mut self, offset: usize) -> u8;
}

#[cfg(feature = "std")]
impl Writer for Vec<u8> {
    fn write(&mut self, byte: u8) {
        self.push(byte)
    }

    fn get(&mut self, offset: usize) -> u8 {
        self[self.len() - offset]
    }
}

struct BitMuncher<'a> {
    data: &'a [u8],
    pos: usize,
    val: u16,
    val_bits: u8,
}

impl<'a> BitMuncher<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            val: 0,
            val_bits: 0,
        }
    }

    fn read(&mut self, bits: u8) -> Result<u8, Error> {
        if self.val_bits < bits {
            if self.pos >= self.data.len() {
                return Err(Error::UnexpectedEof);
            }

            self.val |= (self.data[self.pos] as u16) << self.val_bits;
            self.val_bits += 8;
            self.pos += 1;
        }

        let res = self.val & ((1 << bits) - 1);
        self.val_bits -= bits;
        self.val >>= bits;
        Ok(res as u8)
    }

    fn read_u8(&mut self) -> Result<u8, Error> {
        self.val_bits = 0; // Remove in-progress bit
        if self.pos + 1 > self.data.len() {
            return Err(Error::UnexpectedEof);
        }

        let res = self.data[self.pos];
        self.pos += 1;
        Ok(res)
    }

    fn read_u16(&mut self) -> Result<u16, Error> {
        self.val_bits = 0; // Remove in-progress bit
        if self.pos + 2 > self.data.len() {
            return Err(Error::UnexpectedEof);
        }

        let res = self.data[self.pos] as u16 | ((self.data[self.pos + 1] as u16) << 8);
        self.pos += 2;
        Ok(res)
    }
}

const MAX_SYMBOL_COUNT: usize = 288;
const MAX_SYMBOL_LEN: usize = 16;

struct Tree {
    // How many codes are there with length i
    len_count: [u16; MAX_SYMBOL_LEN],
    // Symbol corresponding to the lexicographically i-th code
    symbol: [u16; MAX_SYMBOL_COUNT],
}

impl Tree {
    fn zeroed() -> Self {
        Self {
            len_count: [0; MAX_SYMBOL_LEN],
            symbol: [0; MAX_SYMBOL_COUNT],
        }
    }

    fn decode(
        &mut self,
        munch: &mut BitMuncher,
        len_tree: &Tree,
        num_lits: usize,
    ) -> Result<(), Error> {
        let mut lens = [0u8; MAX_SYMBOL_COUNT];
        let mut i = 0;
        while i < num_lits {
            match decode_symbol(munch, len_tree)? {
                sym @ 0..=15 => {
                    lens[i] = sym as _;
                    i += 1;
                }
                16 => {
                    let n = 3 + munch.read(2)? as usize;
                    let last = lens[i - 1];
                    lens[i..i + n].fill(last);
                    i += n;
                }
                17 => {
                    let n = 3 + munch.read(3)? as usize;
                    i += n;
                }
                18 => {
                    let n = 11 + munch.read(7)? as usize;
                    i += n;
                }
                _ => unreachable!(),
            }
        }

        self.build(&lens);
        Ok(())
    }

    fn build(&mut self, lens: &[u8; MAX_SYMBOL_COUNT]) {
        self.len_count.fill(0);
        for &len in lens {
            self.len_count[len as usize] += 1;
        }
        self.len_count[0] = 0;

        let mut offs = [0u16; MAX_SYMBOL_LEN];
        let mut sum = 0;
        for i in 1..MAX_SYMBOL_LEN {
            offs[i] = sum;
            sum += self.len_count[i];
        }

        for (i, &len) in lens.iter().enumerate() {
            if len != 0 {
                self.symbol[offs[len as usize] as usize] = i as _;
                offs[len as usize] += 1;
            }
        }
    }

    fn fixed_len(&mut self) {
        self.len_count.fill(0);
        self.len_count[7] = 24;
        self.len_count[8] = 152;
        self.len_count[9] = 112;

        for i in 0..MAX_SYMBOL_COUNT {
            let offs = match i {
                0..=23 => 256 - 0,
                24..=167 => 0 - 24,
                168..=175 => 280 - 168,
                176..=287 => 144 - 176,
                _ => unreachable!(),
            };
            self.symbol[i] = (i as i32 + offs) as _;
        }
    }

    fn fixed_dist(&mut self) {
        self.len_count.fill(0);
        self.len_count[5] = 32;

        for i in 0..32 {
            self.symbol[i] = i as u16;
        }
    }
}

fn decode_symbol(munch: &mut BitMuncher, tree: &Tree) -> Result<u16, Error> {
    let mut code_len = 0;
    let mut code = 0;
    let mut sum = 0;

    loop {
        let bit = munch.read(1)? as i32;
        code = code * 2 + bit;
        code_len += 1;

        assert!(code_len < MAX_SYMBOL_LEN);

        code -= tree.len_count[code_len] as i32;
        sum += tree.len_count[code_len] as i32;
        if code < 0 {
            break;
        }
    }
    sum += code;

    assert!(sum >= 0 && (sum as usize) < tree.symbol.len());
    Ok(tree.symbol[sum as usize])
}

static LEN_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
static LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

static DIST_EXTRA_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
static DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

static HUFF_LEN_ORDER: [u8; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

pub fn inflate(data: &[u8], w: &mut impl Writer) -> Result<(), Error> {
    let munch = &mut BitMuncher::new(data);

    let mut lit_tree = Tree::zeroed();
    let mut dist_tree = Tree::zeroed();

    loop {
        let block_final = munch.read(1)?;
        let block_type = munch.read(2)?;

        match block_type {
            0b00 => {
                let len = munch.read_u16()?;
                let nlen = munch.read_u16()?;
                assert_eq!(len, !nlen);

                for _ in 0..len {
                    w.write(munch.read_u8()?);
                }
            }
            0b01 | 0b10 => {
                if block_type == 0b01 {
                    lit_tree.fixed_len();
                    dist_tree.fixed_dist();
                } else {
                    let hlit = munch.read(5)? as usize + 257;
                    let hdist = munch.read(5)? as usize + 1;
                    let hclen = munch.read(4)? as usize + 4;

                    let mut lens = [0u8; MAX_SYMBOL_COUNT];
                    for i in 0..hclen {
                        lens[HUFF_LEN_ORDER[i] as usize] = munch.read(3)? as _;
                    }

                    let mut len_tree = Tree::zeroed();
                    len_tree.build(&lens);

                    lit_tree.decode(munch, &len_tree, hlit)?;
                    dist_tree.decode(munch, &len_tree, hdist)?;
                }

                loop {
                    match decode_symbol(munch, &lit_tree)? {
                        sym @ 0..=255 => w.write(sym as _),
                        256 => break,
                        sym @ 257..=285 => {
                            let sym = sym as usize - 257;
                            let len =
                                LEN_BASE[sym] as usize + munch.read(LEN_EXTRA_BITS[sym])? as usize;
                            let sym = decode_symbol(munch, &dist_tree)? as usize;
                            let dist = DIST_BASE[sym] as usize
                                + munch.read(DIST_EXTRA_BITS[sym])? as usize;
                            for _ in 0..len {
                                let byte = w.get(dist);
                                w.write(byte);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }
            _ => unreachable!(),
        }

        if block_final == 1 {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod test_cases;

#[cfg(test)]
mod test {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn test_inflate() {
        for (deflated, raw) in test_cases::TEST_CASES {
            let mut got: Vec<u8> = Vec::new();
            inflate(deflated, &mut got).unwrap();
            assert_eq!(&got, raw);
        }
    }
}
