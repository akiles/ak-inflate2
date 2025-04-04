#![cfg_attr(not(any(test, feature = "std")), no_std)]
use core::future::Future;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    UnexpectedEof,
    Malformed,
    Io,
}

pub trait Writer {
    type Error;
    fn write(&mut self, byte: u8) -> impl Future<Output = Result<(), Self::Error>>;
    fn get(&mut self, offset: usize) -> impl Future<Output = Result<u8, Self::Error>>;
}

#[cfg(feature = "std")]
impl Writer for Vec<u8> {
    type Error = core::convert::Infallible;
    async fn write(&mut self, byte: u8) -> Result<(), Self::Error> {
        self.push(byte);
        Ok(())
    }

    async fn get(&mut self, offset: usize) -> Result<u8, Self::Error> {
        Ok(self[self.len() - offset])
    }
}

struct BitMuncher<'a> {
    data: &'a [u8],
    pos: usize,
    val: u32,
    val_bits: u8,
}

#[derive(Clone, Copy, Debug)]
struct MunchState {
    pos: usize,
    val: u32,
    val_bits: u8,
}

impl<'a> BitMuncher<'a> {
    fn new(data: &'a [u8], state: MunchState) -> Self {
        Self {
            data,
            pos: 0,
            val: state.val,
            val_bits: state.val_bits,
        }
    }

    fn state(&self) -> MunchState {
        MunchState {
            val: self.val,
            val_bits: self.val_bits,
            pos: self.pos,
        }
    }

    fn read(&mut self, bits: u8) -> Result<u32, Error> {
        // println!("muncher: read {} {:?}", bits, self.state());
        while self.val_bits < bits {
            if self.pos >= self.data.len() {
                return Err(Error::UnexpectedEof);
            }

            let b = self.data[self.pos];
            self.val |= (b as u32) << self.val_bits;
            self.val_bits += 8;
            self.pos += 1;

            // println!("muncher: refill {} {:?}", b, self.state());
        }

        let res = self.val & ((1 << bits) - 1);
        self.val_bits -= bits;
        self.val >>= bits;

        // println!("muncher: read done {} {:?}", res, self.state());
        Ok(res)
    }

    fn read_u8(&mut self) -> Result<u8, Error> {
        if self.pos + 1 > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        self.val_bits = 0; // Remove in-progress bit
        self.val = 0;

        let res = self.data[self.pos];
        self.pos += 1;
        Ok(res)
    }

    fn read_u16(&mut self) -> Result<u16, Error> {
        if self.pos + 2 > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        self.val_bits = 0; // Remove in-progress bit
        self.val = 0;

        let res = self.data[self.pos] as u16 | ((self.data[self.pos + 1] as u16) << 8);
        self.pos += 2;
        Ok(res)
    }
}

const MAX_SYMBOL_COUNT: usize = 288;
const MAX_SYMBOL_LEN: usize = 16;

#[derive(Debug)]
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

    fn reset(&mut self) {
        self.len_count.copy_from_slice(&[0; MAX_SYMBOL_LEN]);
        self.symbol.copy_from_slice(&[0; MAX_SYMBOL_COUNT]);
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
        code = (code << 1) + bit;
        code_len += 1;

        // println!(
        //     "[decode] code {}, code_len {}, bit {}, sum {}",
        //     code, code_len, bit, sum
        // );

        assert!(code_len < MAX_SYMBOL_LEN);

        // let len_count = tree.len_count[code_len];
        // println!("len count: {}", len_count);
        code -= tree.len_count[code_len] as i32;
        sum += tree.len_count[code_len] as i32;
        // println!("after: code {}, sum {}", code, sum);
        if code < 0 {
            break;
        }
    }
    sum += code;
    // println!("end sum = {}", sum);

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

pub struct Inflater {
    lit_tree: Tree,
    dist_tree: Tree,
    state: State,
    final_block: bool,
    munch: MunchState,
}

#[derive(Debug, PartialEq)]
pub enum State {
    ReadHeader,
    ReadUncompressed { len: usize },
    ReadCompressedSymbol,
    ReadCompressed { dist: usize, len: usize },
    Done,
}

impl Inflater {
    pub fn new() -> Self {
        Self {
            dist_tree: Tree::zeroed(),
            lit_tree: Tree::zeroed(),
            state: State::ReadHeader,
            munch: MunchState {
                val_bits: 0,
                val: 0,
                pos: 0,
            },
            final_block: false,
        }
    }

    /// Inflate into writer from data.
    ///
    /// The returned status indicates if the inflate was done, or if more
    /// data is needed, in which case the amount of data consumed is returned.
    ///
    /// It's the callers responsibility to add more data to the buffer before calling inflate again.
    pub async fn inflate<W: Writer>(&mut self, data: &[u8], w: &mut W) -> Result<Status, Error> {
        self.munch.pos = 0;
        let munch = &mut BitMuncher::new(data, self.munch);
        while State::Done != self.state {
            match self.inflate_step(munch, w).await {
                Ok(_) => {}
                Err(Error::UnexpectedEof) => {
                    let consumed = self.munch.pos;
                    return Ok(Status::Fill { consumed });
                }
                Err(e) => return Err(e),
            }
        }
        Ok(Status::Done)
    }

    fn update_state(&mut self, state: State, munch: &mut BitMuncher<'_>) {
        self.state = state;
        self.munch = munch.state();
    }

    async fn inflate_step<W: Writer>(
        &mut self,
        munch: &mut BitMuncher<'_>,
        w: &mut W,
    ) -> Result<(), Error> {
        // println!("step state {:?}", self.state);
        match self.state {
            State::ReadHeader => {
                let block_final = munch.read(1)?;
                let block_type = munch.read(2)?;
                match block_type {
                    0b00 => {
                        let len = munch.read_u16()?;
                        let nlen = munch.read_u16()?;
                        assert_eq!(len, !nlen);
                        self.update_state(State::ReadUncompressed { len: len as usize }, munch);
                    }
                    0b01 => {
                        self.lit_tree.reset();
                        self.dist_tree.reset();
                        self.lit_tree.fixed_len();
                        self.dist_tree.fixed_dist();
                        self.update_state(State::ReadCompressedSymbol, munch);
                    }
                    0b10 => {
                        self.lit_tree.reset();
                        self.dist_tree.reset();
                        let hlit = munch.read(5)? as usize + 257;
                        let hdist = munch.read(5)? as usize + 1;
                        let hclen = munch.read(4)? as usize + 4;

                        let mut lens = [0u8; MAX_SYMBOL_COUNT];
                        for i in 0..hclen {
                            lens[HUFF_LEN_ORDER[i] as usize] = munch.read(3)? as _;
                        }

                        let mut len_tree = Tree::zeroed();
                        len_tree.build(&lens);

                        self.lit_tree.decode(munch, &len_tree, hlit)?;
                        self.dist_tree.decode(munch, &len_tree, hdist)?;
                        self.update_state(State::ReadCompressedSymbol, munch);
                    }
                    _ => unreachable!(),
                }
                self.final_block = block_final == 1;
            }
            State::ReadUncompressed { len } => {
                for i in 0..len {
                    w.write(munch.read_u8()?).await.map_err(|_| Error::Io)?;
                    self.update_state(State::ReadUncompressed { len: len - i - 1 }, munch);
                }
                let state = if self.final_block {
                    State::Done
                } else {
                    State::ReadHeader
                };
                self.update_state(state, munch);
            }
            State::ReadCompressedSymbol => match decode_symbol(munch, &self.lit_tree)? {
                sym @ 0..=255 => {
                    w.write(sym as _).await.map_err(|_| Error::Io)?;
                    self.munch = munch.state();
                }
                256 => {
                    let state = if self.final_block {
                        State::Done
                    } else {
                        State::ReadHeader
                    };
                    self.update_state(state, munch);
                }
                sym @ 257..=285 => {
                    let sym = sym as usize - 257;
                    let len = LEN_BASE[sym] as usize + munch.read(LEN_EXTRA_BITS[sym])? as usize;
                    let sym = decode_symbol(munch, &self.dist_tree)? as usize;
                    let dist_base = DIST_BASE[sym] as usize;
                    let bits_to_read = DIST_EXTRA_BITS[sym];
                    let dist_extra = munch.read(bits_to_read)? as usize;
                    let dist = dist_base + dist_extra;

                    self.update_state(State::ReadCompressed { dist, len }, munch);
                }
                _ => unreachable!(),
            },
            State::ReadCompressed { dist, len } => {
                for i in 0..len {
                    let byte = w.get(dist).await.map_err(|_| Error::Io)?;
                    w.write(byte).await.map_err(|_| Error::Io)?;
                    self.update_state(
                        State::ReadCompressed {
                            dist,
                            len: len - i - 1,
                        },
                        munch,
                    );
                }
                self.update_state(State::ReadCompressedSymbol, munch);
            }
            State::Done => {}
        }
        Ok(())
    }
}

/// Indicates status of inflation session
#[derive(Debug)]
pub enum Status {
    /// Decmpression is done.
    Done,
    /// Needs more data.
    Fill {
        /// Data consumed from the buffer so far
        consumed: usize,
    },
}

#[cfg(test)]
mod test_cases;

#[cfg(test)]
mod test {
    use super::*;
    use std::vec::Vec;

    #[futures_test::test]
    async fn test_inflate() {
        for (deflated, raw) in test_cases::TEST_CASES {
            let mut got: Vec<u8> = Vec::new();
            let mut inf = Inflater::new();
            inf.inflate(deflated, &mut got).await.unwrap();
            assert_eq!(&got, raw);
        }
    }

    #[futures_test::test]
    async fn test_inflate_stream() {
        for (deflated, raw) in test_cases::TEST_CASES {
            let mut got: Vec<u8> = Vec::new();
            let mut inf = Inflater::new();
            let mut rpos = 0;
            let mut blen = 0;
            let mut buf = [0; 5];
            loop {
                match inf.inflate(&buf[..blen], &mut got).await.unwrap() {
                    Status::Done => {
                        break;
                    }
                    Status::Fill { consumed } => {
                        buf.rotate_left(consumed);
                        blen = blen - consumed;

                        let to_copy = (deflated.len() - rpos).min(buf.len() - blen);
                        buf[blen..blen + to_copy].copy_from_slice(&deflated[rpos..rpos + to_copy]);
                        blen = blen + to_copy;
                        rpos += to_copy;
                    }
                }
            }
            assert_eq!(&got, raw, "Failed for input {:02x?}", deflated);
        }
    }

    #[futures_test::test]
    async fn test_inflate_example() {
        use flate2::Compress;
        use flate2::Compression;
        use flate2::write::ZlibEncoder;
        use hex_literal::hex;
        use std::io::prelude::*;
        let raw = hex!(
            "8b8bec8bffff8b8b8b4000000f0400000000ffe5ff6d6d0000000053530f0400000000ffe5ff6d6d0000000053535353535353535353535337000040535353535353ffffff1500005353533253ffff15000000ff6dff0f0000abb1b1b1b3b1b1b1b1b1b1b1b1b1b1b1b1b1b1b1b10000000000ffffff6df7f723f7f7f7f7f7ce0000000000a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3000000000000000000000000000000000000000000000000000000000000a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3000088f7f7f7f7f7f7f7ffffff0008fff7f7f743f7f7f7f7000000ffe5ff6d6d000000005353535353535353535353533600a60040535353535353ffff000007000040000000000037000040318b078b0000ff3740ffffff15000000ff6d4747474747474747474747474747474747474747474747ffffffffacf7f7f7f7f7f7f7ce0000000000000088f7f7f7f7f7f7f7ffffff0008fff7f7f743f7f7f7f70000006ddb0000ff3740ffffff15000000ff6dffffffff6df7f7f7f7f7f741f7f76d00000d000000ffffff330000408bffff37f4efefefefefefefefefefefefefefefeffffff7f7ffffffffffffffffffffffffffffffffffffffffffff0200000000000000ffffff6363636373636363ffffffaaffffffffffffbf48ff0000000000000000ffff000000000000004000ffffffff1efffffffffffffffffbffffffffffffffffffffffffffffffffffffffff8c3d025dffffffffffff6dffffffff6df774f7883f00001ad5ffff29fdd5fffdff29ffff2bfd4a692173f8f7f7f7f7ff5aff000000038b8bff3700004031052f492006138b6774f5ff3f00002998ffffff3d8b8b23c8153a720088fdd100005007070000002822071172988c07ff50000000fd8824000011dc360000ab939393939393408bffff37f4efefefefefefefefefefefefefefefeffffff7f7ffffffffffffffffffffffffffffffffffffffffffff0200000000000000ffffff6363636373636363ffffffaaffffffffffffbf48ff0000000000000000ffff000000000000004000ffffffff1efffffffffffffffffbffffffffffffffffffffffffffffffffffffffff8c3d025dffffffffffff6dffffffff6df774f7883f00001ad5ffff29fdd5fffdff29ffff2bfd4a692173f8f7f7f7f7ff5aff000000038b8bff3700004031052f492006138b6774f5ff3f00002998ffffff3d8b8b23c8153a720088fdd100005007070000002822071172988c07ff50000000fd8824000011dc360000ab9393939393939393939393939393939393939393939393939393939393939393930a72ff170a72ff05492000ff170d00000000000000000000000800003afffffff7f7f7f743f7f7f76d48000000ffe5ff6d6d87fff9ffffff00ff87ffffe500000000fffff7f7f70300000000000000f727f7f7f743f7f7f76d48000000000048488b8b23c8153a7200888b8bec8bff010b8b8b04000000006d6de500ffff0000005353535353535353535353533700004053535353535b5353533253535353008bff000000000007000040000000004000000000000037000040318bfffdd100005007070000002207117298ff50008b0000ff3740ffffff15000000ff6dffffffff020400fa0004000000000000ffff8b8b8b8bffff39004100000dff37400340ff30000092929292929292000000040000400092929292923c41f7f76d00000d00ff00ff5d0000000040318b9292929292929492929292929292929292929292929292929292928b06318b44b8929292925d0000ffffff330000408bffff37f4ff3b5b2f0a06318b44b8929292925d0000ffffff330a00408bffff37f4ff3b5b2f0a72ff1fa503a0b8052f4920061300407af9fd023d501c010000fcfcff3040ff00005d40ff6df7f79393939393939393939393939393939393939393939393939393930a72ff170a72ff05492000ff170d00000000000000000000000800003afffffff7f7f7f743f7f7f76d48000000ffe5ff6d6d87fff9ffffff00ff87ffffe500000000fffff7f7f70300000000000000f727f7f7f743f7f7f76d48000000000048488b8b23c8153a7200888b8bec8bff010b8b8b04000000006d6de500ffff0000005353535353535353535353533700004053535353535b5353533253535353008bff000000000007000040000000004000000000000037000040318bfffdd100005007070000002207117298ff50008b0000ff3740ffffff15000000ff6dffffffff020400fa0004000000000000ffff8b8b8b8bffff39004100000dff37400340ff30000092929292929292000000040000400092929292923c41f7f76d00000d00ff00ff5d0000000040318b9292929292929492929292929292929292929292929292929292928b06318b44b8929292925d0000ffffff330000408bffff37f4ff3b5b2f0a06318b44b8929292925d0000ffffff330a00408bffff37f4ff3b5b2f0a72ff1fa503a0b8052f4920061300407af9fd023d501c010000fcfcff3040ff00005d40ff6df7f7f7f7f7f7f7ce00000000000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffddf8117298fcff002c0100000000000004f97fffffffffff1717171788f7f7f7f7f7f7f7ffffff0008fff7f7f743f7f7f7f70000006d480000000000484848ffff6dff0001ff6df79af7ffffff4100000de200000000000000fffff7f7ff6363636373636363ffffffaaffffffffffff0000fd88230a24000011dcff8b00003600002b0a72ff1703ff3717171717ffffffff5353538b00005353536a6aca6aff00ffff000040bf0000ffff0005004000ffffff41f7f76dffffffffffffffffffffffffffffffffffffffffffff00000d00ff318bffff8b268b48ff31ffff17171788f7f7f7f7f7f7"
        );
        let c = Compress::new_with_window_bits(Compression::new(9), false, 15);
        let mut z = ZlibEncoder::new_with_compress(Vec::new(), c);
        z.write_all(&raw).unwrap();

        let deflated = z.finish().unwrap();
        println!("DEFLATED: '{}", hex::encode(&deflated[..]));
        std::fs::write("deflated.bin", &deflated).unwrap();

        let mut got: Vec<u8> = Vec::new();
        let mut inf = Inflater::new();
        inf.inflate(&deflated, &mut got).await.unwrap();
        assert_eq!(got.len(), raw.len());
        for i in 0..got.len() {
            let mut diff = "";
            if got[i] != raw[i] {
                diff = "DIFFERENT!";
            }
            println!("{} = {} {} {}", i, got[i], raw[i], diff);
        }
        assert_eq!(&got, &raw);
    }

    #[test]
    fn test_muncher_bitreading() {
        let data = [0x55, 0x55];
        for bits in 1..16 {
            println!("Reading {} bits", bits);
            let mut m = BitMuncher::new(
                &data,
                MunchState {
                    val_bits: 0,
                    val: 0,
                    pos: 0,
                },
            );

            let expected = 0x5555 & ((1 << bits) - 1);
            let val = m.read(bits).unwrap();
            assert_eq!(expected, val, "failed to read {} bits", bits);
        }
    }

    #[test]
    fn test_muncher_bitread2() {
        let data = [0x55, 0x55];
        let mut m = BitMuncher::new(
            &data,
            MunchState {
                val_bits: 0,
                val: 0,
                pos: 0,
            },
        );

        assert_eq!(0x5, m.read(4).unwrap());
        assert_eq!(0x55, m.read(8).unwrap());
        assert_eq!(0x5, m.read(4).unwrap());
    }

    #[test]
    fn test_muncher_bitread3() {
        let data = [0x55, 0x55, 0x55];
        let mut m = BitMuncher::new(
            &data,
            MunchState {
                val_bits: 0,
                val: 0,
                pos: 0,
            },
        );

        assert_eq!(0x15, m.read(6).unwrap());
        assert_eq!(0x555, m.read(12).unwrap());
        assert_eq!(0x15, m.read(6).unwrap());
    }

    #[test]
    fn test_muncher_bitread_mix() {
        let data = [0x55, 0xaa, 0x88];
        let mut m = BitMuncher::new(
            &data,
            MunchState {
                val_bits: 0,
                val: 0,
                pos: 0,
            },
        );

        assert_eq!(0x5, m.read(4).unwrap());
        assert_eq!(0xaa, m.read_u8().unwrap());
        assert_eq!(0x8, m.read(4).unwrap());
    }
}
