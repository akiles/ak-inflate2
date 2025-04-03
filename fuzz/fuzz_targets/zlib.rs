#![no_main]

use ak_inflate2::Inflater;
use flate2::Compress;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use futures::executor::block_on;
use libfuzzer_sys::fuzz_target;
use std::io::prelude::*;

fuzz_target!(|data: &[u8]| {
    let mut out = Vec::new();
    let c = Compress::new_with_window_bits(Compression::new(9), false, 15);
    let mut z = ZlibEncoder::new_with_compress(&mut out, c);

    z.write_all(data).unwrap();
    let compressed = z.finish().unwrap();

    let mut inflated = Vec::new();
    let mut inf = Inflater::new();

    block_on(inf.inflate(compressed, &mut inflated)).unwrap();
    assert_eq!(data, &inflated[..]);
});
