use arrayvec::ArrayVec;
use rayon::iter::IndexedParallelIterator as _;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator as _;
use rayon::iter::ParallelIterator as _;

const LG_RADIX: usize = 8;
const MAX_PASSES: usize = 5;
const MAX_SORT_LEN: usize = 1 << (MAX_PASSES * LG_RADIX);

/// Parses a compile-time decimal override, falling back to `default`.
const fn tunable(s: Option<&str>, default: usize) -> usize {
    match s {
        None => default,
        Some(s) => {
            let b = s.as_bytes();
            let mut v = 0usize;
            let mut i = 0;
            while i < b.len() {
                assert!(b[i].is_ascii_digit());
                v = v * 10 + (b[i] - b'0') as usize;
                i += 1;
            }
            v
        }
    }
}

// Radix of the top-level parallel MSD pass. Wider than the LSD radix: it
// determines both the number of parallel sub-sorts and how many bits remain
// for the per-bucket LSD passes. 11 bits won a sweep of 8..=12 at 32e9
// elements on 360 cores: enough sub-sorts (2048) to balance wide machines and
// small enough sub-sorts to need only 3 LSD passes (ending in `data`, so no
// fix-up copy), while keeping the scatter's per-thread destination working
// set within L2.
const MSD_LG_RADIX: usize = tunable(option_env!("SORT_MSD_LG_RADIX"), 11);
const MSD_RADIX: usize = 1 << MSD_LG_RADIX;

// Upper bound on the parallel chunks of the MSD histogram/scatter phases; the
// effective count is capped by MIN_PAR_CHUNK_SIZE. 32 starved these phases on
// a 360-core machine; 512..=2048 measured equal, so 1024 leaves headroom.
const MAX_PAR_CHUNKS: usize = tunable(option_env!("SORT_MAX_PAR_CHUNKS"), 1024);

// Digit width of the LSD passes in the sequential sub-sorts. Wider digits mean
// fewer passes over the data at the cost of more scatter streams per pass.
const LSD_LG_RADIX: usize = tunable(option_env!("SORT_LSD_LG_RADIX"), LG_RADIX);
const LSD_RADIX: usize = 1 << LSD_LG_RADIX;
const LSD_MASK: usize = LSD_RADIX - 1;
const LSD_MAX_PASSES: usize = 64usize.div_ceil(LSD_LG_RADIX);

fn empty_mut_slice() -> &'static mut [u64] {
    &mut []
}

/// Sorts `data` using `scratch` (at least as long as `data`) as working space.
///
/// The caller provides the scratch buffer because it sees the surrounding
/// loops: allocating (and above all unmapping) a data-sized buffer per call
/// dwarfs the sort itself at large sizes (~33 s of munmap for a 256 GiB
/// scratch vs ~14 s of sorting), so the buffer should be allocated once,
/// preferably with huge pages and prefaulted (see `common::alloc_mmap`), and
/// reused across repetitions and passes.
pub fn sort_uniform_u64s(data: &mut [u64], num_bits: u32, scratch: &mut [u64]) {
    assert!(
        data.len() <= MAX_SORT_LEN,
        "Sorting {} elements, but we only support up to {} elements",
        data.len(),
        MAX_SORT_LEN
    );
    assert!(
        scratch.len() >= data.len(),
        "Scratch buffer ({} elements) smaller than data ({} elements)",
        scratch.len(),
        data.len()
    );
    let scratch = &mut scratch[..data.len()];
    // 8-bit-digit radix sort following the following algorithm:
    // * if the array is large enough for parallelism, do a single parallel MSD pass
    //   and then split into chunks and process each chunk in parallel.
    // * on each chunk, we do "diverting LSD" sort. Instead of e.g. doing the full 8
    //   LSD passes (on u64), we do a smaller number of passes so that on average
    //   the unsorted group size is <=1. We do these passes on the highest
    //   (remaining) digits.
    // * at this point, the array is approximately sorted. Fix up the rest by
    //   insertion sort, which can be fused with the final LSD pass through the
    //   array.
    const MIN_PAR_CHUNK_SIZE: usize = 1024 * 1024;

    let verbose = std::env::var_os("SORT_VERBOSE").is_some();
    let mut sw = std::time::Instant::now();
    let mut lap = |label: &str| {
        if verbose {
            eprint!("{{{label}: {:.3}s}} ", sw.elapsed().as_secs_f64());
        }
        sw = std::time::Instant::now();
    };

    if data.len() < MIN_PAR_CHUNK_SIZE {
        sort_sequential(data, scratch, false, num_bits);
        return;
    }

    // If num_bits <= MSD_LG_RADIX, the MSD pass consumes all bits: shift is 0,
    // every key is its own digit (the high buckets stay empty), and the
    // sub-sorts get num_bits == 0 (each bucket holds equal keys only).
    let shift = (num_bits as usize).saturating_sub(MSD_LG_RADIX) as u32;
    let mask = (MSD_RADIX - 1) as u64;

    let par_chunk_size = MIN_PAR_CHUNK_SIZE.max(data.len().div_ceil(MAX_PAR_CHUNKS));
    let num_par_chunks = data.len().div_ceil(par_chunk_size);
    assert!(num_par_chunks <= MAX_PAR_CHUNKS);
    let par_chunks: ArrayVec<&[u64], MAX_PAR_CHUNKS> = data.chunks(par_chunk_size).collect();

    // Form histograms within each parallel chunk.
    let histograms: Vec<[usize; MSD_RADIX]> = par_chunks
        .par_iter()
        .map(|par_chunk| {
            let mut hist = [0usize; MSD_RADIX];
            let seq_chunk_iter = par_chunk.chunks_exact(4);
            for chunk in seq_chunk_iter.clone() {
                hist[((chunk[0] >> shift) & mask) as usize] += 1;
                hist[((chunk[1] >> shift) & mask) as usize] += 1;
                hist[((chunk[2] >> shift) & mask) as usize] += 1;
                hist[((chunk[3] >> shift) & mask) as usize] += 1;
            }
            for v in seq_chunk_iter.remainder() {
                hist[((v >> shift) & mask) as usize] += 1;
            }
            hist
        })
        .collect();
    lap("msd_hist");

    // Deal scratch ranges to parallel chunks. Also form a global histogram.
    //
    // This is the sequential bottleneck.
    let mut histogram = vec![0usize; MSD_RADIX];
    let mut ranges: Vec<[&mut [u64]; MSD_RADIX]> = Vec::with_capacity(num_par_chunks);
    for _ in 0..num_par_chunks {
        ranges.push(std::array::from_fn(|_| empty_mut_slice()));
    }
    let mut dst = &mut *scratch;
    for d in 0..MSD_RADIX {
        let mut count = 0;
        for (ranges, hist) in ranges.iter_mut().zip(histograms.iter()) {
            let (range, rest) = dst.split_at_mut(hist[d]);
            ranges[d] = range;
            dst = rest;
            count += hist[d];
        }
        histogram[d] = count;
    }
    lap("deal_ranges");

    // MSD pass on each par_chunk, writing to ranges in scratch. The scatter
    // writes elements directly: a thread's MSD_RADIX active destination lines
    // fit in L2, and software write-combining measured slower here.
    par_chunks
        .into_par_iter()
        .zip(ranges)
        .for_each(|(src, dst_ranges)| {
            let mut dst_ranges = dst_ranges.map(|range| range.into_iter());
            let mut deal = |word: u64| {
                let d = ((word >> shift) & mask) as usize;
                *dst_ranges[d].next().unwrap() = word;
            };
            let seq_chunk_iter = src.chunks_exact(4);
            for chunk in seq_chunk_iter.clone() {
                deal(chunk[0]);
                deal(chunk[1]);
                deal(chunk[2]);
                deal(chunk[3]);
            }
            for v in seq_chunk_iter.remainder() {
                deal(*v);
            }
        });

    lap("msd_scatter");

    // Split scratch and data into digits.
    drop(par_chunks);
    let mut scratch_slice = scratch;
    let mut data_slice = data;
    let mut sub_sort_chunks: Vec<(&mut [u64], &mut [u64])> = Vec::with_capacity(MSD_RADIX);
    for d in 0..MSD_RADIX {
        let (scratch_range, scratch_rest) = scratch_slice.split_at_mut(histogram[d]);
        let (data_range, data_rest) = data_slice.split_at_mut(histogram[d]);
        scratch_slice = scratch_rest;
        data_slice = data_rest;
        sub_sort_chunks.push((scratch_range, data_range));
    }

    // Sort each digit.
    sub_sort_chunks
        .into_par_iter()
        .for_each(|(scratch_range, data_range)| {
            sort_sequential(scratch_range, data_range, true, shift);
        });
    lap("sub_sorts");
}

// If "needs_final_copy", then we need to copy from src to scratch after
// sorting.
fn sort_sequential(
    src: &mut [u64],
    scratch: &mut [u64],
    want_result_in_scratch: bool,
    num_bits: u32,
) {
    assert!(src.len() == scratch.len());
    if src.len() <= 256 {
        src.sort_unstable();
        if want_result_in_scratch {
            scratch.copy_from_slice(src);
        }
        return;
    }
    if num_bits == 0 {
        if want_result_in_scratch {
            scratch.copy_from_slice(src);
        }
        return;
    }

    // Limit the number of passes to:
    // * get LSD groups down to size <=1 on average, so that the final insertion
    //   sort is very fast
    // * at most num_bits / LSD_LG_RADIX
    let passes = (src.len().ilog2().min(num_bits) as usize).div_ceil(LSD_LG_RADIX);

    let len = src.len();
    // We don't support arrays bigger than 2^32 elements sequentially, and somewhat
    // larger in parallel, although worst case (all zeros) is still a limit of
    // 2^32.
    assert!(
        (1..=LSD_MAX_PASSES).contains(&passes),
        "passes={passes}, num_bits={num_bits}, len={len}"
    );

    let shift = (num_bits as usize).saturating_sub(passes * LSD_LG_RADIX);
    let mut hists = vec![[0usize; LSD_RADIX]; passes];

    // Gather histograms for all passes, in a single pass through the data.
    let mut record_stats = |word: u64| {
        let mut d = (word >> shift) as usize;
        for hist in hists.iter_mut() {
            hist[d & LSD_MASK] += 1;
            d >>= LSD_LG_RADIX;
        }
    };
    let chunks = src.chunks_exact(4);
    for chunk in chunks.clone() {
        record_stats(chunk[0]);
        record_stats(chunk[1]);
        record_stats(chunk[2]);
        record_stats(chunk[3]);
    }
    for v in chunks.remainder() {
        record_stats(*v);
    }
    let is_in_scratch = (passes & 1) == 1;
    let should_copy = is_in_scratch != want_result_in_scratch;
    let need_final_sort = shift != 0;
    let last_lsd_pass_should_sort = need_final_sort && !should_copy;
    let normal_lsd_passes = passes - (if last_lsd_pass_should_sort { 1 } else { 0 });
    drop(record_stats);
    // Perform passes, from LSD to MSD.
    let mut from = src;
    let mut to = scratch;
    for pass in 0..normal_lsd_passes {
        let heads = &mut hists[pass];
        // Prefix scan over histogram to form starting positions for each digit.
        let mut pos = 0;
        for head in heads.iter_mut() {
            let len = *head;
            *head = pos;
            pos += len;
        }
        let pass_shift = shift + pass * LSD_LG_RADIX;
        // Deal into digits.
        let mut deal = |word: u64| {
            let d = (word >> pass_shift) as usize & LSD_MASK;
            let pos = heads[d];
            heads[d] += 1;
            to[pos] = word;
        };
        let chunks = from.chunks_exact(4);
        for chunk in chunks.clone() {
            deal(chunk[0]);
            deal(chunk[1]);
            deal(chunk[2]);
            deal(chunk[3]);
        }
        for v in chunks.remainder() {
            deal(*v);
        }
        drop(deal);
        // Swap from and to.
        std::mem::swap(&mut from, &mut to);
    }
    if last_lsd_pass_should_sort {
        let pass = normal_lsd_passes;
        // Unlike a normal LSD pass, we need to know the beginnings of the heads, not
        // just the ends.
        let mut heads: [(usize, usize); LSD_RADIX] = [(0, 0); LSD_RADIX];
        // Prefix scan over histogram to form starting positions for each digit.
        let mut pos = 0;
        for (&len, head) in hists[pass].iter().zip(heads.iter_mut()) {
            *head = (pos, pos);
            pos += len;
        }
        let pass_shift = shift + pass * LSD_LG_RADIX;
        // Deal into digits.
        let mut deal = |word: u64| {
            let d = (word >> pass_shift) as usize & LSD_MASK;
            let (head_start, pos) = heads[d];
            heads[d].1 += 1;
            // Insertion sort backwards towards the beginning of the group.
            let mut j = pos;
            while j > head_start && to[j - 1] > word {
                to[j] = to[j - 1];
                j -= 1;
            }
            to[j] = word;
        };
        let chunks = from.chunks_exact(4);
        for chunk in chunks.clone() {
            deal(chunk[0]);
            deal(chunk[1]);
            deal(chunk[2]);
            deal(chunk[3]);
        }
        for v in chunks.remainder() {
            deal(*v);
        }
        drop(deal);
        // Swap from and to.
        std::mem::swap(&mut from, &mut to);
    }
    if should_copy {
        if need_final_sort {
            // Insertion sort while copying.
            assert!(from.len() == to.len());
            for i in 0..from.len() {
                let cur = from[i];
                let mut j = i;
                while j > 0 && to[j - 1] > cur {
                    to[j] = to[j - 1];
                    j -= 1;
                }
                to[j] = cur;
            }
        } else {
            // Just copy.
            to.copy_from_slice(from);
        }
    }
}

#[cfg(test)]
mod tests {
    use rayon::slice::ParallelSliceMut as _;

    use super::*;

    const WY_CONST_0: u64 = 0x2d35_8dcc_aa6c_78a5;
    const WY_CONST_1: u64 = 0x8bb8_4b93_962e_acc9;

    struct WyRand(u64);

    impl WyRand {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            let s = self.0.wrapping_add(WY_CONST_0);
            self.0 = s;
            let t = (s as u128) * ((s ^ WY_CONST_1) as u128);
            ((t >> 64) as u64) ^ (t as u64)
        }
    }

    fn test_array(data: &mut [u64], num_bits: u32) {
        let mut data_clone = data.to_vec();
        data_clone.par_sort_unstable();
        let mut scratch = vec![0u64; data.len()];
        sort_uniform_u64s(data, num_bits, &mut scratch);
        let len = data.len();
        if data_clone != data {
            println!("Failure at num_bits={num_bits}, len={len}");
            println!("      expected          actual");
            for i in 0..len.min(100) {
                println!(
                    "{i:4}: 0x{:016x}   0x{:016x}{}",
                    data_clone[i],
                    data[i],
                    if data_clone[i] == data[i] {
                        ""
                    } else {
                        " ****"
                    }
                );
            }
        }
        assert!(
            data_clone == data,
            "Failure at num_bits={num_bits}, len={len}"
        );
    }

    fn test_size_and_width(size: usize, bit_width: u32) {
        let mut rng = WyRand::new(size as u64 * 64 + bit_width as u64);
        let mask = if bit_width == 64 {
            u64::MAX
        } else {
            (1u64 << bit_width) - 1
        };
        let mut data: Vec<u64> = (0..size).map(|_| rng.next_u64() & mask).collect();
        test_array(&mut data, bit_width);
    }

    #[test]
    fn test_one() {
        test_size_and_width(1 << 25, 44);
    }

    #[test]
    fn test_power_of_two_sizes() {
        let limit = if cfg!(debug_assertions) { 13 } else { 27 };

        // Test sizes 2^0 to 2^30
        for lg_size in 0..=limit {
            let size = 1usize << lg_size;
            // Test size-1, size, size+1
            for delta in [-1isize, 0, 1] {
                let test_size = (size as isize + delta) as usize;

                // Test bit widths at lg2(size) - 7, lg2(size), lg2(size) + 7
                for bit_width_delta in [-7, 0, 7] {
                    let bit_width = (lg_size + bit_width_delta).max(1).min(52) as u32;
                    test_size_and_width(test_size, bit_width);
                }
            }
        }
    }

    #[test]
    fn test_bit_widths() {
        // Test every multiple of 4 between 0 and 52
        let test_size = 1024;
        for bit_width in (0..=52).step_by(4) {
            if bit_width == 0 {
                continue;
            }
            test_size_and_width(test_size, bit_width as u32);
        }
    }
}
