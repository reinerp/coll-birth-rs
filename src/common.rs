/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Infrastructure shared by the collision and birthday-spacings tests.

use std::mem::size_of;

use mmap_rs::{MmapFlags, MmapMut, MmapOptions};
use num::BigUint;
use num::traits::ToPrimitive;
use rayon::prelude::*;

use crate::birthday::{run_birthday, run_birthday_tradeoff};
use crate::cell::{Cell, cell_index, decimate_candidate, decimate_once};
use crate::cli::Args;
use crate::collision::{run_collision, run_collision_decimate, run_collision_tradeoff};
use crate::prng::Prng;
use crate::stats::{expected_collisions, format_p_value, p_value};
use crate::util::{Stopwatch, parallelism, superscript};

/// Allocates an mmap-backed slice and eagerly faults it as transparent huge pages.
///
/// We deliberately do not pass [`MmapFlags::MAP_POPULATE`], as it would
/// prefault the whole region as base (4 KiB) pages during the `mmap()` syscall.
///
/// Instead, we use [`MmapFlags::TRANSPARENT_HUGE_PAGES`] and prefault the
/// region ourselves by touching one byte per 2 MiB, so each fault takes the
/// huge-page path. On a kernel with THP disabled the touches still prefault (as
/// base pages), so behaviour degrades gracefully rather than regressing.
///
/// [`MmapFlags::MAP_POPULATE`]: mmap.rs::MmapFlags::MAP_POPULATE
/// [`MmapFlags::TRANSPARENT_HUGE_PAGES`]: mmap.rs::MmapFlags::TRANSPARENT_HUGE_PAGES
pub(crate) fn alloc_mmap<T>(n: usize) -> MmapMut {
    let mut mapped = MmapOptions::new(n * size_of::<T>())
        .expect("mmap size overflow")
        .with_flags(MmapFlags::TRANSPARENT_HUGE_PAGES)
        .map_mut()
        .expect("mmap() failed");
    const HUGE_PAGE: usize = 2 * 1024 * 1024;
    let bytes: &mut [u8] = &mut mapped;
    bytes
        .par_chunks_mut(HUGE_PAGE)
        .for_each(|chunk| chunk[0] = 0);
    mapped
}

/// Buffer size needed for the active sampling mode.
///
/// A pass keeps the samples whose selection key (the *t* · *d* decimation
/// residue bits plus the *b* top tradeoff bits, `partition_bits` in all)
/// matches a fixed value, that is, one "bin" of a balls-into-bins experiment
/// with `points` balls and *n* = 2^`partition_bits` bins. A single bin load
/// is a sum of independent indicators, so writing *m* for its mean
/// `points`/*n*, Bernstein's inequality bounds Pr[load ≥ *m* + λ] by
/// exp(−λ²/(2(*m* + λ/3))); a union bound over the *n* bins then makes the
/// probability that *any* bin exceeds *m* + λ at most *n* · exp(−λ²/(2(*m* +
/// λ/3))), which is below 10⁻¹⁰⁰⁰ for λ = *L*/3 + √(*L*²/9 + 2·*m*·*L*) with
/// *L* = ln *n* + 1000 · ln 10 (the exact inversion of the exponent). The
/// headroom's shape is tight: by Theorem 1 of [Raab & Steger] the maximum load
/// actually reaches *m* + √(2·*m*·ln *n*) · (1 − o(1)) in the heavily-loaded
/// regime, so little can be shaved.
///
/// [Raab & Steger]: https://doi.org/10.1007/3-540-49543-6_13
pub fn buffer_size(points: usize, partition_bits: usize) -> usize {
    if partition_bits == 0 {
        return points;
    }
    let mean = points as f64 / 2.0f64.powi(partition_bits as i32);
    let ln_n = (partition_bits as f64) * std::f64::consts::LN_2;
    let l = ln_n + 1000.0 * std::f64::consts::LN_10;
    // Exact inversion of the Bernstein exponent: λ²/(2(mean + λ/3)) = L.
    let dev = l / 3.0 + (l * l / 9.0 + 2.0 * mean * l).sqrt();
    ((mean + dev).ceil() as usize).min(points)
}

/// Aborts cleanly when a sampling bin (tradeoff pass, decimation keep-set,
/// spacing class, …) overflows its [`buffer_size`]-sized buffer.
///
/// The headroom is set so that, for a uniform generator, a bin exceeds its
/// buffer with probability below 10⁻¹⁰⁰⁰ (see [`buffer_size`]). An overflow
/// therefore means the generator has just failed a trivial serial load-balance
/// test by an astronomical margin. Exits the whole process (also from a worker
/// thread) rather than panicking, so the message isn't buried under a
/// backtrace.
pub(crate) fn bin_overflow(what: &str) -> ! {
    eprintln!(
        "\n{what} overflowed its buffer: a bin received more elements than its \
         balls-into-bins headroom, which happens with probability below 10⁻¹⁰⁰⁰ \
         for a uniform generator: this is overwhelming evidence that the generator under \
         test is grossly non-uniform. Rerun in plain mode (no -b/-d) for an exact \
         p-value."
    );
    std::process::exit(1);
}

/// Samples scanned per pass: `points` · 2*ᵗᵈ*. Uses an overflow check:
/// [`checked_shl`] alone only guards the shift amount, not the resulting value, so
/// a too-large *t* · *d* would silently wrap. A configuration whose sample budget
/// does not fit in a usize (points · 2*ᵗᵈ* ≥ 2⁶⁴; points already carries the
/// 2*ᵇ* tradeoff factor) surfaces here as a clean overflow error.
///
/// [`checked_shl`]: u32::checked_shl
pub(crate) fn scan_samples(points: usize, t: usize, d: usize) -> usize {
    1usize
        .checked_shl((t * d) as u32)
        .and_then(|factor| points.checked_mul(factor))
        .expect("points · 2ᵗᵈ overflows usize")
}

/// "full 64-bit output" or "lowest N bits": which bits of each draw the test reads.
pub(crate) fn bits_read_desc(s: usize) -> String {
    if s == 0 {
        "full 64-bit output".to_string()
    } else {
        format!("lowest {} bits", 64 - s)
    }
}

/// The shared decimation descriptor for the header's mode list.
pub(crate) fn decimation_desc(d: usize, t: usize) -> String {
    format!(
        "decimating {} bits per dimension (~2{} candidate samples per kept sample)",
        d,
        superscript(d * t)
    )
}

/// Joins the per-mode descriptors into the header's trailing ", a, b" suffix (empty
/// when there are none).
pub(crate) fn join_mode_parts(parts: &[String]) -> String {
    if parts.is_empty() {
        String::new()
    } else {
        format!(", {}", parts.join(", "))
    }
}

/// The " (effective cells after decimation: 2*ᴺ*)" header note (empty when `d == 0`).
pub(crate) fn effective_cells_suffix(d: usize, u: usize, t: usize) -> String {
    if d > 0 {
        format!(
            " (effective cells after decimation: 2{})",
            superscript((u - d) * t)
        )
    } else {
        String::new()
    }
}

/// Geometric parameters of a test grid.
///
/// The whole-tuple decimation bits are encoded as a const generic on the test
/// runners (and on [`GridParams::draw`]) rather than a field, so the `B == 0`
/// fast path inside [`cell_index`] is selected at compile time.
pub struct GridParams<'a> {
    /// log₂ of the number of subdivisions per dimension (the *u* value).
    pub u: usize,
    /// Number of dimensions.
    pub t: usize,
    /// Left shift applied to each PRNG output before extracting cell bits.
    pub s: usize,
    /// Decimation bits: low *d* bits of each element forced to zero (0 = none).
    pub d: usize,
    /// Total cell count; needed by the birthday-spacings wrap-around.
    pub cells: &'a BigUint,
}

impl GridParams<'_> {
    #[inline]
    pub fn draw<T: Cell, const DIM: usize, const FULL: bool>(&self, prng: &mut Prng) -> T {
        cell_index::<T, DIM, FULL>(prng, self.t, self.u, self.s)
    }

    /// One decimating attempt (exactly *t* draws) returning the compacted dense
    /// index when accepted, `None` when rejected. See [`decimate_once`].
    #[inline]
    pub fn draw_decimate_once<T: Cell, const DIM: usize, const FULL: bool>(
        &self,
        prng: &mut Prng,
    ) -> Option<T> {
        decimate_once::<T, DIM, FULL>(prng, self.t, self.u, self.s, self.d)
    }

    /// One candidate sample (exactly *t* draws) with a *keep* flag instead of
    /// an `Option`: with `DECIMATE` the flag is the decimation acceptance,
    /// otherwise it is always true. See [`decimate_candidate`].
    #[inline]
    pub fn draw_candidate<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
        &self,
        prng: &mut Prng,
    ) -> (T, bool) {
        if DECIMATE {
            decimate_candidate::<T, DIM, FULL>(prng, self.t, self.u, self.s, self.d)
        } else {
            (cell_index::<T, DIM, FULL>(prng, self.t, self.u, self.s), true)
        }
    }
}

/// Branchless candidate scan shared by every generate loop: draws `scan_len`
/// candidate samples (exactly *t* PRNG draws each), keeping the decimation
/// survivors and, when `TRADEOFF`, only those whose top-*b* value-interval key
/// equals `pass`; kept samples are written densely to the front of `buf` and
/// the kept count is returned. `prng` is advanced in place by exactly
/// `scan_len` samples.
///
/// The hot loop has no data-dependent branch: whether a sample is kept is
/// random (probability 2^(-t·d), 2^(-b), or their product), so an if-around-
/// the-store mispredicts constantly. Instead every candidate is written
/// unconditionally at the cursor, and the cursor advances only when the sample
/// is kept:
///
/// ```text
/// write(dst, x); dst += keep as usize;
/// ```
///
/// Rejected values are simply overwritten by the next candidate (the slot is
/// re-written until something sticks, staying in L1), so the actual write
/// traffic is only the kept points. The keep mask and pass key are hoisted
/// out of the loop.
///
/// The scan runs in blocks: a block whose worst case fits the remaining
/// capacity takes the branchless path (the unconditional store needs
/// headroom); otherwise a careful per-element path preserves the exact
/// old overflow semantics (`bin_overflow` fires only when a *kept* point
/// exceeds `buf`). This also makes an exactly-sized `buf` safe: the last
/// few blocks fall back to the careful path and never write past the end.
#[inline]
pub(crate) fn scan_keep<
    T: Cell,
    const DIM: usize,
    const DECIMATE: bool,
    const FULL: bool,
    const TRADEOFF: bool,
>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    scan_len: usize,
    pass: u64,
    tradeoff_b: usize,
    overflow_what: &'static str,
) -> usize {
    let t = params.t;
    let d = params.d;
    let elem_width = if DECIMATE { params.u - d } else { params.u };
    // Pass k holds the points whose top b bits of the combined index equal k
    // (one contiguous value interval); see run_collision_tradeoff. Computed
    // unconditionally but only meaningful (and only applied) when TRADEOFF.
    let key_shift = if TRADEOFF {
        t * elem_width - tradeoff_b
    } else {
        0
    };
    let target = T::from_u64(pass);

    const BLOCK: usize = 4096;
    let cap = buf.len();
    let base = buf.as_mut_ptr();
    let mut len = 0usize;
    let mut remaining = scan_len;
    while remaining > 0 {
        let block = remaining.min(BLOCK);
        if len + block <= cap {
            // Fast path: even if every candidate in this block is kept it
            // fits, so the unconditional store below stays in bounds.
            let mut dst = unsafe { base.add(len) };
            for _ in 0..block {
                let (x, mut keep) = params.draw_candidate::<T, DIM, DECIMATE, FULL>(prng);
                if TRADEOFF {
                    let mut key = x;
                    key >>= key_shift;
                    keep &= key == target;
                }
                // SAFETY: dst stays within buf: it starts at base + len and
                // advances at most `block` times, and len + block <= cap.
                unsafe {
                    std::ptr::write(dst, x);
                    dst = dst.add(keep as usize);
                }
            }
            len = unsafe { dst.offset_from(base) } as usize;
        } else {
            // Careful path, reachable only within `block` slots of the end of
            // the headroom: keep-checked stores, overflow on the first kept
            // point past cap.
            for _ in 0..block {
                let (x, mut keep) = params.draw_candidate::<T, DIM, DECIMATE, FULL>(prng);
                if TRADEOFF {
                    let mut key = x;
                    key >>= key_shift;
                    keep &= key == target;
                }
                if keep {
                    *buf.get_mut(len)
                        .unwrap_or_else(|| bin_overflow(overflow_what)) = x;
                    len += 1;
                }
            }
        }
        remaining -= block;
    }
    len
}

/// Store-free twin of [`scan_keep`]: replays the same candidate stream and
/// returns only the kept count. Used as the first half of count-then-fill
/// generation (see [`gen_unit_contiguous`]): knowing each thread's exact kept
/// count lets threads write directly into exactly-sized disjoint regions,
/// removing the single-threaded gap-compaction memmove that otherwise
/// dominates the generate phase.
#[inline]
fn scan_count<
    T: Cell,
    const DIM: usize,
    const DECIMATE: bool,
    const FULL: bool,
    const TRADEOFF: bool,
>(
    mut prng: Prng,
    params: &GridParams,
    scan_len: usize,
    pass: u64,
    tradeoff_b: usize,
) -> usize {
    let t = params.t;
    let elem_width = if DECIMATE {
        params.u - params.d
    } else {
        params.u
    };
    let key_shift = if TRADEOFF {
        t * elem_width - tradeoff_b
    } else {
        0
    };
    let target = T::from_u64(pass);

    let mut count = 0usize;
    for _ in 0..scan_len {
        let (x, mut keep) = params.draw_candidate::<T, DIM, DECIMATE, FULL>(&mut prng);
        if TRADEOFF {
            let mut key = x;
            key >>= key_shift;
            keep &= key == target;
        }
        count += keep as usize;
    }
    count
}

/// Per-thread kept-count of one pass: the counting counterpart of
/// [`gen_pass_dispatch`], with the same `DIM`/`DECIMATE`/`FULL`/tradeoff
/// specialization matrix.
fn gen_count_dispatch<T: Cell>(
    snapshot: Prng,
    params: &GridParams,
    stream_len: usize,
    pass: u64,
    tradeoff_b: usize,
    decimating: bool,
    full: bool,
) -> usize {
    macro_rules! go {
        ($dim:literal) => {{
            if tradeoff_b > 0 {
                match (decimating, full) {
                    (true, true) => scan_count::<T, $dim, true, true, true>(
                        snapshot, params, stream_len, pass, tradeoff_b,
                    ),
                    (true, false) => scan_count::<T, $dim, true, false, true>(
                        snapshot, params, stream_len, pass, tradeoff_b,
                    ),
                    (false, true) => scan_count::<T, $dim, false, true, true>(
                        snapshot, params, stream_len, pass, tradeoff_b,
                    ),
                    (false, false) => scan_count::<T, $dim, false, false, true>(
                        snapshot, params, stream_len, pass, tradeoff_b,
                    ),
                }
            } else if full {
                scan_count::<T, $dim, true, true, false>(snapshot, params, stream_len, 0, 0)
            } else {
                scan_count::<T, $dim, true, false, false>(snapshot, params, stream_len, 0, 0)
            }
        }};
    }
    match params.t {
        1 => go!(1),
        2 => go!(2),
        3 => go!(3),
        4 => go!(4),
        5 => go!(5),
        6 => go!(6),
        7 => go!(7),
        8 => go!(8),
        _ => go!(0),
    }
}

/// Counts adjacent equal pairs in a sorted slice (i.e. one less than the
/// multiplicity sum).
///
/// Parallelized by contiguous chunks rather than `par_windows`: each chunk counts
/// the equal pairs strictly inside it, and the one pair straddling each chunk
/// boundary is added back (the border fix). Chunked streaming reads are far kinder
/// to cache and the memory subsystem than overlapping windows at very large `n`,
/// where `par_windows` becomes a bottleneck.
#[inline]
pub(crate) fn count_adjacent_equals<T: Cell>(v: &[T]) -> usize {
    if v.len() < 2 {
        return 0;
    }
    let chunk_size = (v.len() / (parallelism() * 10)).max(1024);
    // Pairs that lie wholly within a chunk.
    let within: usize = v
        .par_chunks(chunk_size)
        .map(|c| {
            let mut count = 0usize;
            for w in c.windows(2) {
                if w[0] == w[1] {
                    count += 1;
                }
            }
            count
        })
        .sum();
    // Pairs straddling a chunk boundary: this chunk's first vs. the previous tail.
    let borders: usize = v
        .par_chunks(chunk_size)
        .enumerate()
        .skip(1)
        .filter(|(i, c)| v[i * chunk_size - 1] == c[0])
        .count();
    within + borders
}

/// Three-pointer right-to-left merge of `buf[..prefix_len]` (sorted) with `src` (sorted)
/// into `buf[..prefix_len + src.len()]`.
///
/// Walking the write pointer from the high end downward means we only ever overwrite
/// positions strictly above any still-unread element of the prefix, so no swaps or
/// scratch slots are needed beyond `src` itself. Prefix elements that remain when `src`
/// is exhausted are already at their correct positions and require no copy.
pub(crate) fn merge_into<T: Cell>(buf: &mut [T], prefix_len: usize, src: &[T]) {
    let mut i = prefix_len;
    let mut j = src.len();
    let mut w = i + j;
    while i > 0 && j > 0 {
        w -= 1;
        if buf[i - 1] >= src[j - 1] {
            i -= 1;
            buf[w] = buf[i];
        } else {
            j -= 1;
            buf[w] = src[j];
        }
    }
    while j > 0 {
        w -= 1;
        j -= 1;
        buf[w] = src[j];
    }
}

/// The faithful contiguous-orbit partition shared by both parallel runners. A scan
/// of `scan_total` samples is split into `num_cpus` contiguous sample-ranges; each
/// range's start is reached by jump-ahead (`try_skip`) for skip-capable generators
/// or by a chained sequential pre-scan ([`prescan_checkpoints`]) otherwise.
pub(crate) struct OrbitPartition {
    pub(crate) num_cpus: usize,
    pub(crate) scan_total: usize,
    pub(crate) base_chunk: usize,
    pub(crate) rem: usize,
    t: usize,
    skip_capable: bool,
    seed: u64,
    /// Pre-scan path only: the orbit start of the next repetition, chained across
    /// reps (a non-jumpable generator cannot recompute an absolute offset).
    prescan_start: Prng,
}

impl OrbitPartition {
    pub(crate) fn new(seed: u64, num_cpus: usize, scan_total: usize, t: usize) -> Self {
        let skip_capable = {
            let mut probe = Prng::new(seed);
            probe.try_skip(0).is_ok()
        };
        // Never spawn more threads than there are samples: otherwise a thread's chunk
        // is empty, buffer_size(0, …) is 0, and alloc_mmap(0) aborts with InvalidSize.
        let num_cpus = num_cpus.min(scan_total).max(1);
        Self {
            num_cpus,
            scan_total,
            base_chunk: scan_total / num_cpus,
            rem: scan_total % num_cpus,
            t,
            skip_capable,
            seed,
            prescan_start: Prng::new(seed),
        }
    }

    pub(crate) fn start_sample(&self, i: usize) -> usize {
        i * self.base_chunk + i.min(self.rem)
    }

    pub(crate) fn split_desc(&self) -> &'static str {
        if self.skip_capable {
            "jump-ahead"
        } else {
            "pre-scan"
        }
    }

    /// Per-thread orbit-start snapshots: thread `i` starts at sample `base +
    /// boundaries[i]` of the orbit, reached over a `scan_len`-sample window. Skip-capable
    /// generators jump to the absolute offset; others chain a pre-scan (advancing
    /// `prescan_start`). When `prescan_label` is `Some`, the pre-scan is timed and
    /// announced (the standard per-rep path); the checkpoint path passes `None`.
    pub(crate) fn snapshots(
        &mut self,
        base: usize,
        scan_len: usize,
        boundaries: &[usize],
        prescan_label: Option<&str>,
    ) -> Box<[Prng]> {
        if self.skip_capable {
            boundaries
                .iter()
                .map(|&b| {
                    let off = ((base + b) as u64)
                        .checked_mul(self.t as u64)
                        .expect("orbit offset overflows u64");
                    let mut p = Prng::new(self.seed);
                    p.try_skip(off).unwrap();
                    p
                })
                .collect()
        } else {
            let mut sw = prescan_label.map(|lbl| {
                eprint!("{lbl}");
                Stopwatch::new()
            });
            let (snaps, end) =
                prescan_checkpoints(self.prescan_start, self.t, scan_len, boundaries);
            self.prescan_start = end;
            if let Some(sw) = sw.as_mut() {
                eprintln!("[{:.3}s]", sw.lap());
            }
            snaps
        }
    }

    /// Snapshots for one full-scan repetition (the standard per-rep path): each thread
    /// owns its contiguous chunk, and the pre-scan (if any) is announced as `Pre-scan`.
    pub(crate) fn rep_snapshots(&mut self, rep: usize) -> Box<[Prng]> {
        let boundaries: Box<[usize]> = (0..self.num_cpus).map(|i| self.start_sample(i)).collect();
        self.snapshots(
            (rep - 1) * self.scan_total,
            self.scan_total,
            &boundaries,
            Some("Pre-scan..."),
        )
    }
}

/// Sequentially walk `total_cells` samples of the orbit starting from `start`
/// (each sample consumes exactly *t* PRNG draws, in every mode; decimation still
/// draws *t* per sample, it just conditionally rejects the result), capturing a
/// copy of the generator when the walk reaches each cell index in `boundaries`
/// (which must be sorted ascending and lie in `0..total_cells`).
///
/// Returns one snapshot per boundary plus the end state (after all
/// `total_cells` cells), so the caller can chain the next repetition's walk.
/// Snapshot `i` is bit-identical to `try_skip(boundaries[i] * t)` for
/// jump-capable generators.
///
/// Used to give non-jumpable generators a faithful parallel split: the walk is
/// inherently sequential (it follows the orbit) and cannot be parallelized
/// without jump-ahead, which is exactly what these generators lack.
pub(crate) fn prescan_checkpoints(
    start: Prng,
    t: usize,
    total_cells: usize,
    boundaries: &[usize],
) -> (Box<[Prng]>, Prng) {
    let mut p = start;
    let mut snaps = Vec::with_capacity(boundaries.len());
    let mut next = 0usize;
    for cell in 0..total_cells {
        while next < boundaries.len() && boundaries[next] == cell {
            snaps.push(p);
            next += 1;
        }
        for _ in 0..t {
            p.next_u64();
        }
    }
    debug_assert_eq!(
        next,
        boundaries.len(),
        "every boundary must be < total_cells"
    );
    (snaps.into_boxed_slice(), p)
}

/// Per-thread generate (no sort) of one pass for the parallel collision test.
/// Returns the number of valid elements written to `buf`, together with the
/// orbit position reached after consuming `stream_len` draws (so the caller can
/// continue this thread's orbit in the next repetition). Sorting is a separate,
/// independently-timed phase in the caller.
///
/// `snapshot` is the thread's orbit start; in tradeoff mode it is replayed for
/// every pass, so callers pass the same snapshot each time and vary `pass`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gen_pass_dispatch<T: Cell>(
    snapshot: Prng,
    params: &GridParams,
    buf: &mut [T],
    stream_len: usize,
    pass: u64,
    tradeoff_b: usize,
    decimating: bool,
    full: bool,
) -> (usize, Prng) {
    macro_rules! go {
        ($dim:literal) => {{
            if tradeoff_b > 0 {
                // FULL (u == 64 && s == 0) short-circuits the shift+mask in the
                // per-sample draw, exactly as in the plain path below.
                match (decimating, full) {
                    (true, true) => gen_pass_tradeoff::<T, $dim, true, true>(
                        snapshot, params, buf, stream_len, pass, tradeoff_b,
                    ),
                    (true, false) => gen_pass_tradeoff::<T, $dim, true, false>(
                        snapshot, params, buf, stream_len, pass, tradeoff_b,
                    ),
                    (false, true) => gen_pass_tradeoff::<T, $dim, false, true>(
                        snapshot, params, buf, stream_len, pass, tradeoff_b,
                    ),
                    (false, false) => gen_pass_tradeoff::<T, $dim, false, false>(
                        snapshot, params, buf, stream_len, pass, tradeoff_b,
                    ),
                }
            } else {
                // No tradeoff: a single pass that consumes stream_len fresh draws.
                // gen_plain advances prng in place, so it ends at the
                // thread's next orbit position.
                let mut prng = snapshot;
                let used = if decimating {
                    if full {
                        gen_plain::<T, $dim, true, true>(&mut prng, params, buf, stream_len)
                    } else {
                        gen_plain::<T, $dim, true, false>(&mut prng, params, buf, stream_len)
                    }
                } else if full {
                    gen_plain::<T, $dim, false, true>(&mut prng, params, buf, stream_len)
                } else {
                    gen_plain::<T, $dim, false, false>(&mut prng, params, buf, stream_len)
                };
                (used, prng)
            }
        }};
    }
    match params.t {
        1 => go!(1),
        2 => go!(2),
        3 => go!(3),
        4 => go!(4),
        5 => go!(5),
        6 => go!(6),
        7 => go!(7),
        8 => go!(8),
        _ => go!(0),
    }
}

/// Faithfully generate one work-unit's points into a single contiguous buffer,
/// by count-then-fill:
///
/// 1. every thread replays its orbit segment with a store-free scan
///    ([`scan_count`]) to learn its *exact* kept count (skipped in plain mode,
///    where every draw is kept);
/// 2. `buf` is split into exactly-sized disjoint regions at the counts' prefix
///    sums, and every thread re-replays its segment writing its kept points
///    directly into its region ([`gen_pass_dispatch`]).
///
/// The counting scan re-draws the whole stream, but it is store-free and fully
/// parallel — far cheaper than the alternative, closing the inter-region gaps
/// afterwards with a single-threaded memmove over the whole buffer (measured
/// ~10x the fill time at 32e9 points). The kept points end up contiguous in
/// `buf[..total]` with no compaction and no per-thread headroom.
///
/// Returns `total` (= Σ per-thread kept counts); aborts via [`bin_overflow`]
/// if that exceeds `buf.len()`.
pub(crate) fn gen_unit_contiguous<T: Cell>(
    buf: &mut [T],
    snapshots: &[Prng],
    params: &GridParams,
    chunk: impl Fn(usize) -> usize + Sync,
    pass: u64,
    tradeoff_b: usize,
    decimating: bool,
    full: bool,
) -> usize {
    let num_cpus = snapshots.len();
    let verbose = std::env::var_os("GEN_VERBOSE").is_some();
    let sw = std::time::Instant::now();

    // Phase 1: exact per-thread kept counts. In plain mode (no decimation, no
    // tradeoff) every draw is kept, so the counts are the stream lengths.
    let counts: Vec<usize> = if !decimating && tradeoff_b == 0 {
        (0..num_cpus).map(&chunk).collect()
    } else {
        (0..num_cpus)
            .into_par_iter()
            .map(|i| {
                gen_count_dispatch::<T>(
                    snapshots[i],
                    params,
                    chunk(i),
                    pass,
                    tradeoff_b,
                    decimating,
                    full,
                )
            })
            .collect()
    };
    let total: usize = counts.iter().sum();
    if total > buf.len() {
        bin_overflow("a parallel generation unit");
    }
    if verbose {
        eprint!("{{count: {:.3}s}} ", sw.elapsed().as_secs_f64());
    }
    let sw = std::time::Instant::now();

    // Phase 2: split buf into the exactly-sized disjoint regions and let the
    // Rayon global pool fill each one from its segment's orbit snapshot. One
    // task per region, so with the standard sizing (num_cpus == pool threads)
    // each Rayon worker owns exactly one segment.
    let regions: Vec<&mut [T]> = {
        let mut regions = Vec::with_capacity(num_cpus);
        let mut rest = &mut buf[..total];
        for &count in &counts {
            let (region, tail) = rest.split_at_mut(count);
            regions.push(region);
            rest = tail;
        }
        regions
    };
    regions
        .into_par_iter()
        .enumerate()
        .for_each(|(i, region)| {
            let (used, _) = gen_pass_dispatch::<T>(
                snapshots[i],
                params,
                region,
                chunk(i),
                pass,
                tradeoff_b,
                decimating,
                full,
            );
            debug_assert_eq!(used, region.len());
        });
    if verbose {
        eprint!("{{fill: {:.3}s}} ", sw.elapsed().as_secs_f64());
    }
    total
}

/// Generate (no sort) one plain pass: `points` fresh draws into `buf`. Returns
/// the number written (`points`); `prng` is advanced in place to the thread's
/// next orbit position. Sorting is done as a separate phase by the caller so it
/// can be timed independently (mirroring the sequential `gen/sort/count` split).
fn gen_plain<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    scan_len: usize,
) -> usize {
    if DECIMATE {
        // Fixed-sample: scan scan_len candidate tuples, keep the accepted ones.
        scan_keep::<T, DIM, true, FULL, false>(
            prng,
            params,
            buf,
            scan_len,
            0,
            0,
            "a parallel decimation chunk",
        )
    } else {
        for x in buf[..scan_len].iter_mut() {
            *x = params.draw::<T, DIM, FULL>(prng);
        }
        scan_len
    }
}

/// Tradeoff, one pass (generate only, no sort): replay `snapshot` for
/// `stream_len` draws and keep only the points whose packed tradeoff key equals
/// `pass`. Returns the number kept (~`stream_len` / 2*ᵇ*) and the orbit
/// position reached. The caller invokes this once per pass reusing the same
/// `buf` and `snapshot`, so only one pass is ever resident, and sorts as a
/// separate phase so it can be timed independently.
///
/// The key extraction mirrors [`run_collision_tradeoff`]: the pass is selected by
/// the top *b* bits of the combined index, so pass *k* is one contiguous interval
/// of the value space.
///
/// [`run_collision_tradeoff`]: crate::collision::run_collision_tradeoff
fn gen_pass_tradeoff<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
    snapshot: Prng,
    params: &GridParams,
    buf: &mut [T],
    stream_len: usize,
    pass: u64,
    b: usize,
) -> (usize, Prng) {
    let mut local = snapshot;
    let len = scan_keep::<T, DIM, DECIMATE, FULL, true>(
        &mut local,
        params,
        buf,
        stream_len,
        pass,
        b,
        "a tradeoff bin",
    );
    // local has advanced by exactly stream_len draws regardless of pass
    // (every pass replays the same snapshot), so it is the orbit position from
    // which the next repetition should continue.
    (len, local)
}

/// Poisson mean of a test that examined `points` points on `cells` cells:
/// [`expected_collisions`] for the collision test, points³/(4 · cells) for
/// birthday spacings. Used both a priori (nominal point count, for the header
/// line and the default sizing) and per repetition, conditioned on the points
/// actually kept, identical to the a-priori value except under decimation,
/// where the kept count is random and conditioning avoids overdispersing the
/// null distribution.
pub fn test_lambda(points: usize, cells: f64, birthday_spacings: bool) -> f64 {
    if birthday_spacings {
        // TestU01 long guide: lambda = n³ / (4k).
        BigUint::from(points).pow(3).to_f64().unwrap() / (cells * 4.0)
    } else {
        expected_collisions(points as f64, cells)
    }
}

/// Computes the Poisson mean and the point count to use, applying the test-specific defaults.
pub fn compute_lambda_and_points(args: &Args, cells: &BigUint) -> (f64, usize) {
    // cells is already the effective cell count: main() computes it as
    // (2ᵘ⁻ᵈ)ᵗ, incorporating any whole-tuple decimation.
    let effective_cells_f64 = cells.to_f64().unwrap();

    // In tradeoff mode the user-supplied m is the per-pass memory; the actual
    // number of points is m · 2ᵇ, where b is the total number of top tradeoff
    // bits of the combined index (the partition has 2ᵇ contiguous value intervals).
    // Decimation and the other modes use m as-is. Use a checked shift so an
    // out-of-range product is reported.
    let pass_factor = match args.tradeoff_bits {
        Some(b) => 1usize.checked_shl(b as u32).expect("2ᵇ overflows usize"),
        None => 1,
    };

    let points;
    let lambda = if args.birthday_spacings {
        // TestU01 long guide p. 133: choose points to maximise birthday-spacings power.
        let max_points = (effective_cells_f64.powf(5.0 / 12.0)
            / (2.0 * args.reps as f64).powf(1.0 / 3.0)) as usize;
        // With a tradeoff, m is the per-class/per-interval memory and the total
        // point count is m · 2ᵇ (only ~points / 2ᵇ are ever resident), mirroring
        // the collision tradeoff; without one, pass_factor is 1 and points = m.
        let m = args.m.unwrap_or(max_points / pass_factor.max(1));
        points = m.checked_mul(pass_factor).expect("m · 2ᵇ overflows usize");
        if points > max_points {
            Args::die(
                "the given combination of memory, repetitions and cells is out of range \
                 (omit -m to use the maximum)",
            );
        }
        test_lambda(points, effective_cells_f64, true)
    } else {
        let m_default = (cells.clone() / pass_factor)
            .min(usize::MAX.into())
            .to_usize()
            .unwrap();
        let m = args.m.unwrap_or(m_default);
        points = m.checked_mul(pass_factor).expect("m · 2ᵇ overflows usize");

        if BigUint::from(points) > *cells {
            Args::die(&format!(
                "more points ({}) than {}cells ({})",
                points,
                if args.decimation_bits.is_some() {
                    "effective "
                } else {
                    ""
                },
                cells
            ));
        }
        test_lambda(points, effective_cells_f64, false)
    };

    if points < 10000 {
        Args::die(&format!(
            "the number of points ({points}) is smaller than 10000"
        ));
    }

    (lambda, points)
}

/// Runs the test sequentially, dispatching the hot loop's const generics once per
/// test.
///
/// The cell type `T` is chosen by the caller (`dispatch` in `main`). Here we pick:
/// - `DIM` (the dimension *t*): *t* = 1..=8 is monomorphized so the draw loop
///   unrolls; larger *t* uses the `DIM = 0` runtime fallback.
/// - `DECIMATE` (*d* > 0) and `FULL` (*u* = 64 and *s* = 0): one-time
///   branches selecting the specialized [`cell_index`] instantiation.
///
/// Returns the total collision count and the summed per-repetition Poisson
/// means, each conditioned on the points the repetition actually kept
/// (identical to `lambda * reps` when not decimating).
///
/// [`cell_index`]: crate::cell::cell_index
pub fn run_test<T: Cell>(args: &Args, points: usize, cells: &BigUint, lambda: f64) -> (u128, f64) {
    let seed = args.seed;
    eprintln!("Seed: {:#018x}", seed);

    let mut prng = Prng::new(seed);

    let d = args.decimation_bits.unwrap_or(0);
    let tradeoff_b = args.tradeoff_bits(); // tradeoff bits b (0 when absent)
    // Fixed-sample model: a pass scans scan_len = points · 2ᵗᵈ samples and
    // keeps the accepted (decimation) and key-matching (tradeoff) subset. Buffer
    // headroom is the balls-into-bins bound over the t·d + b selectivity bits.
    let partition_bits = args.t * d + tradeoff_b;
    let scan_len = scan_samples(points, args.t, d);
    // The birthday tradeoff accumulates one spacing-class (~points / 2ᵇ spacings)
    // in this buffer and keeps each value-interval points in an internal scratch;
    // every other mode fills this buffer directly from the sample scan.
    let buf_len = if args.birthday_spacings && tradeoff_b > 0 {
        buffer_size(points, tradeoff_b)
    } else {
        buffer_size(scan_len, partition_bits)
    };

    let output_type = bits_read_desc(args.s);
    let test_type = if args.birthday_spacings {
        "birthday-spacings"
    } else {
        "collision"
    };

    let headroom_suffix = if partition_bits > 0 {
        // `partition_bits` = t·d + b can exceed 63 via the decimation term even
        // though b < 64, so compute 2^partition_bits in floating point to avoid a
        // shift overflow (this is a cosmetic header figure only).
        let mean = (scan_len as f64) / 2.0f64.powi(partition_bits as i32);
        format!(" (+{:.2}%)", (buf_len as f64 / mean - 1.0) * 100.0)
    } else {
        String::new()
    };

    let mut mode_parts: Vec<String> = Vec::new();
    if tradeoff_b > 0 {
        mode_parts.push(format!(
            "tradeoff on {} top bits over {} passes",
            tradeoff_b,
            1u64 << tradeoff_b
        ));
    }
    if d > 0 {
        mode_parts.push(decimation_desc(d, args.t));
    }
    let mode_suffix = join_mode_parts(&mode_parts);

    eprintln!(
        "Running a {}-dimensional {} test on the upper {} bits of the {} ({} points, {}-bit cells, {} memory locations, {:.3} GiB RAM{}{})",
        args.t,
        test_type,
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
        points >> tradeoff_b,
        (buf_len * size_of::<T>()) as f64 / 2.0f64.powi(30),
        headroom_suffix,
        mode_suffix
    );

    let cells_suffix = effective_cells_suffix(d, args.u, args.t);
    eprintln!(
        "u: {} t: {} cells: {:.0} expected collisions: {}{}",
        args.u, args.t, cells, lambda, cells_suffix
    );

    let mut mapped = alloc_mmap::<T>(buf_len);
    let buf: &mut [T] = bytemuck::try_cast_slice_mut(&mut mapped).unwrap();

    // Sort scratch for the runners that use the out-of-place sort_mt, allocated
    // once (huge pages, prefaulted) and reused across repetitions:
    // mapping/unmapping it per sort would cost far more than the sort itself.
    // The birthday tradeoff runner sorts in place (sort_st) precisely to avoid
    // doubling RSS, so it gets no scratch.
    let needs_mt_scratch = !(args.birthday_spacings && tradeoff_b > 0);
    let mut scratch_mapped = needs_mt_scratch.then(|| alloc_mmap::<T>(buf_len));
    let scratch: &mut [T] = match scratch_mapped.as_mut() {
        Some(m) => bytemuck::try_cast_slice_mut(m).unwrap(),
        None => &mut [],
    };

    let params = GridParams {
        u: args.u,
        t: args.t,
        s: args.s,
        d,
        cells,
    };

    let full = args.u == 64 && args.s == 0;
    let decimating = d > 0;

    // cells already incorporates decimation (see main()), so no further shift.
    let effective_cells_f64 = cells.to_f64().unwrap();

    let mut sw = Stopwatch::new();
    let mut tot: u128 = 0;
    let mut lambda_sum = 0.0f64;
    for _rep in 1..=args.reps {
        // go!(DIM) expands the FULL/DECIMATE/mode matrix for one DIM literal;
        // the outer match picks DIM (0 = runtime fallback).
        macro_rules! go {
            ($dim:literal) => {{
                if args.birthday_spacings {
                    if tradeoff_b > 0 {
                        match (decimating, full) {
                            (false, false) => run_birthday_tradeoff::<T, $dim, false, false>(
                                &mut prng,
                                &params,
                                buf,
                                points,
                                tradeoff_b,
                                args.pretty_p,
                                args.pass,
                            ),
                            (true, false) => run_birthday_tradeoff::<T, $dim, true, false>(
                                &mut prng,
                                &params,
                                buf,
                                points,
                                tradeoff_b,
                                args.pretty_p,
                                args.pass,
                            ),
                            (false, true) => run_birthday_tradeoff::<T, $dim, false, true>(
                                &mut prng,
                                &params,
                                buf,
                                points,
                                tradeoff_b,
                                args.pretty_p,
                                args.pass,
                            ),
                            (true, true) => run_birthday_tradeoff::<T, $dim, true, true>(
                                &mut prng,
                                &params,
                                buf,
                                points,
                                tradeoff_b,
                                args.pretty_p,
                                args.pass,
                            ),
                        }
                    } else {
                        match (decimating, full) {
                            (false, false) => run_birthday::<T, $dim, false, false>(
                                &mut prng, &params, buf, scratch, points,
                            ),
                            (true, false) => run_birthday::<T, $dim, true, false>(
                                &mut prng, &params, buf, scratch, points,
                            ),
                            (false, true) => run_birthday::<T, $dim, false, true>(
                                &mut prng, &params, buf, scratch, points,
                            ),
                            (true, true) => run_birthday::<T, $dim, true, true>(
                                &mut prng, &params, buf, scratch, points,
                            ),
                        }
                    }
                } else if tradeoff_b > 0 {
                    let cells_per_pass = effective_cells_f64 / (1u64 << tradeoff_b) as f64;
                    if decimating {
                        run_collision_tradeoff::<T, $dim, true>(
                            &mut prng,
                            &params,
                            buf,
                            scratch,
                            points,
                            tradeoff_b,
                            cells_per_pass,
                            args.pretty_p,
                            args.pass,
                        )
                    } else {
                        run_collision_tradeoff::<T, $dim, false>(
                            &mut prng,
                            &params,
                            buf,
                            scratch,
                            points,
                            tradeoff_b,
                            cells_per_pass,
                            args.pretty_p,
                            args.pass,
                        )
                    }
                } else if decimating {
                    if full {
                        run_collision_decimate::<T, $dim, true>(
                            &mut prng,
                            &params,
                            buf,
                            scratch,
                            points,
                            effective_cells_f64,
                            args.checkpoints,
                            args.pretty_p,
                        )
                    } else {
                        run_collision_decimate::<T, $dim, false>(
                            &mut prng,
                            &params,
                            buf,
                            scratch,
                            points,
                            effective_cells_f64,
                            args.checkpoints,
                            args.pretty_p,
                        )
                    }
                } else if full {
                    run_collision::<T, $dim, true>(&mut prng, &params, buf, scratch)
                } else {
                    run_collision::<T, $dim, false>(&mut prng, &params, buf, scratch)
                }
            }};
        }

        let (c, used) = match args.t {
            1 => go!(1),
            2 => go!(2),
            3 => go!(3),
            4 => go!(4),
            5 => go!(5),
            6 => go!(6),
            7 => go!(7),
            8 => go!(8),
            _ => go!(0),
        };

        tot += c as u128;
        // Condition the Poisson mean on the points actually examined: identical
        // to the a-priori lambda except under decimation, where the kept count
        // is random and conditioning avoids overdispersion.
        let lambda_rep = test_lambda(used, effective_cells_f64, args.birthday_spacings);
        lambda_sum += lambda_rep;
        let rep_p = format_p_value(p_value(c as f64, lambda_rep), args.pretty_p);
        if args.reps > 1 {
            eprintln!(
                "{c}\tp={rep_p}\tcombined: {tot}\tp={}",
                format_p_value(p_value(tot as f64, lambda_sum), args.pretty_p)
            );
        } else {
            eprintln!("{c}\tp={rep_p}");
        }
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, lambda_sum)
}

#[cfg(all(test, feature = "incr"))]
mod prescan_tests {
    use super::*;

    // incr has analytic state: next_u64 does x += 1 then returns x, so a
    // generator seeded at seed reaches state x = seed + n after n steps and its
    // next output is seed + n + 1. That lets us verify prescan_checkpoints landed
    // each snapshot exactly where a jump-by-(boundary*t) would, with a
    // hand-computed ground truth (no try_skip in the assertions).
    #[test]
    fn prescan_lands_at_jump_targets() {
        let seed = 0x1234_5678_9abc_def0u64;
        let t = 3usize;
        let total = 1000usize;
        let boundaries = [0usize, 137, 500, 999];
        let (snaps, end) = prescan_checkpoints(Prng::new(seed), t, total, &boundaries);
        assert_eq!(snaps.len(), boundaries.len());
        for (k, &b) in boundaries.iter().enumerate() {
            let mut s = snaps[k];
            let expected = seed.wrapping_add((b * t) as u64).wrapping_add(1);
            assert_eq!(
                s.next_u64(),
                expected,
                "snapshot {k} at boundary {b} landed wrong"
            );
        }
        let mut e = end;
        let expected_end = seed.wrapping_add((total * t) as u64).wrapping_add(1);
        assert_eq!(e.next_u64(), expected_end, "chained end state landed wrong");
    }
}

