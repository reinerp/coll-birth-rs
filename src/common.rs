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
use crate::cell::{Cell, cell_index, decimate_once};
use crate::cli::Args;
use crate::collision::{run_collision, run_collision_decimate, run_collision_tradeoff};
use crate::prng::Prng;
use crate::stats::{expected_collisions, format_p_value, p_value};
use crate::util::{Stopwatch, parallelism};

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
         for a uniform generator — overwhelming evidence that the generator under \
         test is grossly non-uniform. Rerun in plain mode (no -b/-d) for an exact \
         p-value."
    );
    std::process::exit(1);
}

/// Samples scanned per pass: `points` · 2*ᵗᵈ*. Uses an overflow check:
/// [`checked_shl`] alone only guards the shift amount, not the resulting value, so
/// a too-large t·d would silently wrap. A configuration whose sample budget
/// does not fit in a usize (points · 2*ᵗᵈ* ≥ 2⁶⁴; points already carries the
/// 2*ᵇ* tradeoff factor) surfaces here as a clean overflow error.
///
/// [`checked_shl`]: u32::checked_shl
pub(crate) fn scan_samples(points: usize, t: usize, d: usize) -> usize {
    1usize
        .checked_shl((t * d) as u32)
        .and_then(|factor| points.checked_mul(factor))
        .expect("points · 2^(t·d) overflows usize")
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
        "decimating {} bits per dimension (~2^{} candidate samples per kept sample)",
        d,
        d * t
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

/// The " (effective cells after decimation: 2^N)" header note (empty when d == 0).
pub(crate) fn effective_cells_suffix(d: usize, u: usize, t: usize) -> String {
    if d > 0 {
        format!(" (effective cells after decimation: 2^{})", (u - d) * t)
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

/// Close the gaps left by under-filled per-thread sub-regions, in place.
///
/// `buf` is partitioned into `caps[i]`-sized sub-regions (sub-region `i` begins at
/// the prefix sum of `caps[..i]`); thread `i` wrote `used[i] ≤ caps[i]` kept points
/// into the front of its sub-region. This shifts each block left to close the gaps,
/// leaving the kept points contiguous in `buf[..total]`, and returns `total = Σ used`.
///
/// Every destination is ≤ its source (since `Σ used ≤ Σ caps`), so the moves are
/// leftward; performed left to right, block `i`'s destination ends at the start of
/// block `i+1`'s still-untouched source and block `i−1` has already vacated, so no
/// move clobbers an unread source. `copy_within` covers the intra-block overlap. In
/// plain mode `used[i] == caps[i]`, every destination equals its source, and this is
/// a no-op.
fn compact_blocks<T: Cell>(buf: &mut [T], caps: &[usize], used: &[usize]) -> usize {
    debug_assert_eq!(caps.len(), used.len());
    let mut src_base = 0usize;
    let mut dst = 0usize;
    for (&cap, &len) in caps.iter().zip(used) {
        debug_assert!(len <= cap);
        if dst != src_base {
            buf.copy_within(src_base..src_base + len, dst);
        }
        dst += len;
        src_base += cap;
    }
    dst
}

/// Faithfully generate one work-unit's points into a single contiguous buffer.
///
/// Threads fill disjoint `caps`-sized sub-regions of `buf` from their orbit
/// `snapshots` (via [`gen_pass_dispatch`]); the gaps left by under-filled
/// sub-regions are then closed in place by [`compact_blocks`]. Returns `total_used`;
/// on return `buf[..total_used]` holds the kept points (unsorted) so that a single
/// [`Cell::sort_mt`] + linear [`count_adjacent_equals`] serves both tests — replacing
/// the per-thread buffers + k-way merge the parallel collision runner used before.
///
/// `buf.len()` must be at least `Σ caps`; each `caps[i]` is the headroom-sized
/// capacity of thread `i`'s sub-region and `chunk(i)` its sample-stream length.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gen_unit_contiguous<T: Cell>(
    buf: &mut [T],
    caps: &[usize],
    snapshots: &[Prng],
    params: &GridParams,
    chunk: impl Fn(usize) -> usize,
    pass: u64,
    tradeoff_b: usize,
    decimating: bool,
    full: bool,
) -> usize {
    let num_cpus = caps.len();
    debug_assert_eq!(snapshots.len(), num_cpus);

    // Phase 1: each thread generates into its own disjoint sub-region.
    let used: Box<[usize]> = std::thread::scope(|scope| {
        let mut rest = &mut buf[..];
        let mut handles = Vec::with_capacity(num_cpus);
        for (i, &cap) in caps.iter().enumerate() {
            let (region, tail) = rest.split_at_mut(cap);
            rest = tail;
            let snap = snapshots[i];
            let stream_len = chunk(i);
            handles.push(scope.spawn(move || {
                gen_pass_dispatch::<T>(
                    snap, params, region, stream_len, pass, tradeoff_b, decimating, full,
                )
                .0
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Phase 2: close the gaps so the kept points are contiguous.
    compact_blocks(buf, caps, &used)
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
        let mut len = 0usize;
        for _ in 0..scan_len {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *buf.get_mut(len)
                    .unwrap_or_else(|| bin_overflow("a parallel decimation chunk")) = x;
                len += 1;
            }
        }
        len
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
    let t = params.t;
    let u = params.u;
    let d = params.d;
    let elem_width = if DECIMATE { u - d } else { u };
    // Pass k holds the points whose top b bits of the combined index equal k
    // (one contiguous value interval); see run_collision_tradeoff.
    let key_shift = t * elem_width - b;

    let key_of = |x: T| -> T {
        let mut key = x;
        key >>= key_shift;
        key
    };

    let mut local = snapshot;
    let target = T::from_u64(pass);
    let mut len = 0usize;
    for _ in 0..stream_len {
        let x = if DECIMATE {
            match params.draw_decimate_once::<T, DIM, FULL>(&mut local) {
                Some(x) => x,
                None => continue,
            }
        } else {
            params.draw::<T, DIM, FULL>(&mut local)
        };
        if key_of(x) == target {
            *buf.get_mut(len)
                .unwrap_or_else(|| bin_overflow("a tradeoff bin")) = x;
            len += 1;
        }
    }
    // local has advanced by exactly stream_len draws regardless of pass
    // (every pass replays the same snapshot), so it is the orbit position from
    // which the next repetition should continue.
    (len, local)
}

/// Poisson mean of a test that examined `points` points on `cells` cells:
/// [`expected_collisions`] for the collision test, points³/(4 · cells) for
/// birthday spacings. Used both a priori (nominal point count, for the header
/// line and the default sizing) and per repetition, conditioned on the points
/// actually kept — identical to the a-priori value except under decimation,
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
    let pass_factor = match args.tradeoff {
        Some(b) => 1usize.checked_shl(b as u32).expect("2^b overflows usize"),
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
        points = m.checked_mul(pass_factor).expect("m · 2^b overflows usize");
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
                if args.decimate.is_some() {
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

    let d = args.decimate.unwrap_or(0);
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
        "Running a {} test on the upper {} bits of the {} ({} points, {}-bit cells, {} memory locations, {:.3} GiB RAM{}{})",
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
                                &mut prng, &params, buf, points,
                            ),
                            (true, false) => run_birthday::<T, $dim, true, false>(
                                &mut prng, &params, buf, points,
                            ),
                            (false, true) => run_birthday::<T, $dim, false, true>(
                                &mut prng, &params, buf, points,
                            ),
                            (true, true) => {
                                run_birthday::<T, $dim, true, true>(&mut prng, &params, buf, points)
                            }
                        }
                    }
                } else if tradeoff_b > 0 {
                    let cells_per_pass = effective_cells_f64 / (1u64 << tradeoff_b) as f64;
                    if decimating {
                        run_collision_tradeoff::<T, $dim, true>(
                            &mut prng,
                            &params,
                            buf,
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
                            points,
                            effective_cells_f64,
                            args.checkpoints,
                            args.pretty_p,
                        )
                    }
                } else if full {
                    run_collision::<T, $dim, true>(&mut prng, &params, buf)
                } else {
                    run_collision::<T, $dim, false>(&mut prng, &params, buf)
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

// Pure compaction (no PRNG): every block's destination is the prefix sum of the
// used counts, so the result is exactly the concatenation of the kept prefixes.
#[cfg(test)]
mod compact_tests {
    use super::*;

    /// Lay sentinel-padded blocks into a buffer of capacity `Σ caps`, run
    /// `compact_blocks`, and check the kept prefixes come out concatenated.
    fn check(caps: &[usize], used: &[usize]) {
        let cap_total: usize = caps.iter().sum();
        // Each kept slot of block i carries a unique value (i*1000 + j); padding
        // and trailing gap carry a distinguishable sentinel.
        let mut buf = vec![u64::MAX; cap_total];
        let mut base = 0usize;
        let mut expected: Vec<u64> = Vec::new();
        for (i, (&cap, &len)) in caps.iter().zip(used).enumerate() {
            for j in 0..len {
                let v = (i as u64) * 1000 + j as u64;
                buf[base + j] = v;
                expected.push(v);
            }
            base += cap;
        }
        let total = compact_blocks::<u64>(&mut buf, caps, used);
        assert_eq!(
            total,
            used.iter().sum::<usize>(),
            "total for {caps:?}/{used:?}"
        );
        assert_eq!(
            &buf[..total],
            &expected[..],
            "compacted for {caps:?}/{used:?}"
        );
    }

    #[test]
    fn compaction_cases() {
        check(&[5, 5, 5], &[5, 5, 5]); // all full: plain no-op
        check(&[5, 5, 5], &[3, 4, 2]); // generic gaps
        check(&[5, 5, 5], &[0, 4, 2]); // first empty
        check(&[5, 5, 5], &[3, 0, 2]); // middle empty
        check(&[5, 5, 5], &[3, 4, 0]); // last empty
        check(&[5, 5, 5], &[0, 0, 0]); // all empty
        check(&[10, 1, 7], &[1, 1, 7]); // big leftward shift, full last block
        check(&[4], &[2]); // single block
        check(&[0, 5, 0, 3], &[0, 5, 0, 3]); // zero-capacity sub-regions
    }
}
