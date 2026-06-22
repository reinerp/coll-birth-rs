/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

mod cdf;
mod cell;
mod cli;
mod prng;
mod stats;
mod test;
mod util;

use std::cmp::Reverse;
use std::mem::size_of;
use std::time::SystemTime;

use clap::Parser;
use dary_heap::QuaternaryHeap;
use mmap_rs::{MmapFlags, MmapMut, MmapOptions};
use num::BigUint;
use num::traits::ToPrimitive;
use rayon::prelude::*;

use cell::Cell;
use cli::Args;
use prng::Prng;
use stats::{expected_collisions, format_p_value, p_value};
use test::{
    GridParams, count_adjacent_equals, run_birthday, run_birthday_tradeoff, run_collision,
    run_collision_decimate, run_collision_tradeoff,
};
use util::Stopwatch;
use util::parallelism;

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
fn alloc_mmap<T>(n: usize) -> MmapMut {
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

/// Returns nanoseconds since the Unix epoch, used as the default PRNG seed.
fn current_nanos_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Buffer size needed for the active sampling mode.
///
/// A pass keeps the samples whose selection key — the *t* · *d* decimation
/// residue bits plus the *b* top tradeoff bits, `partition_bits` in all —
/// matches a fixed value, that is, one "bin" of a balls-into-bins experiment
/// with `points` balls and *n* = 2^`partition_bits` bins. A single bin's load
/// is a sum of independent indicators, so writing *m* for its mean
/// `points`/*n*, Bernstein's inequality bounds P[load ≥ *m* + λ] by
/// exp(−λ²/(2(*m* + λ/3))); a union bound over the *n* bins then makes the
/// probability that *any* bin exceeds *m* + λ at most
/// *n* · exp(−λ²/(2(*m* + λ/3))), which is below 10⁻¹⁰⁰⁰ for
/// λ = *L*/3 + √(*L*²/9 + 2·*m*·*L*) with *L* = ln *n* + 1000 · ln 10 (the
/// exact inversion of the exponent). The headroom's shape is tight: by
/// Theorem 1 of [Raab & Steger] the maximum load actually reaches
/// *m* + √(2·*m*·ln *n*) · (1 − o(1)) in the heavily-loaded regime, so little
/// can be shaved.
///
/// [Raab & Steger]: https://doi.org/10.1007/3-540-49543-6_13
pub(crate) fn buffer_size(points: usize, partition_bits: usize) -> usize {
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

/// Samples scanned per pass: points · 2^(t·d). Uses a genuine overflow check —
/// checked_shl alone only guards the shift amount, not the resulting value, so a
/// too-large t·d would silently wrap. A configuration whose sample budget does
/// not fit in a usize (points · 2^(t·d) ≥ 2^64; points already carries the 2^b
/// tradeoff factor) surfaces here as a clean overflow error.
pub(crate) fn scan_samples(points: usize, t: usize, d: usize) -> usize {
    1usize
        .checked_shl((t * d) as u32)
        .and_then(|factor| points.checked_mul(factor))
        .expect("points · 2^(t·d) overflows usize")
}

/// Picks the smallest cell integer type that can hold every stored value.
///
/// Cell indices are below `cells`; birthday spacings additionally store the
/// wrap-around `cells - max + min`, which is evaluated through the representable
/// `cells - 1` and fits the same width (the degenerate `min == max` case, where
/// it would equal `cells`, is dropped without affecting the collision count —
/// see `compute_spacings`). Both tests therefore use `u32` when `cells` ≤ 2³²,
/// `u64` when ≤ 2⁶⁴, and `u128` otherwise.
fn dispatch<F32, F64, F128, R>(cells: &BigUint, f32: F32, f64: F64, f128: F128) -> R
where
    F32: FnOnce() -> R,
    F64: FnOnce() -> R,
    F128: FnOnce() -> R,
{
    if cells <= &BigUint::from(1u128 << 32) {
        f32()
    } else if cells <= &BigUint::from(1u128 << 64) {
        f64()
    } else {
        f128()
    }
}

/// Runs the test, dispatching the hot loop's const generics once per test.
///
/// The cell type `T` is chosen by the caller ([`dispatch`]). Here we pick:
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
fn run_test<T: Cell>(args: &Args, points: usize, cells: &BigUint, lambda: f64) -> (u128, f64) {
    let seed = args.seed.unwrap_or_else(current_nanos_seed);
    eprintln!("Seed: {:#018x}", seed);

    let mut prng = Prng::new(seed);

    let d = args.decimate.unwrap_or(0);
    let tradeoff_b = args.tradeoff_bits(); // tradeoff bits b (0 when absent)
    // Fixed-sample model: a pass scans scan_len = points · 2^(t·d) samples and
    // keeps the accepted (decimation) and key-matching (tradeoff) subset. Buffer
    // headroom is the balls-into-bins bound over the t·d + b selectivity bits.
    let partition_bits = args.t * d + tradeoff_b;
    let scan_len = scan_samples(points, args.t, d);
    // The birthday tradeoff accumulates one spacing-class (~points / 2^b spacings)
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
        let mean = (scan_len as f64) / (1u64 << partition_bits) as f64;
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
        "Running a {} test on the upper {} bits of the {} using {} points ({}-bit cells, {:.3} GiB{}{})",
        test_type,
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
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
        eprintln!(
            "{}\tp={}\tcombined: {}\tp={}",
            c,
            format_p_value(p_value(c as f64, lambda_rep), args.pretty_p),
            tot,
            format_p_value(p_value(tot as f64, lambda_sum), args.pretty_p)
        );
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, lambda_sum)
}

/// Counts adjacent-equal elements across *p* sorted slices, as if they were
/// merged into one sorted sequence: the union's collision count equals the
/// number of adjacent-equal pairs in its sorted order.
///
/// The work is parallelized by cutting the value domain into segments at *value*
/// boundaries, so every copy of a value lands in the same segment and the
/// per-segment counts sum to the exact total with no boundary correction (the
/// last element of a segment is strictly below the first element of the next).
/// Splitters are sampled from the largest slice's quantiles; for the ~uniform
/// cell values of this test that balances the segments, and correctness holds
/// for any monotone choice of splitters.
fn merge_count_collisions<T: Cell>(sorted: &[&[T]]) -> usize {
    let n_total: usize = sorted.iter().map(|s| s.len()).sum();
    // Aim for several segments per thread for load balancing.
    let num_segs = (parallelism() * 4).min(n_total.max(1));
    merge_count_collisions_segmented(sorted, num_segs)
}

/// Implementation of [`merge_count_collisions`] with an explicit segment count
/// (factored out so tests can drive the value-partitioning path deterministically
/// regardless of the host's core count).
fn merge_count_collisions_segmented<T: Cell>(sorted: &[&[T]], num_segs: usize) -> usize {
    let n_total: usize = sorted.iter().map(|s| s.len()).sum();
    let num_segs = num_segs.min(n_total);
    if num_segs <= 1 {
        return merge_count_segment(sorted);
    }

    // Splitter values from the largest slice's evenly spaced quantiles.
    let pivot = sorted.iter().copied().max_by_key(|s| s.len()).unwrap();
    let splitters: Vec<T> = (1..num_segs)
        .map(|k| pivot[(pivot.len() * k) / num_segs])
        .collect();

    // Segment seg holds values in [splitters[seg-1], splitters[seg]), with
    // open ends at the extremes; each slice's sub-range is found by binary search.
    (0..num_segs)
        .into_par_iter()
        .map(|seg| {
            let subs: Vec<&[T]> = sorted
                .iter()
                .map(|s| {
                    let start = if seg == 0 {
                        0
                    } else {
                        s.partition_point(|v| *v < splitters[seg - 1])
                    };
                    let end = if seg == num_segs - 1 {
                        s.len()
                    } else {
                        s.partition_point(|v| *v < splitters[seg])
                    };
                    &s[start..end]
                })
                .collect();
            merge_count_segment(&subs)
        })
        .sum()
}

/// Serial k-way merge of *p* sorted slices using a quaternary min-heap, counting
/// adjacent-equal pairs. Each heap entry is `(value, stream_index)`; we pop the
/// minimum, advance that stream, and check whether consecutive pops are equal.
fn merge_count_segment<T: Cell>(sorted: &[&[T]]) -> usize {
    let mut heap: QuaternaryHeap<Reverse<(T, usize)>> = QuaternaryHeap::new();
    let mut pos = vec![0usize; sorted.len()];
    for (i, s) in sorted.iter().enumerate() {
        if !s.is_empty() {
            heap.push(Reverse((s[0], i)));
            pos[i] = 1;
        }
    }
    let mut collisions = 0usize;
    let mut prev: Option<T> = None;
    while let Some(Reverse((val, i))) = heap.pop() {
        if prev == Some(val) {
            collisions += 1;
        }
        prev = Some(val);
        if pos[i] < sorted[i].len() {
            heap.push(Reverse((sorted[i][pos[i]], i)));
            pos[i] += 1;
        }
    }
    collisions
}

/// K-way merge of sorted slices into one sorted `Vec` (no counting). Used by the
/// parallel checkpoint path to fold a stage's per-thread sorted sub-buffers into a
/// single sorted run before merging it into the cumulative buffer.
fn merge_sorted<T: Cell>(sorted: &[&[T]]) -> Vec<T> {
    let total: usize = sorted.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total);
    let mut heap: QuaternaryHeap<Reverse<(T, usize)>> = QuaternaryHeap::new();
    let mut pos = vec![0usize; sorted.len()];
    for (i, s) in sorted.iter().enumerate() {
        if !s.is_empty() {
            heap.push(Reverse((s[0], i)));
            pos[i] = 1;
        }
    }
    while let Some(Reverse((val, i))) = heap.pop() {
        out.push(val);
        if pos[i] < sorted[i].len() {
            heap.push(Reverse((sorted[i][pos[i]], i)));
            pos[i] += 1;
        }
    }
    out
}

/// "full 64-bit output" or "lowest N bits": which bits of each draw the test reads.
fn bits_read_desc(s: usize) -> String {
    if s == 0 {
        "full 64-bit output".to_string()
    } else {
        format!("lowest {} bits", 64 - s)
    }
}

/// The shared decimation descriptor for the header's mode list.
fn decimation_desc(d: usize, t: usize) -> String {
    format!(
        "decimating {} bits per dimension (~2^{} candidate samples per kept sample)",
        d,
        d * t
    )
}

/// Joins the per-mode descriptors into the header's trailing ", a, b" suffix (empty
/// when there are none).
fn join_mode_parts(parts: &[String]) -> String {
    if parts.is_empty() {
        String::new()
    } else {
        format!(", {}", parts.join(", "))
    }
}

/// The " (effective cells after decimation: 2^N)" header note (empty when d == 0).
fn effective_cells_suffix(d: usize, u: usize, t: usize) -> String {
    if d > 0 {
        format!(" (effective cells after decimation: 2^{})", (u - d) * t)
    } else {
        String::new()
    }
}

/// The faithful contiguous-orbit partition shared by both parallel runners. A scan
/// of `scan_total` samples is split into `num_cpus` contiguous sample-ranges; each
/// range's start is reached by jump-ahead (`try_skip`) for skip-capable generators
/// or by a chained sequential pre-scan ([`prescan_checkpoints`]) otherwise. This is
/// the single home of the faithfulness-critical orbit arithmetic; the result is
/// bit-identical to a sequential run for any `num_cpus`.
struct OrbitPartition {
    num_cpus: usize,
    scan_total: usize,
    base_chunk: usize,
    rem: usize,
    t: usize,
    skip_capable: bool,
    seed: u64,
    /// Pre-scan path only: the orbit start of the next repetition, chained across
    /// reps (a non-jumpable generator cannot recompute an absolute offset).
    prescan_start: Prng,
}

impl OrbitPartition {
    fn new(seed: u64, num_cpus: usize, scan_total: usize, t: usize) -> Self {
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

    fn start_sample(&self, i: usize) -> usize {
        i * self.base_chunk + i.min(self.rem)
    }

    fn split_desc(&self) -> &'static str {
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
    fn snapshots(
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
    fn rep_snapshots(&mut self, rep: usize) -> Box<[Prng]> {
        let boundaries: Box<[usize]> = (0..self.num_cpus).map(|i| self.start_sample(i)).collect();
        self.snapshots(
            (rep - 1) * self.scan_total,
            self.scan_total,
            &boundaries,
            Some("Pre-scan..."),
        )
    }
}

/// Parallel version of the collision test.
///
/// The sequential orbit segment of a pass (`scan_total` samples) is split into
/// `num_cpus` contiguous sample-ranges; each thread owns one range and one
/// buffer of `~points / num_cpus` slots, reused across passes. A thread reaches
/// the start of its range by jump-ahead (`try_skip`) or, for non-jumpable
/// generators, through a sequential pre-scan ([`prescan_checkpoints`]);
/// repetitions *continue* each orbit rather than reseeding. The result is
/// bit-identical to the sequential [`run_test`] for every CPU count,
/// repetition by repetition.
/// A "pass" is a single sweep when no tradeoff is active; with `--tradeoff b`
/// there are 2*ᵇ* passes, one per top-bit value interval, exactly as in the
/// sequential [`run_collision_tradeoff`].
///
/// For each pass every thread fills and single-thread-sorts its buffer, then
/// the `num_cpus` sorted buffers are counted by a parallel value-segmented merge
/// (see [`merge_count_collisions`]) so collisions *across* threads are counted
/// without serializing on one core. The per-pass counts are summed. Because the
/// pass loop wraps the thread spawn —
/// rather than living inside each thread — the live memory is one pass' worth
/// (~`points` / 2*ᵇ*, split `num_cpus` ways) regardless of *b*, matching the
/// sequential tradeoff's space behaviour.
///
/// Returns the total collision count and the summed per-repetition Poisson
/// means, each conditioned on the points the repetition actually kept (see
/// [`run_test`]).
fn run_test_parallel<T: Cell>(
    args: &Args,
    points: usize,
    cells: &BigUint,
    lambda: f64,
    num_cpus: usize,
) -> (u128, f64) {
    let seed = args.seed.unwrap_or_else(current_nanos_seed);
    eprintln!("Seed: {:#018x}", seed);

    let d = args.decimate.unwrap_or(0);
    let tradeoff_b = args.tradeoff_bits();
    let num_passes: u64 = 1u64 << tradeoff_b; // tradeoff passes (1 when none)
    // Buffer headroom spans the t·d + b selectivity bits (decimation + tradeoff).
    let partition_bits = args.t * d + tradeoff_b;

    let output_type = bits_read_desc(args.s);

    let mut mode_parts: Vec<String> = Vec::new();
    if tradeoff_b > 0 {
        mode_parts.push(format!(
            "tradeoff on {} top bits over {} passes",
            tradeoff_b, num_passes
        ));
    }
    if d > 0 {
        mode_parts.push(decimation_desc(d, args.t));
    }
    let mode_suffix = join_mode_parts(&mode_parts);

    let decimating = d > 0;

    // Fixed-sample: each pass scans scan_total = points · 2^(t·d) samples, split into
    // num_cpus contiguous sample-ranges reached by jump-ahead or a chained pre-scan
    // (see OrbitPartition). Every mode is faithful — there is no decorrelated fallback.
    // Per-chunk buffer headroom spans the t·(d+b) residue bits.
    let scan_total = scan_samples(points, args.t, d);
    let mut partition = OrbitPartition::new(seed, num_cpus, scan_total, args.t);
    let num_cpus = partition.num_cpus;
    let base_chunk = partition.base_chunk;
    let rem = partition.rem;
    let chunk = |i: usize| base_chunk + if i < rem { 1 } else { 0 };
    let buf_len = |i: usize| buffer_size(chunk(i), partition_bits);
    let total_buf: usize = (0..num_cpus).map(buf_len).sum();
    let split_desc = partition.split_desc();
    eprintln!(
        "Running a parallel collision test ({} CPUs, {}) on the upper {} bits of the {} \
         using {} points ({}-bit cells, {:.3} GiB RAM{})",
        num_cpus,
        split_desc,
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
        (total_buf * size_of::<T>()) as f64 / 2.0f64.powi(30),
        mode_suffix
    );

    let cells_suffix = effective_cells_suffix(d, args.u, args.t);
    eprintln!(
        "u: {} t: {} cells: {:.0} expected collisions: {}{}",
        args.u, args.t, cells, lambda, cells_suffix
    );

    let full = args.u == 64 && args.s == 0;

    let params = GridParams {
        u: args.u,
        t: args.t,
        s: args.s,
        d,
        cells,
    };

    let mut sw = Stopwatch::new();
    let mut tot: u128 = 0;
    let mut lambda_sum = 0.0f64;
    let cells_f64 = cells.to_f64().unwrap();
    // Cells covered by a single tradeoff pass (the whole space when no tradeoff),
    // needed for the per-pass Poisson means.
    let cells_per_pass = cells_f64 / num_passes as f64;

    // Parallel checkpoints: only with decimation (⇒ num_passes == 1). Stages run
    // sequentially; each stage's contiguous sample-range is split across threads,
    // the per-thread sorted results are k-way merged into one stage run and folded
    // into a cumulative sorted buffer, and a cumulative p-value is emitted. With
    // num_cpus == 1 this reproduces the sequential checkpoint runner exactly.
    if args.checkpoints {
        let effective_cells_f64 = cells.to_f64().unwrap();
        let num_checkpoints = (((1u64 << d) as f64).sqrt() as usize).clamp(1, scan_total);
        let acc_cap = buffer_size(scan_total, partition_bits);
        let max_stage = scan_total.div_ceil(num_checkpoints);
        let thread_cap = buffer_size(max_stage.div_ceil(num_cpus) + 1, partition_bits).max(1);
        for rep in 1..=args.reps {
            let mut acc = alloc_mmap::<T>(acc_cap);
            let mut acc_len = 0usize;
            let mut bufs: Vec<MmapMut> =
                (0..num_cpus).map(|_| alloc_mmap::<T>(thread_cap)).collect();
            let mut scanned = 0usize;
            let mut c = 0usize;
            for k in 1..=num_checkpoints {
                let mut psw = Stopwatch::new();
                eprint!("Checkpoint {}/{}: gen...", k, num_checkpoints);
                let target_scanned = scan_total * k / num_checkpoints;
                let stage = target_scanned - scanned;
                let sbase = stage / num_cpus;
                let srem = stage % num_cpus;
                let schunk = |i: usize| sbase + if i < srem { 1 } else { 0 };
                let sstart = |i: usize| i * sbase + i.min(srem);

                // Per-thread orbit starts within this stage's sample-range.
                let boundaries: Box<[usize]> = (0..num_cpus).map(sstart).collect();
                let snapshots =
                    partition.snapshots((rep - 1) * scan_total + scanned, stage, &boundaries, None);

                // Phase 1 — each thread scans its stage sub-range (decimation, no tradeoff).
                let results: Box<[(usize, Prng)]> = std::thread::scope(|scope| {
                    let handles: Vec<_> = bufs
                        .iter_mut()
                        .enumerate()
                        .zip(snapshots.iter())
                        .map(|((i, mapped), snap)| {
                            let params = &params;
                            let snap = *snap;
                            let sl = schunk(i);
                            scope.spawn(move || {
                                let buf: &mut [T] = bytemuck::try_cast_slice_mut(mapped).unwrap();
                                gen_pass_dispatch::<T>(snap, params, buf, sl, 0, 0, true, full)
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                eprint!("[{:.3}s] sort...", psw.lap());

                // Phase 2 — sort each thread's prefix, one buffer at a time with the
                // multithreaded sort (see the main pass loop: num_cpus concurrent
                // single-threaded sorts serialize on the glibc malloc arena lock).
                for (mapped, (used, _)) in bufs.iter_mut().zip(&results) {
                    let buf: &mut [T] = bytemuck::try_cast_slice_mut(mapped).unwrap();
                    T::sort_mt(&mut buf[..*used]);
                }
                eprint!("[{:.3}s] merge...", psw.lap());

                // Merge the per-thread sorted sub-buffers and fold into the cumulative buffer.
                let slices: Box<[&[T]]> = bufs
                    .iter()
                    .zip(&results)
                    .map(|(m, (used, _))| {
                        let all: &[T] = bytemuck::cast_slice(m.as_ref());
                        &all[..*used]
                    })
                    .collect();
                let stage_run = merge_sorted::<T>(&slices);
                let acc_slice: &mut [T] = bytemuck::try_cast_slice_mut(&mut acc).unwrap();
                // The cumulative kept count is itself headroom-bounded; check before
                // the merge writes past the end of the accumulator.
                if acc_len + stage_run.len() > acc_slice.len() {
                    bin_overflow("the checkpoint accumulator");
                }
                crate::test::merge_into(acc_slice, acc_len, &stage_run);
                acc_len += stage_run.len();
                scanned = target_scanned;
                eprint!("[{:.3}s] count...", psw.lap());

                c = crate::test::count_adjacent_equals(&acc_slice[..acc_len]);
                let lambda_cp = expected_collisions(acc_len as f64, effective_cells_f64);
                eprintln!(
                    "[{:.3}s], {acc_len} points, {c} collisions\tp={}",
                    psw.lap(),
                    format_p_value(p_value(c as f64, lambda_cp), args.pretty_p),
                );
            }
            tot += c as u128;
            // Condition the per-rep Poisson mean on the points actually kept.
            let lambda_rep = test_lambda(acc_len, effective_cells_f64, false);
            lambda_sum += lambda_rep;
            eprintln!(
                "{}\tp={}\tcombined: {}\tp={}",
                c,
                format_p_value(p_value(c as f64, lambda_rep), args.pretty_p),
                tot,
                format_p_value(p_value(tot as f64, lambda_sum), args.pretty_p)
            );
        }
        eprintln!("Test completed in {:.2} seconds", sw.lap());
        return (tot, lambda_sum);
    }

    for rep in 1..=args.reps {
        // Per-thread buffers (sized to each thread's own chunk), reused across
        // every pass of this repetition.
        let mut bufs: Vec<MmapMut> = (0..num_cpus).map(|i| alloc_mmap::<T>(buf_len(i))).collect();

        // Per-thread orbit starts for this rep (jump-ahead or chained pre-scan).
        let snapshots = partition.rep_snapshots(rep);

        let mut rep_coll = 0usize;
        let mut total_points = 0usize;
        // Single-pass mode (--pass K) runs only value-interval K; the per-pass
        // counts are independently summable, so one interval can run alone.
        let (pass_lo, pass_hi) = match args.pass {
            Some(k) => (k, k + 1),
            None => (0, num_passes),
        };
        for pass in pass_lo..pass_hi {
            // Generate / sort / count are run as separate, independently-timed phases,
            // matching the sequential run_collision_tradeoff progress line.
            let mut psw = Stopwatch::new();
            eprint!("Pass {}/{}: gen...", pass + 1, num_passes);

            // Phase 1 — generate (no sort): each thread fills its buffer from its
            // own orbit snapshot and reports how many it kept plus its next orbit
            // position. The mutable borrow of bufs ends with the scope.
            let results: Box<[(usize, Prng)]> = std::thread::scope(|scope| {
                let handles: Vec<_> = bufs
                    .iter_mut()
                    .enumerate()
                    .zip(snapshots.iter())
                    .map(|((i, mapped), snap)| {
                        let params = &params;
                        let snap = *snap;
                        let stream_len = chunk(i);
                        scope.spawn(move || {
                            let buf: &mut [T] = bytemuck::try_cast_slice_mut(mapped).unwrap();
                            gen_pass_dispatch::<T>(
                                snap, params, buf, stream_len, pass, tradeoff_b, decimating, full,
                            )
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            eprint!("[{:.3}s] sort...", psw.lap());

            // Phase 2 — sort each thread's prefix, one buffer at a time with the
            // multithreaded sort. Running num_cpus single-threaded sorts concurrently
            // instead serializes them on the glibc malloc arena lock: ska_sort
            // reallocs a bucket array per recursion node, and num_cpus threads doing
            // that at once spend essentially all their time in __lll_lock (≈100% sys,
            // no progress). One sort at a time uses every core on a single buffer and
            // touches the allocator far less; sort_mt is in-place, so peak RSS is unchanged.
            for (mapped, (used, _)) in bufs.iter_mut().zip(&results) {
                let buf: &mut [T] = bytemuck::try_cast_slice_mut(mapped).unwrap();
                T::sort_mt(&mut buf[..*used]);
            }
            eprint!("[{:.3}s] count...", psw.lap());

            // Phase 3 — count: k-way merge across the sorted per-thread buffers,
            // so collisions that span two threads are counted too.
            let slices: Box<[&[T]]> = bufs
                .iter()
                .zip(&results)
                .map(|(m, (used, _))| {
                    let all: &[T] = bytemuck::cast_slice(m.as_ref());
                    &all[..*used]
                })
                .collect();
            let c = merge_count_collisions::<T>(&slices);

            // Per-pass and cumulative statistics, formatted exactly like the
            // sequential run_collision_tradeoff per-pass line.
            let pass_points: usize = results.iter().map(|(used, _)| *used).sum();
            total_points += pass_points;
            rep_coll += c;
            let lambda_pass = expected_collisions(pass_points as f64, cells_per_pass);
            let lambda_so_far =
                expected_collisions(total_points as f64, (pass + 1) as f64 * cells_per_pass);
            eprintln!(
                "[{:1.3}s], {pass_points} points, {c} collisions, p={}; combined: {total_points} points, {rep_coll} collisions, p={}",
                psw.lap(),
                format_p_value(p_value(c as f64, lambda_pass), args.pretty_p),
                format_p_value(p_value(rep_coll as f64, lambda_so_far), args.pretty_p),
            );
        }

        tot += rep_coll as u128;
        // Condition the per-rep Poisson mean on the points actually kept.
        let lambda_rep = test_lambda(total_points, cells_f64, false);
        lambda_sum += lambda_rep;
        eprintln!(
            "{}\tp={}\tcombined: {}\tp={}",
            rep_coll,
            format_p_value(p_value(rep_coll as f64, lambda_rep), args.pretty_p),
            tot,
            format_p_value(p_value(tot as f64, lambda_sum), args.pretty_p)
        );
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, lambda_sum)
}

/// Parallel birthday-spacings test with the two-level top-bit tradeoff.
///
/// Reuses [`run_test_parallel`]'s faithful orbit split. The combined index is split
/// into 2*ᵇ* contiguous value intervals by its top *b* bits, visited in order; per
/// interval the per-thread blocks are gathered into one buffer and sorted. The sorted
/// interval is kept read-only; its spacings (`interval[i] − interval[i−1]`, with the
/// first element of the first non-empty interval taken against `prev_max`, the previous
/// interval's maximum) are computed on the fly by the parallel filter and the matching
/// ones compacted into the current spacing-class's buffer. Spacings are accumulated per
/// spacing-class (their low *b* bits — balanced, unlike the top bits which cluster near
/// zero) and counted. With `b == 0` this degenerates to a plain parallel birthday test
/// (one interval, one class). The summed count is bit-identical to the sequential
/// [`run_birthday_tradeoff`] for any CPU count.
///
/// Returns the total collision count and the summed per-repetition Poisson
/// means, each conditioned on the points the repetition actually kept (see
/// [`run_test`]).
fn run_birthday_parallel<T: Cell>(
    args: &Args,
    points: usize,
    cells: &BigUint,
    lambda: f64,
    num_cpus: usize,
) -> (u128, f64) {
    let seed = args.seed.unwrap_or_else(current_nanos_seed);
    eprintln!("Seed: {:#018x}", seed);

    let d = args.decimate.unwrap_or(0);
    let b = args.tradeoff_bits();
    let num_passes: u64 = 1u64 << b;
    let t = args.t;
    let decimating = d > 0;
    let full = args.u == 64 && args.s == 0;
    let partition_bits = t * d + b;
    let spacing_mask = T::low_bits_mask(b);
    // cells itself may be unrepresentable (2^N in N-bit storage); cells − 1 always
    // is, and the wrap-around spacing is evaluated through it.
    let cells_m1 = T::from_u128((cells - BigUint::from(1u8)).to_u128().unwrap());

    let scan_total = scan_samples(points, t, d);
    let mut partition = OrbitPartition::new(seed, num_cpus, scan_total, t);
    let num_cpus = partition.num_cpus;
    let base_chunk = partition.base_chunk;
    let rem = partition.rem;
    let chunk = |i: usize| base_chunk + if i < rem { 1 } else { 0 };

    let block_cap = |i: usize| buffer_size(chunk(i), partition_bits).max(1);
    let interval_cap = buffer_size(scan_total, partition_bits).max(1);
    let class_cap = buffer_size(points, b).max(1);

    let params = GridParams {
        u: args.u,
        t,
        s: args.s,
        d,
        cells,
    };

    let split_desc = partition.split_desc();

    let output_type = bits_read_desc(args.s);

    let mut mode_parts: Vec<String> = Vec::new();
    if b > 0 {
        // The birthday tradeoff is two-level: 2^b value intervals (inner sweep) by
        // 2^b spacing classes (outer sweep).
        mode_parts.push(format!(
            "tradeoff on {} top bits over {} value intervals x {} spacing classes",
            b, num_passes, num_passes
        ));
    }
    if d > 0 {
        mode_parts.push(decimation_desc(d, t));
    }
    let mode_suffix = join_mode_parts(&mode_parts);

    // Live memory: the per-thread interval blocks plus the shared interval and
    // class buffers, all resident together within a repetition.
    let live_elems: usize = (0..num_cpus).map(block_cap).sum::<usize>() + interval_cap + class_cap;

    eprintln!(
        "Running a parallel birthday-spacings test ({} CPUs, {}) on the upper {} bits of the {} \
         using {} points ({}-bit cells, {:.3} GiB RAM{})",
        num_cpus,
        split_desc,
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
        (live_elems * size_of::<T>()) as f64 / 2.0f64.powi(30),
        mode_suffix
    );
    eprintln!(
        "u: {} t: {} cells: {:.0} expected collisions: {}",
        args.u, t, cells, lambda
    );

    let mut sw = Stopwatch::new();
    let mut tot: u128 = 0;
    let mut lambda_sum = 0.0f64;
    let cells_f64 = cells.to_f64().unwrap();

    for rep in 1..=args.reps {
        let mut blocks: Vec<MmapMut> = (0..num_cpus)
            .map(|i| alloc_mmap::<T>(block_cap(i)))
            .collect();
        let mut interval_buf = alloc_mmap::<T>(interval_cap);
        let mut class_buf = alloc_mmap::<T>(class_cap);

        // Per-thread orbit starts for the scan sub-ranges, reused for every interval.
        let snapshots = partition.rep_snapshots(rep);

        let mut rep_coll = 0usize;
        let mut rep_points = 0usize;
        let mut psw = Stopwatch::new();
        // Single-pass mode (--pass K) runs only spacing-class K; each class still
        // sweeps every value-interval internally, but per-class counts are
        // independently summable, so one class can run alone.
        let (pass_lo, pass_hi) = match args.pass {
            Some(k) => (k, k + 1),
            None => (0, num_passes),
        };
        eprintln!(
            "Rep {}/{}: {} value intervals x {} spacing classes",
            rep, args.reps, num_passes, num_passes
        );
        // Per-class progress heartbeat (collision-style): each spacing-class sweeps
        // all 2^b value-intervals, so without this the rep is silent for hours.
        let mut class_sw = Stopwatch::new();
        // Nominal per-class Poisson mean (lambda_total / 2^b) for the progress
        // p-values; the final rep line below conditions on the actual kept count.
        let lambda_class = test_lambda(points, cells_f64, true) / num_passes as f64;

        for j in pass_lo..pass_hi {
            let class_target = T::from_u64(j);
            let class: &mut [T] = bytemuck::try_cast_slice_mut(&mut class_buf).unwrap();
            let mut class_len = 0usize;
            let mut prev_max: Option<T> = None;
            let mut global_min: Option<T> = None;
            let mut global_max: Option<T> = None;

            for k in 0..num_passes {
                // Per-interval heartbeat: each value-interval is one full scan, so it
                // is the true analog of a collision pass; print gen.../sort... as the
                // phases complete, exactly like the collision per-pass line.
                let mut isw = Stopwatch::new();
                eprint!(
                    "  Class {}/{}, interval {}/{}: gen...",
                    j + 1,
                    num_passes,
                    k + 1,
                    num_passes
                );
                // Phase 1: faithful parallel generation of interval k's points.
                let lens: Box<[usize]> = std::thread::scope(|scope| {
                    let handles: Vec<_> = blocks
                        .iter_mut()
                        .enumerate()
                        .zip(snapshots.iter())
                        .map(|((i, mapped), snap)| {
                            let snap = *snap;
                            let params = &params;
                            let stream_len = chunk(i);
                            scope.spawn(move || {
                                let buf: &mut [T] = bytemuck::try_cast_slice_mut(mapped).unwrap();
                                let (len, _) = gen_pass_dispatch::<T>(
                                    snap, params, buf, stream_len, k, b, decimating, full,
                                );
                                len
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });

                let total: usize = lens.iter().sum();
                // Each kept point lies in exactly one value interval, so summing
                // the intervals of one distance sweep (the first executed) counts the
                // points; pass_lo is 0 for a full run and K for a single-pass run.
                if j == pass_lo {
                    rep_points += total;
                }
                if total == 0 {
                    eprintln!("[{:.3}s] empty", isw.lap());
                    continue;
                }
                // Each block respects its own headroom, but their sum can exceed the
                // interval buffer's (smaller) global headroom; check before slicing.
                if total > interval_cap {
                    bin_overflow("a birthday value interval");
                }
                eprint!("[{:.3}s] sort...", isw.lap());
                // Gather the per-thread blocks into one contiguous interval buffer,
                // in parallel: split the buffer into disjoint prefix-sum slots and
                // let each thread copy its own block into its slot.
                {
                    let interval: &mut [T] =
                        &mut bytemuck::try_cast_slice_mut(&mut interval_buf).unwrap()[..total];
                    let mut rest = interval;
                    let mut dsts: Vec<&mut [T]> = Vec::with_capacity(num_cpus);
                    for &len in &lens {
                        let (head, tail) = rest.split_at_mut(len);
                        dsts.push(head);
                        rest = tail;
                    }
                    let dsts = dsts.into_boxed_slice();
                    std::thread::scope(|scope| {
                        for (i, dst) in dsts.into_iter().enumerate() {
                            let block = &blocks[i];
                            scope.spawn(move || {
                                let src: &[T] = &bytemuck::cast_slice(block.as_ref())[..dst.len()];
                                dst.copy_from_slice(src);
                            });
                        }
                    });
                }
                let interval: &mut [T] =
                    &mut bytemuck::try_cast_slice_mut(&mut interval_buf).unwrap()[..total];
                T::sort_mt(interval);
                let interval_max = interval[total - 1];
                if global_min.is_none() {
                    global_min = Some(interval[0]);
                }
                global_max = Some(interval_max);

                eprint!("[{:.3}s] filter...", isw.lap());
                // Compact this interval's matching spacings into the class buffer. A
                // spacing is interval[i] − interval[i−1]; the first element of the very
                // first interval has no predecessor (its wrap is deferred), so it starts
                // at 1. The spacings are never reused (interval_max and interval[0] were
                // already captured) and the class buffer is sorted later, so order is
                // irrelevant: keep interval read-only and compact in parallel with a
                // count-then-write two-pass over num_cpus chunks. This was the one phase
                // that stayed single-threaded.
                let start = if prev_max.is_none() { 1 } else { 0 };
                let interval: &[T] = interval;
                let span = total - start;
                // Pass 1 — count matching spacings per chunk.
                let counts: Box<[usize]> = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..num_cpus)
                        .map(|c| {
                            scope.spawn(move || {
                                let (lo, hi) = (
                                    start + c * span / num_cpus,
                                    start + (c + 1) * span / num_cpus,
                                );
                                let mut cnt = 0usize;
                                for i in lo..hi {
                                    let s = if i == 0 {
                                        interval[0] - prev_max.unwrap()
                                    } else {
                                        interval[i] - interval[i - 1]
                                    };
                                    if s & spacing_mask == class_target {
                                        cnt += 1;
                                    }
                                }
                                cnt
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let matched: usize = counts.iter().sum();
                if class_len + matched > class.len() {
                    bin_overflow("a birthday-spacings class");
                }
                // Pass 2 — write each chunk's matches into its own disjoint slot of the
                // class buffer (prefix-sum offsets), recomputing the same spacings.
                {
                    let mut rest = &mut class[class_len..class_len + matched];
                    let mut dsts: Vec<&mut [T]> = Vec::with_capacity(num_cpus);
                    for &cnt in &counts {
                        let (head, tail) = rest.split_at_mut(cnt);
                        dsts.push(head);
                        rest = tail;
                    }
                    let dsts = dsts.into_boxed_slice();
                    std::thread::scope(|scope| {
                        for (c, dst) in dsts.into_iter().enumerate() {
                            scope.spawn(move || {
                                let (lo, hi) = (
                                    start + c * span / num_cpus,
                                    start + (c + 1) * span / num_cpus,
                                );
                                let mut w = 0usize;
                                for i in lo..hi {
                                    let s = if i == 0 {
                                        interval[0] - prev_max.unwrap()
                                    } else {
                                        interval[i] - interval[i - 1]
                                    };
                                    if s & spacing_mask == class_target {
                                        dst[w] = s;
                                        w += 1;
                                    }
                                }
                            });
                        }
                    });
                }
                class_len += matched;
                eprintln!("[{:.3}s], {total} points", isw.lap());
                prev_max = Some(interval_max);
            }

            // Wrap-around spacing of the global minimum: cells − global_max + global_min.
            // It equals cells — possibly unrepresentable — iff gmin == gmax, i.e., all
            // points coincide; every other spacing is then 0 ≠ wrap, so dropping it
            // leaves the collision count unchanged.
            if let (Some(gmin), Some(gmax)) = (global_min, global_max) {
                if gmin != gmax {
                    let mut wrap = cells_m1 - gmax;
                    wrap += gmin;
                    wrap += T::from_u64(1);
                    if wrap & spacing_mask == class_target {
                        *class
                            .get_mut(class_len)
                            .unwrap_or_else(|| bin_overflow("a birthday-spacings class")) = wrap;
                        class_len += 1;
                    }
                }
            }
            T::sort_mt(&mut class[..class_len]);
            let class_coll = count_adjacent_equals(&class[..class_len]);
            rep_coll += class_coll;
            let classes_done = (j - pass_lo + 1) as f64;
            eprintln!(
                "  Class {}/{} done: [{:.3}s], {class_len} spacings, {class_coll} collisions, p={}; combined: {rep_coll} collisions, p={}",
                j + 1,
                num_passes,
                class_sw.lap(),
                format_p_value(p_value(class_coll as f64, lambda_class), args.pretty_p),
                format_p_value(
                    p_value(rep_coll as f64, classes_done * lambda_class),
                    args.pretty_p
                ),
            );
        }

        tot += rep_coll as u128;
        // Condition the per-rep Poisson mean on the points actually kept.
        let lambda_rep = test_lambda(rep_points, cells_f64, true);
        lambda_sum += lambda_rep;
        eprintln!(
            "[{:.3}s] {}\tp={}\tcombined: {}\tp={}",
            psw.lap(),
            rep_coll,
            format_p_value(p_value(rep_coll as f64, lambda_rep), args.pretty_p),
            tot,
            format_p_value(p_value(tot as f64, lambda_sum), args.pretty_p)
        );
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, lambda_sum)
}

/// Sequentially walk `total_cells` samples of the orbit starting from `start`
/// (each sample consumes exactly *t* PRNG draws, in every mode — decimation still
/// draws *t* per sample, it just conditionally rejects the result), capturing a
/// copy of the generator when the walk reaches each cell index in `boundaries`
/// (which must be sorted ascending and lie in `0..total_cells`). Returns one
/// snapshot per boundary plus the end state (after all `total_cells` cells), so
/// the caller can chain the next repetition's walk. Snapshot `i` is bit-identical
/// to `try_skip(boundaries[i] * t)` for jump-capable generators.
///
/// Used to give non-jumpable generators a faithful parallel split: the walk is
/// inherently sequential (it follows the orbit) and cannot be parallelized
/// without jump-ahead, which is exactly what these generators lack.
fn prescan_checkpoints(
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
fn gen_pass_dispatch<T: Cell>(
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
fn test_lambda(points: usize, cells: f64, birthday_spacings: bool) -> f64 {
    if birthday_spacings {
        // TestU01 long guide: lambda = n³ / (4k).
        BigUint::from(points).pow(3).to_f64().unwrap() / (cells * 4.0)
    } else {
        expected_collisions(points as f64, cells)
    }
}

/// Computes the Poisson mean and the point count to use, applying the test-specific defaults.
fn compute_lambda_and_points(args: &Args, cells: &BigUint) -> (f64, usize) {
    // cells is already the effective cell count: main() computes it as
    // 2^((u − d) · t), incorporating any whole-tuple decimation.
    let effective_cells_f64 = cells.to_f64().unwrap();

    // In tradeoff mode the user-supplied m is the per-pass memory; the actual
    // number of points is m · 2^b, where b is the total number of top tradeoff
    // bits of the combined index (the partition has 2^b contiguous value intervals).
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
        // point count is m · 2^b (only ~points / 2^b are ever resident), mirroring
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
        points = m.checked_mul(pass_factor).expect("m · 2^b overflows usize");

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

fn main() {
    cli::init_env_logger().expect("Failed to initialize the logger");

    let args = Args::parse();
    args.validate();

    eprintln!("Generator: {}", Prng::NAME);

    // Report the kernel's transparent-huge-page policy once: if it reads "[never]",
    // MADV_HUGEPAGE is ignored system-wide and the large buffers stay base-paged
    // regardless of what alloc_mmap requests (a separate, system-level cause).
    #[cfg(target_os = "linux")]
    if let Ok(thp) = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
        eprintln!("Transparent huge pages: {}", thp.trim());
    }

    let cells = BigUint::from(2u32)
        .pow((args.u - args.decimate.unwrap_or(0)) as _)
        .pow(args.t as u32);
    if cells > BigUint::from(2u32).pow(128) {
        Args::die("you cannot have more than 2^128 cells");
    }

    let (lambda, points) = compute_lambda_and_points(&args, &cells);

    let (tot, lambda_total) = if let Some(num_cpus) = args.parallel_cpus() {
        if args.birthday_spacings {
            dispatch(
                &cells,
                || run_birthday_parallel::<u32>(&args, points, &cells, lambda, num_cpus),
                || run_birthday_parallel::<u64>(&args, points, &cells, lambda, num_cpus),
                || run_birthday_parallel::<u128>(&args, points, &cells, lambda, num_cpus),
            )
        } else {
            dispatch(
                &cells,
                || run_test_parallel::<u32>(&args, points, &cells, lambda, num_cpus),
                || run_test_parallel::<u64>(&args, points, &cells, lambda, num_cpus),
                || run_test_parallel::<u128>(&args, points, &cells, lambda, num_cpus),
            )
        }
    } else {
        dispatch(
            &cells,
            || run_test::<u32>(&args, points, &cells, lambda),
            || run_test::<u64>(&args, points, &cells, lambda),
            || run_test::<u128>(&args, points, &cells, lambda),
        )
    };

    if let Some(k) = args.pass {
        // Single-pass mode (--pass): emit this unit's raw count and its nominal
        // lambda share (lambda_total / 2^b, summed over repetitions). Summing all
        // 2^b single-pass runs reproduces a full -b run's (count, lambda, p) for a
        // non-decimated test; see docs/superpowers/specs/2026-06-18-single-pass-tradeoff-design.md.
        let num_passes = 1u64 << args.tradeoff_bits();
        let lambda_k = args.reps as f64
            * test_lambda(points, cells.to_f64().unwrap(), args.birthday_spacings)
            / num_passes as f64;
        eprintln!(
            "Single pass {k} of {num_passes}: recombine the 2^b runs via p_value(Σ counts, Σ lambdas)"
        );
        println!("{tot}\tlambda={lambda_k}");
    } else {
        // lambda_total is the sum of the per-repetition means, each conditioned on
        // the points that repetition actually kept (= lambda · reps when not
        // decimating).
        println!(
            "{}\tp={}",
            tot,
            format_p_value(p_value(tot as f64, lambda_total), args.pretty_p)
        );
    }
}

// Gated away from the degenerate incr counter: these compare a parallel run to
// the sequential one in tradeoff/decimation modes, whose buffer headroom assumes
// ~uniform spread across residue bins. A counter that maps every sample to ~one
// cell overflows that headroom (same rationale as the splitmix gate on the
// test module). Every real generator (splitmix, wyrand, MSWS-CTR, LCG, MWC,
// Romu) exercises both the jump and pre-scan snapshot paths here.
#[cfg(all(test, not(feature = "incr")))]
mod parallel_faithful_tests {
    use super::*;

    fn make_args(u: usize, t: usize, m: usize, tradeoff: Option<usize>, seed: u64) -> Args {
        Args {
            u,
            t,
            m: Some(m),
            s: 0,
            tradeoff,
            decimate: None,
            checkpoints: false,
            birthday_spacings: false,
            reps: 1,
            seed: Some(seed),
            pretty_p: false,
            parallel: Some(0),
            pass: None,
        }
    }

    fn cells_for(args: &Args) -> BigUint {
        BigUint::from(2u32).pow(args.u as u32).pow(args.t as u32)
    }

    // Single-pass (--pass K) runs one of the 2^b summable units; the per-unit
    // counts must sum to a full -b run, in both the sequential and parallel paths.
    // This pins the recombination guarantee of the single-pass design.
    #[test]
    fn single_pass_collision_sum_matches_full() {
        let seed = 0xABCD_1234_5678_9ABC;
        let b = 2u32;
        let mut args = make_args(16, 2, 1 << 16, Some(b as usize), seed);
        let cells = cells_for(&args);
        let (lambda, points) = compute_lambda_and_points(&args, &cells);

        let (full_seq, _) = run_test::<u64>(&args, points, &cells, lambda);
        let (full_par, _) = run_test_parallel::<u64>(&args, points, &cells, lambda, 4);

        let mut sum_seq = 0u128;
        let mut sum_par = 0u128;
        for k in 0..(1u64 << b) {
            args.pass = Some(k);
            sum_seq += run_test::<u64>(&args, points, &cells, lambda).0;
            sum_par += run_test_parallel::<u64>(&args, points, &cells, lambda, 4).0;
        }
        assert_eq!(
            full_seq, sum_seq,
            "sequential single-pass counts must sum to full"
        );
        assert_eq!(
            full_par, sum_par,
            "parallel single-pass counts must sum to full"
        );

        // The per-pass nominal lambda shares (lambda_total / 2^b) sum back to the
        // full lambda exactly (power-of-two divisor → exact in f64).
        let num_passes = 1u64 << b;
        let lambda_k = test_lambda(points, cells.to_f64().unwrap(), false) / num_passes as f64;
        assert_eq!(
            num_passes as f64 * lambda_k,
            test_lambda(points, cells.to_f64().unwrap(), false),
            "single-pass lambda shares must sum to the full lambda"
        );
    }

    #[test]
    fn single_pass_birthday_sum_matches_full() {
        let seed = 0x0BAD_F00D_DEAD_BEEF;
        let b = 2u32;
        let mut args = make_args(30, 2, 1 << 16, Some(b as usize), seed);
        args.birthday_spacings = true;
        let cells = cells_for(&args);
        let (lambda, points) = compute_lambda_and_points(&args, &cells);

        let (full_seq, _) = run_test::<u64>(&args, points, &cells, lambda);
        let (full_par, _) = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 4);

        let mut sum_seq = 0u128;
        let mut sum_par = 0u128;
        for k in 0..(1u64 << b) {
            args.pass = Some(k);
            sum_seq += run_test::<u64>(&args, points, &cells, lambda).0;
            sum_par += run_birthday_parallel::<u64>(&args, points, &cells, lambda, 4).0;
        }
        assert_eq!(
            full_seq, sum_seq,
            "sequential single-pass birthday counts must sum to full"
        );
        assert_eq!(
            full_par, sum_par,
            "parallel single-pass birthday counts must sum to full"
        );
    }

    // Every non-decimating generator now has a faithful parallel split, so a
    // parallel run must equal the sequential run for the same seed: jump-capable
    // generators reach each thread's start via jump-ahead, others via pre-scan.
    #[test]
    fn faithful_plain_matches_sequential() {
        let seed = 0x00C0_FFEE_1234_5678;
        let args = make_args(12, 2, 1 << 18, None, seed);
        let cells = cells_for(&args);
        let (lambda, points) = compute_lambda_and_points(&args, &cells);
        let seq = run_test::<u64>(&args, points, &cells, lambda);
        let par = run_test_parallel::<u64>(&args, points, &cells, lambda, 4);
        assert_eq!(
            seq, par,
            "faithful plain parallel must equal the sequential run"
        );
    }

    #[test]
    fn faithful_tradeoff_matches_sequential() {
        let seed = 0x0D15_EA5E_0BAD_F00D;
        let args = make_args(12, 2, 1 << 14, Some(2), seed);
        let cells = cells_for(&args);
        let (lambda, points) = compute_lambda_and_points(&args, &cells);
        let seq = run_test::<u64>(&args, points, &cells, lambda);
        let par = run_test_parallel::<u64>(&args, points, &cells, lambda, 3);
        assert_eq!(
            seq, par,
            "faithful tradeoff parallel must equal the sequential run"
        );
    }

    // Fixed-sample decimation is faithfully parallel: scanning a fixed sample
    // budget makes each thread's contiguous sample-range reachable by jump/pre-scan.
    #[test]
    fn faithful_decimation_matches_sequential() {
        let seed = 0x0DEC_1A7E_0000_0001;
        let mut args = make_args(14, 2, 1 << 14, None, seed);
        args.decimate = Some(2);
        let cells = BigUint::from(2u32)
            .pow((args.u - 2) as u32)
            .pow(args.t as u32);
        let (lambda, points) = compute_lambda_and_points(&args, &cells);
        let seq = run_test::<u64>(&args, points, &cells, lambda);
        let par = run_test_parallel::<u64>(&args, points, &cells, lambda, 4);
        assert_eq!(
            seq, par,
            "faithful parallel decimation must equal sequential"
        );
    }

    #[test]
    fn faithful_decimation_tradeoff_matches_sequential() {
        let seed = 0x0DEC_1A7E_0000_0002;
        let mut args = make_args(14, 2, 1 << 12, Some(2), seed);
        args.decimate = Some(2);
        let cells = BigUint::from(2u32)
            .pow((args.u - 2) as u32)
            .pow(args.t as u32);
        let (lambda, points) = compute_lambda_and_points(&args, &cells);
        let seq = run_test::<u64>(&args, points, &cells, lambda);
        let par = run_test_parallel::<u64>(&args, points, &cells, lambda, 3);
        assert_eq!(
            seq, par,
            "faithful parallel decimation+tradeoff must equal sequential"
        );
    }

    fn checkpoint_args(seed: u64) -> (Args, BigUint) {
        let mut args = make_args(16, 2, 1 << 14, None, seed);
        args.decimate = Some(2);
        args.checkpoints = true;
        let cells = BigUint::from(2u32)
            .pow((args.u - 2) as u32)
            .pow(args.t as u32);
        (args, cells)
    }

    // Parallel checkpoints must be faithful: same final cumulative count for any P.
    #[test]
    fn parallel_checkpoints_match_across_cpus() {
        let (args, cells) = checkpoint_args(0x0C0C_0C0C_0000_0001);
        let (lambda, points) = compute_lambda_and_points(&args, &cells);
        let r1 = run_test_parallel::<u64>(&args, points, &cells, lambda, 1);
        let r4 = run_test_parallel::<u64>(&args, points, &cells, lambda, 4);
        assert_eq!(r1, r4, "parallel checkpoints must match across CPU counts");
    }

    // P=1 parallel checkpoints must equal the sequential checkpoint runner.
    #[test]
    fn parallel_checkpoints_p1_match_sequential_runner() {
        let (args, cells) = checkpoint_args(0x0C0C_0C0C_0000_0002);
        let (lambda, points) = compute_lambda_and_points(&args, &cells);
        let seq = run_test::<u64>(&args, points, &cells, lambda);
        let par1 = run_test_parallel::<u64>(&args, points, &cells, lambda, 1);
        assert_eq!(
            seq, par1,
            "P=1 checkpoints must equal the sequential runner"
        );
    }

    // Parallel birthday-spacings (plain, b = 0) must equal the sequential run for any
    // CPU count: the gathered interval is the same point multiset, so the spacings —
    // and their collisions — match.
    #[test]
    fn faithful_birthday_plain_matches_sequential() {
        let seed = 0x0B17_4DA9_0000_0001;
        let mut args = make_args(20, 2, 40_000, None, seed);
        args.birthday_spacings = true;
        let cells = cells_for(&args); // 2^40
        let (lambda, points) = compute_lambda_and_points(&args, &cells);
        let seq = run_test::<u64>(&args, points, &cells, lambda);
        let par1 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 1);
        let par3 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 3);
        assert_eq!(seq, par1, "P=1 parallel birthday must equal sequential");
        assert_eq!(seq, par3, "P=3 parallel birthday must equal sequential");
    }

    // Same, with the two-level top-bit tradeoff (b > 0).
    #[test]
    fn faithful_birthday_tradeoff_matches_sequential() {
        let seed = 0x0B17_4DA9_0000_0002;
        let mut args = make_args(20, 2, 10_000, Some(2), seed);
        args.birthday_spacings = true;
        let cells = cells_for(&args); // 2^40
        let (lambda, points) = compute_lambda_and_points(&args, &cells); // points = 10000 · 4
        let seq = run_test::<u64>(&args, points, &cells, lambda);
        let par1 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 1);
        let par3 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 3);
        assert_eq!(
            seq, par1,
            "P=1 parallel birthday tradeoff must equal sequential"
        );
        assert_eq!(
            seq, par3,
            "P=3 parallel birthday tradeoff must equal sequential"
        );
    }

    // Birthday at the cells == 2^32 storage boundary in u32 cells (the wrap-around
    // is evaluated through cells − 1, so no strictly-wider type is needed): the
    // parallel two-level tradeoff must still match the sequential runner.
    #[test]
    fn faithful_birthday_boundary_u32_matches_sequential() {
        let seed = 0x0B17_4DA9_0000_0003;
        let mut args = make_args(16, 2, 12_500, Some(2), seed);
        args.birthday_spacings = true;
        let cells = cells_for(&args); // exactly 2^32
        let points = 50_000usize; // m · 2^b
        let lambda = (points as f64).powi(3) / (4.0 * cells.to_f64().unwrap());
        let seq = run_test::<u32>(&args, points, &cells, lambda);
        let par3 = run_birthday_parallel::<u32>(&args, points, &cells, lambda, 3);
        assert_eq!(
            seq, par3,
            "boundary-width parallel birthday must equal sequential"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: concatenate, sort, and count adjacent-equal pairs directly.
    fn brute(slices: &[&[u64]]) -> usize {
        let mut all: Vec<u64> = slices.iter().flat_map(|s| s.iter().copied()).collect();
        all.sort_unstable();
        all.windows(2).filter(|w| w[0] == w[1]).count()
    }

    /// The segmented parallel merge must match the brute-force count for every
    /// segment count, including the awkward cases where equal values straddle a
    /// would-be splitter and where sub-ranges come out empty.
    #[test]
    fn segmented_merge_matches_brute_force() {
        // Small value range relative to the element count forces many duplicates
        // (hence collisions) and guarantees splitters land on repeated values.
        // xoroshiro128+, used only to vary the test data.
        let mut s0 = 0x1234_5678_9abc_def0u64;
        let mut s1 = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            let result = s0.wrapping_add(s1);
            s1 ^= s0;
            s0 = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
            s1 = s1.rotate_left(37);
            result
        };

        for &p in &[1usize, 2, 3, 5, 16] {
            for &range in &[4u64, 64, 1000, 100_000] {
                // Build p sorted slices of random length.
                let slices_owned: Vec<Vec<u64>> = (0..p)
                    .map(|_| {
                        let len = (next() % 500) as usize;
                        let mut v: Vec<u64> = (0..len).map(|_| next() % range).collect();
                        v.sort_unstable();
                        v
                    })
                    .collect();
                let slices: Vec<&[u64]> = slices_owned.iter().map(|v| v.as_slice()).collect();

                let expected = brute(&slices);
                // Exercise the serial path and several explicit segment counts,
                // including more segments than elements.
                for &segs in &[0usize, 1, 2, 7, 50, 100_000] {
                    let got = merge_count_collisions_segmented(&slices, segs);
                    assert_eq!(
                        got, expected,
                        "p={p} range={range} segs={segs}: got {got}, expected {expected}"
                    );
                }
                // The production entry point (machine-chosen segment count).
                assert_eq!(merge_count_collisions(&slices), expected);
            }
        }
    }

    #[test]
    fn segmented_merge_edge_cases() {
        let empty: Vec<&[u64]> = vec![];
        assert_eq!(merge_count_collisions(&empty), 0);

        let all_empty: Vec<&[u64]> = vec![&[], &[], &[]];
        assert_eq!(merge_count_collisions(&all_empty), 0);

        // Every element identical and spread across slices: collisions = n - 1.
        let a = [7u64; 5];
        let b = [7u64; 3];
        let slices: Vec<&[u64]> = vec![&a, &b];
        for segs in 0..=10 {
            assert_eq!(merge_count_collisions_segmented(&slices, segs), 7);
        }
    }
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
