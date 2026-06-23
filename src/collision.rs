/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! The collision test: sequential runners (plain, tradeoff, decimation) and the
//! faithful parallel runner [`run_test_parallel`].

use std::cmp::Reverse;
use std::mem::size_of;

use dary_heap::QuaternaryHeap;
use mmap_rs::MmapMut;
use num::BigUint;
use num::traits::ToPrimitive;
use rayon::prelude::*;

use crate::cell::Cell;
use crate::cli::Args;
use crate::common::{
    GridParams, OrbitPartition, alloc_mmap, bin_overflow, bits_read_desc, buffer_size,
    count_adjacent_equals, decimation_desc, effective_cells_suffix, gen_pass_dispatch,
    join_mode_parts, merge_into, scan_samples, test_lambda,
};
use crate::prng::Prng;
use crate::stats::{expected_collisions, format_p_value, p_value};
use crate::util::{Stopwatch, parallelism};

/// Runs a collision test.
///
/// Returns the collision count and the number of points examined (here always
/// `buf.len()`; runners with a random kept count return the actual number, on
/// which the caller conditions the Poisson mean).
///
/// See [`run_collision_tradeoff`] and [`run_collision_decimate`] for
/// alternatives that are slower, but more powerful, using the same
/// amount of memory.
pub fn run_collision<T: Cell, const DIM: usize, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
) -> (usize, usize) {
    let mut sw = Stopwatch::new();
    eprint!("Generating points...");
    for x in buf.iter_mut() {
        *x = params.draw::<T, DIM, FULL>(prng);
    }

    eprint!("[{:.3}s] sorting...", sw.lap());
    T::sort_mt(buf);

    eprint!("[{:.3}s] counting collisions...", sw.lap());
    let c = count_adjacent_equals(buf);

    eprintln!("[{:.3}s] done.", sw.lap());
    (c, buf.len())
}

/// Runs a collision test using a space/time tradeoff on the top bits.
///
/// The combined cell index is partitioned into 2*ᵇ* contiguous value intervals by
/// its top *b* bits; pass *k* keeps the points falling in interval *k*. Equal
/// points share all bits, hence land in the same interval, so the passes are
/// disjoint and their collision counts sum to the exact total while only
/// ~`points` / 2*ᵇ* points are resident at once. Decimation (zero when
/// `DECIMATE`) acts independently on the low *d* bits of each element.
///
/// A *p*-value is emitted after each pass, which can be used to estimate whether
/// the test is succeeding partway through.
///
/// Returns the collision count and the number of points actually kept across
/// all passes (equal to `points` when not decimating).
#[allow(clippy::too_many_arguments)]
pub fn run_collision_tradeoff<T: Cell, const DIM: usize, const DECIMATE: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    points: usize,
    b: usize,
    cells_per_pass: f64,
    pretty_p: bool,
    pass: Option<u64>,
) -> (usize, usize) {
    let t = params.t;
    let u = params.u;
    let d = params.d;
    let num_passes: u64 = 1u64 << (b as u64);
    // Single-pass mode (--pass K) restricts the loop to one value-interval; the
    // per-pass counts are independently summable, so one interval can run alone.
    let (pass_lo, pass_hi) = match pass {
        Some(k) => (k, k + 1),
        None => (0, num_passes),
    };

    // Fixed-sample scan: each pass scans scan_len = points · 2^(t·d) samples
    // (each sample is t draws) and keeps those that survive decimation and match
    // the pass key. With d = 0 this is just points samples, keeping ~points /
    // 2*ᵇ* per pass; the per-pass local advances by exactly scan_len samples.
    let scan_len = scan_samples(points, t, d);

    // Element width in the assembled index: decimation compacts each element to
    // u - d bits, so the combined index spans t · elem_width bits.
    let elem_width = if DECIMATE { u - d } else { u };

    // The tradeoff partitions the combined index into 2^b contiguous value intervals
    // by its top b bits: pass k holds the points whose top b bits equal k. Equal
    // points share all bits, hence the same interval, so per-pass collision counts
    // still sum to the exact total; visiting passes in order 0..2^b walks the
    // intervals in value order (which the birthday border carry will rely on).
    let key_shift = t * elem_width - b;
    let key_of = |x: T| -> T {
        let mut key = x;
        key >>= key_shift;
        key
    };

    let snapshot = *prng;
    let mut end_state = snapshot;
    let mut total_coll = 0usize;
    let mut total_len = 0usize;

    for k in pass_lo..pass_hi {
        eprint!("Pass {}/{}: gen...", k + 1, num_passes);
        let mut sw = Stopwatch::new();

        let mut local = snapshot;
        let target = T::from_u64(k);
        let mut len = 0usize;
        for _ in 0..scan_len {
            let x = if DECIMATE {
                match params.draw_decimate_once::<T, DIM, false>(&mut local) {
                    Some(x) => x,
                    None => continue,
                }
            } else {
                params.draw::<T, DIM, false>(&mut local)
            };
            if key_of(x) == target {
                *buf.get_mut(len)
                    .unwrap_or_else(|| bin_overflow("a collision tradeoff pass")) = x;
                len += 1;
            }
        }
        eprint!("[{:.3}s] sort...", sw.lap());

        let slice = &mut buf[..len];
        T::sort_mt(slice);
        eprint!("[{:.3}s] count...", sw.lap());

        let c = count_adjacent_equals(slice);
        let lambda_pass = expected_collisions(len as f64, cells_per_pass);
        total_coll += c;
        total_len += len;
        let lambda_so_far = expected_collisions(total_len as f64, (k + 1) as f64 * cells_per_pass);
        eprintln!(
            "[{:1.3}s], {len} points, {} collisions, p={}; combined: {total_len} points, {} collisions, p={}",
            sw.lap(),
            c,
            format_p_value(p_value(c as f64, lambda_pass), pretty_p),
            total_coll,
            format_p_value(p_value(total_coll as f64, lambda_so_far), pretty_p)
        );
        end_state = local;
    }
    *prng = end_state;
    (total_coll, total_len)
}

/// Runs a collision test by decimating the samples, keeping only those tuples
/// in which every coordinate has its lower *d* bits equal to zero.
///
/// A fixed budget of `points` · 2*ᵗᵈ* samples is scanned; the kept count is a
/// random variable with mean ~`points`.
///
/// Decimation multiplies the expected number of collisions by 2*ᵗᵈ* because
/// the effective number of cells is divided by the same amount. This can lead
/// to stronger results in detecting faulty generators. The idea of decimation
/// to strengthen the collision test was proposed by [Melissa O'Neill].
///
/// When `checkpoints` is true, the run is split into ⌊√(2*ᵈ*)⌋ equally spaced
/// (in `next_u64`-call count) stages, with a cumulative *p*-value emitted after
/// each — matching the per-pass cadence of [`run_collision_tradeoff`] with
/// *b* = *d*/2, so the two outputs are directly comparable. Each new chunk is
/// sorted in a small auxiliary buffer and merged into the sorted prefix in
/// `buf` via a three-pointer right-to-left merge, so the per-checkpoint cost
/// is linear (not log-linear) in the accumulated size.
///
/// Returns the collision count and the number of points actually kept.
///
/// [Melissa O'Neill]: https://www.pcg-random.org/posts/birthday-test.html
pub fn run_collision_decimate<T: Cell, const DIM: usize, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    points: usize,
    effective_cells: f64,
    checkpoints: bool,
    pretty_p: bool,
) -> (usize, usize) {
    let t = params.t;
    let d = params.d;
    // Fixed-sample: scan scan_len = points · 2^(t·d) candidate tuples and keep
    // the ~points that survive decimation. The kept count is a random variable,
    // so the buffer carries balls-into-bins headroom (see run_test's buf_len).
    let scan_len = scan_samples(points, t, d);

    if !checkpoints {
        let mut sw = Stopwatch::new();
        eprint!(
            "Scanning {scan_len} samples (decimating low {} bits per dimension)...",
            d
        );
        let mut len = 0usize;
        for _ in 0..scan_len {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *buf.get_mut(len)
                    .unwrap_or_else(|| bin_overflow("a decimated collision run")) = x;
                len += 1;
            }
        }
        eprint!("[{:.3}s] sort...", sw.lap());
        T::sort_mt(&mut buf[..len]);
        eprint!("[{:.3}s] count...", sw.lap());
        let c = count_adjacent_equals(&buf[..len]);
        eprintln!("[{:.3}s] {len} points done.", sw.lap());
        return (c, len);
    }

    // Checkpoints: split the scan_len samples into ⌊√(2^d)⌋ equal stages, keeping
    // each stage's accepted points in aux, merging into the cumulative buf, and
    // emitting a cumulative p-value after each stage.
    let num_checkpoints = (((1u64 << d) as f64).sqrt() as usize).clamp(1, scan_len);
    let aux_cap = buffer_size(scan_len.div_ceil(num_checkpoints), t * d).max(1);
    let mut aux: Vec<T> = vec![T::ZERO; aux_cap];

    let mut len = 0usize; // cumulative kept points in `buf`
    let mut scanned = 0usize; // cumulative samples scanned
    let mut c = 0usize;
    for k in 1..=num_checkpoints {
        let target_scanned = scan_len * k / num_checkpoints;
        let stage = target_scanned - scanned;
        let mut sw = Stopwatch::new();
        eprint!("Checkpoint {}/{}: gen...", k, num_checkpoints);

        let mut got = 0usize;
        for _ in 0..stage {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *aux.get_mut(got)
                    .unwrap_or_else(|| bin_overflow("a decimation checkpoint stage")) = x;
                got += 1;
            }
        }
        scanned = target_scanned;
        eprint!("[{:.3}s] sort...", sw.lap());

        T::sort_mt(&mut aux[..got]);
        eprint!("[{:.3}s] merge...", sw.lap());

        // The cumulative kept count is itself headroom-bounded; check before the
        // merge writes past the end of `buf`.
        if len + got > buf.len() {
            bin_overflow("the decimation checkpoint accumulator");
        }
        merge_into(buf, len, &aux[..got]);
        len += got;
        eprint!("[{:.3}s] count...", sw.lap());

        c = count_adjacent_equals(&buf[..len]);
        let lambda = expected_collisions(len as f64, effective_cells);
        eprintln!(
            "[{:.3}s], {len} points, {} collisions\tp={}",
            sw.lap(),
            c,
            format_p_value(p_value(c as f64, lambda), pretty_p)
        );
    }
    (c, len)
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
pub(crate) fn merge_count_collisions<T: Cell>(sorted: &[&[T]]) -> usize {
    let n_total: usize = sorted.iter().map(|s| s.len()).sum();
    // Aim for several segments per thread for load balancing.
    let num_segs = (parallelism() * 4).min(n_total.max(1));
    merge_count_collisions_segmented(sorted, num_segs)
}

/// Implementation of [`merge_count_collisions`] with an explicit segment count
/// (factored out so tests can drive the value-partitioning path deterministically
/// regardless of the host's core count).
pub(crate) fn merge_count_collisions_segmented<T: Cell>(sorted: &[&[T]], num_segs: usize) -> usize {
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
pub(crate) fn merge_sorted<T: Cell>(sorted: &[&[T]]) -> Vec<T> {
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

/// Parallel version of the collision test.
///
/// The sequential orbit segment of a pass (`scan_total` samples) is split into
/// `num_cpus` contiguous sample-ranges; each thread owns one range and one
/// buffer of `~points / num_cpus` slots, reused across passes. A thread reaches
/// the start of its range by jump-ahead (`try_skip`) or, for non-jumpable
/// generators, through a sequential pre-scan ([`crate::common::prescan_checkpoints`]);
/// repetitions *continue* each orbit rather than reseeding. The result is
/// bit-identical to the sequential [`crate::common::run_test`] for every CPU count,
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
/// [`crate::common::run_test`]).
pub fn run_test_parallel<T: Cell>(
    args: &Args,
    points: usize,
    cells: &BigUint,
    lambda: f64,
    num_cpus: usize,
) -> (u128, f64) {
    let seed = args.seed;
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
         ({} points, {}-bit cells, {} memory locations, {:.3} GiB RAM{})",
        num_cpus,
        split_desc,
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
        points >> tradeoff_b,
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
                merge_into(acc_slice, acc_len, &stage_run);
                acc_len += stage_run.len();
                scanned = target_scanned;
                eprint!("[{:.3}s] count...", psw.lap());

                c = count_adjacent_equals(&acc_slice[..acc_len]);
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
