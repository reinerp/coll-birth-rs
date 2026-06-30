/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! The birthday-spacings test.

use std::mem::size_of;

use num::BigUint;
use num::traits::ToPrimitive;
use rayon::prelude::*;

use crate::cell::Cell;
use crate::cli::Args;
use crate::common::{
    GridParams, OrbitPartition, alloc_mmap, bin_overflow, bits_read_desc, buffer_size,
    count_adjacent_equals, decimation_desc, gen_unit_contiguous, join_mode_parts, scan_samples,
    test_lambda,
};
use crate::prng::Prng;
use crate::stats::{format_p_value, p_value};
use crate::util::{Stopwatch, parallelism};

/// Runs a birthday-spacings test.
///
/// With decimation (`DECIMATE`), this uses the same fixed-sample model as the
/// collision tests: scan `points` · 2*ᵗᵈ* candidate tuples and keep the
/// ~`points` accepted (dense) indices. Without decimation it generates exactly
/// `points` points. The parallel counterpart is [`run_birthday_parallel`].
///
/// Returns the spacing-collision count and the number of points actually kept.
pub fn run_birthday<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    buf: &mut [T],
    points: usize,
) -> (usize, usize) {
    let mut sw = Stopwatch::new();
    eprint!("Generating points...");
    let len = if DECIMATE {
        let scan_len = scan_samples(points, params.t, params.d);
        let mut len = 0usize;
        for _ in 0..scan_len {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *buf.get_mut(len)
                    .unwrap_or_else(|| bin_overflow("a decimated birthday run")) = x;
                len += 1;
            }
        }
        len
    } else {
        for x in buf[..points].iter_mut() {
            *x = params.draw::<T, DIM, FULL>(prng);
        }
        points
    };
    let pts = &mut buf[..len];

    eprint!("[{:.3}s] sorting...", sw.lap());
    T::sort_mt(pts);

    eprint!("[{:.3}s] computing deltas...", sw.lap());
    compute_spacings(pts, params.cells);

    eprint!("[{:.3}s] sorting deltas...", sw.lap());
    T::sort_mt(pts);

    eprint!("[{:.3}s] counting collisions...", sw.lap());
    let c = count_adjacent_equals(pts);

    eprintln!("[{:.3}s] {len} points done.", sw.lap());
    (c, len)
}

/// Parallel in-place spacings of a sorted slice, differencing each element against
/// its predecessor in the sorted order.
///
/// `predecessor` is the largest value strictly below this slice (the previous
/// value interval maximum) so the first element receives `v[0] - predecessor`.
/// When it is `None` the first element is the global minimum and is left untouched
/// (its circular wrap-around spacing is applied separately by the caller).
pub(crate) fn compute_spacings_with_predecessor<T: Cell>(v: &mut [T], predecessor: Option<T>) {
    if v.is_empty() {
        return;
    }
    // Aim for ~10 chunks per thread, with a floor of 1024 to amortise scheduling overhead.
    let chunk_size = (v.len() / (parallelism() * 10)).max(1024);

    let chunk_tails: Vec<T> = v.chunks(chunk_size).map(|c| *c.last().unwrap()).collect();

    v.par_chunks_mut(chunk_size).for_each(|c| {
        let mut prev = *c.last().unwrap();
        for i in (1..c.len()).rev() {
            let tmp = c[i - 1];
            c[i] = prev - tmp;
            prev = tmp;
        }
    });

    // Patch the first entry of every chunk after the first against its predecessor's tail.
    v.par_chunks_mut(chunk_size)
        .enumerate()
        .skip(1)
        .for_each(|(i, c)| c[0] -= chunk_tails[i - 1]);

    // The very first element: border spacing to the previous interval, or left as the
    // global minimum (its wrap-around is deferred) when there is no predecessor.
    if let Some(p) = predecessor {
        v[0] -= p;
    }
}

/// Parallel in-place computation of the spacings of a globally-sorted point set:
/// like [`compute_spacings_with_predecessor`] with no predecessor, but the first
/// element receives the circular wrap-around `cells - max + min`.
///
/// The wrap-around lies in `[1..=cells]` and reaches `cells` exactly when `min
/// == max`, which may not be representable (`cells` can be 2*ⁿ* in *n*-bit
/// storage). It is therefore evaluated through the always-representable `cells - 1`,
/// and in the degenerate case replaced by a nonzero stand-in: all other
/// spacings are then zero, so the collision count is unaffected.
pub(crate) fn compute_spacings<T: Cell>(v: &mut [T], cells: &BigUint) {
    if v.is_empty() {
        return;
    }
    let global_min = v[0];
    let global_max = *v.last().unwrap();
    compute_spacings_with_predecessor(v, None); // leaves v[0] == global minimum
    if global_min == global_max {
        v[0] = T::from_u64(1);
    } else {
        // min + (cells - 1 - max) + 1: every intermediate fits, since min < max.
        let cells_m1 = T::from_u128((cells - BigUint::from(1u8)).to_u128().unwrap());
        v[0] += cells_m1 - global_max;
        v[0] += T::from_u64(1);
    }
}

/// Runs a birthday-spacings test using a space/time tradeoff, with two nested
/// levels (each with 2*ᵇ* passes):
///
/// - **Inner (distance) level:** the combined index is split into 2*ᵇ* contiguous
///   value intervals by its top *b* bits, walked in increasing order. Within an
///   interval the points are generated, sorted, and replaced in place by their
///   spacings (scatter-back); the previous interval's maximum is carried as the
///   border predecessor, and the global minimum's wrap-around (`cells` − max + min)
///   is applied once. Because the intervals are contiguous and visited in order,
///   these are exactly the spacings of the global sorted sequence.
///
/// - **Outer (counting) level:** spacings are classified by their *low* *b* bits,
///   not their top bits: spacings cluster near zero (≈ exponential), so a top-bit
///   split would dump almost everything into one class, whereas the low bits are
///   balanced. Only the current class's spacings are kept, sorted, and counted;
///   equal spacings share all bits hence the same class, so the per-class counts
///   sum to the exact total while only ~`points` / 2*ᵇ* spacings (and one
///   interval's ~`points` / 2*ᵇ* points) are ever resident.
///
/// The point multiset is identical to a single sweep, so the total equals the
/// plain [`run_birthday`] count.
///
/// This is the sequential, single-repetition runner: [`crate::common::run_test`]
/// owns the repetition loop and header above it and hands it a ready `Prng` and
/// buffer, with `DECIMATE`/`FULL`/`DIM` specializing the hot loop as const
/// generics. The multi-core counterpart is [`run_birthday_parallel`], which runs
/// these same two levels over a faithful orbit split and is bit-identical.
///
/// Returns the spacing-collision count and the number of points actually kept
/// (summed over the value intervals of one distance sweep; every sweep visits
/// the same point multiset).
pub fn run_birthday_tradeoff<T: Cell, const DIM: usize, const DECIMATE: bool, const FULL: bool>(
    prng: &mut Prng,
    params: &GridParams,
    class_buf: &mut [T],
    points: usize,
    b: usize,
    pretty_p: bool,
    pass: Option<u64>,
) -> (usize, usize) {
    let t = params.t;
    let u = params.u;
    let d = params.d;
    let num_passes: u64 = 1u64 << b;
    // Single-pass mode (--pass K) restricts the outer loop to one spacing-class;
    // each class still sweeps every value-interval internally, but per-class counts
    // are independently summable, so one class can run alone.
    let (pass_lo, pass_hi) = match pass {
        Some(k) => (k, k + 1),
        None => (0, num_passes),
    };
    // The per-class "combined:" suffix only adds information when more than one
    // spacing-class runs (a real tradeoff, not a single --pass class).
    let multi_pass = pass_hi - pass_lo > 1;
    let elem_width = if DECIMATE { u - d } else { u };
    let point_key_shift = t * elem_width - b; // top b bits select the value interval
    let spacing_mask = T::low_bits_mask(b); // low b bits select the spacing class
    let scan_len = scan_samples(points, t, d);
    // cells itself may be unrepresentable (2ᴺ in N-bit storage); cells − 1 always
    // is, and the wrap-around spacing is evaluated through it.
    let cells_m1 = T::from_u128((params.cells - BigUint::from(1u8)).to_u128().unwrap());

    // One value interval keeps ~points / 2ᵇ points (with balls-into-bins headroom
    // over the t·d + b selectivity bits).
    let mut scratch: Vec<T> = vec![T::ZERO; buffer_size(scan_len, t * d + b)];

    let snapshot = *prng;
    let mut end_state = snapshot;
    let mut total_coll = 0usize;
    let mut total_points = 0usize;
    let mut sw = Stopwatch::new();
    eprintln!("Birthday tradeoff over {num_passes} spacing-classes");
    // Per-class progress heartbeat (collision-style): each spacing-class sweeps all
    // 2ᵇ value-intervals, so without this the run is silent for the whole sweep.
    let mut class_sw = Stopwatch::new();
    let cells_f64 = params.cells.to_f64().unwrap();
    // Nominal per-class Poisson mean (lambda_total / 2ᵇ) for the progress p-values.
    let lambda_class = (points as f64).powi(3) / (4.0 * cells_f64) / num_passes as f64;

    for j in pass_lo..pass_hi {
        let class_target = T::from_u64(j);
        let mut class_len = 0usize;
        let mut prev_max: Option<T> = None;
        let mut global_min: Option<T> = None;
        let mut global_max: Option<T> = None;

        for k in 0..num_passes {
            let interval_target = T::from_u64(k);
            // Per-interval heartbeat: each value-interval is one full scan, the true
            // analog of a collision pass; gen.../sort... print as the phases complete.
            let mut isw = Stopwatch::new();
            eprint!(
                "  Class {}/{} interval {}/{}: gen...",
                j + 1,
                num_passes,
                k + 1,
                num_passes
            );
            // Replay the same sample stream, keeping the points in interval k.
            let mut local = snapshot;
            let mut len = 0usize;
            for _ in 0..scan_len {
                let x = if DECIMATE {
                    match params.draw_decimate_once::<T, DIM, FULL>(&mut local) {
                        Some(x) => x,
                        None => continue,
                    }
                } else {
                    params.draw::<T, DIM, FULL>(&mut local)
                };
                let mut key = x;
                key >>= point_key_shift;
                if key == interval_target {
                    *scratch
                        .get_mut(len)
                        .unwrap_or_else(|| bin_overflow("a birthday value interval")) = x;
                    len += 1;
                }
            }
            end_state = local;
            // Each kept point lies in exactly one value interval, so summing the
            // interval lengths of one distance sweep (the first executed) counts the
            // points; pass_lo is 0 for a full run and K for a single-pass run.
            if j == pass_lo {
                total_points += len;
            }
            if len == 0 {
                eprintln!("[{:.3}s] empty", isw.lap());
                continue;
            }
            eprint!("[{:.3}s] sort...", isw.lap());
            T::sort_st(&mut scratch[..len]);
            let interval_min = scratch[0];
            let interval_max = scratch[len - 1];
            if global_min.is_none() {
                global_min = Some(interval_min);
            }
            global_max = Some(interval_max);

            // Scatter-back: replace each point by its spacing to its predecessor.
            for i in (1..len).rev() {
                scratch[i] = scratch[i] - scratch[i - 1];
            }
            // The first element's spacing is the border to the previous interval;
            // for the very first interval it is the global minimum, whose spacing is
            // the deferred wrap-around, so we skip it here.
            let start = match prev_max {
                Some(pm) => {
                    scratch[0] -= pm;
                    0
                }
                None => 1,
            };
            for &s in &scratch[start..len] {
                if s & spacing_mask == class_target {
                    *class_buf
                        .get_mut(class_len)
                        .unwrap_or_else(|| bin_overflow("a birthday-spacings class")) = s;
                    class_len += 1;
                }
            }
            eprintln!("[{:.3}s], {len} points", isw.lap());
            prev_max = Some(interval_max);
        }

        // Wrap-around spacing of the global minimum: cells − global_max +
        // global_min. It equals cells (possibly unrepresentable) iff gmin ==
        // gmax, i.e., all points coincide; every other spacing is then 0 ≠
        // wrap, so dropping it leaves the collision count unchanged.
        if let (Some(gmin), Some(gmax)) = (global_min, global_max) {
            if gmin != gmax {
                let mut wrap = cells_m1 - gmax;
                wrap += gmin;
                wrap += T::from_u64(1);
                if wrap & spacing_mask == class_target {
                    *class_buf
                        .get_mut(class_len)
                        .unwrap_or_else(|| bin_overflow("a birthday-spacings class")) = wrap;
                    class_len += 1;
                }
            }
        }

        T::sort_st(&mut class_buf[..class_len]);
        let class_coll = count_adjacent_equals(&class_buf[..class_len]);
        total_coll += class_coll;
        let classes_done = (j - pass_lo + 1) as f64;
        let elapsed = class_sw.lap();
        let class_p = format_p_value(p_value(class_coll as f64, lambda_class), pretty_p);
        if multi_pass {
            eprintln!(
                "  Class {}/{} done: [{elapsed:.3}s], {class_len} spacings, {class_coll} collisions, p={class_p}; combined: {total_coll} collisions, p={}",
                j + 1,
                num_passes,
                format_p_value(
                    p_value(total_coll as f64, classes_done * lambda_class),
                    pretty_p
                ),
            );
        } else {
            eprintln!(
                "  Class {}/{} done: [{elapsed:.3}s], {class_len} spacings, {class_coll} collisions, p={class_p}",
                j + 1,
                num_passes,
            );
        }
    }
    *prng = end_state;
    eprintln!("[{:.3}s] done.", sw.lap());
    (total_coll, total_points)
}

/// Runs a birthday-spacings test using the same two nested levels as
/// [`run_birthday_tradeoff`] (each with 2*ᵇ* passes), split across `num_cpus`
/// cores via the faithful orbit partition of
/// [`crate::collision::run_test_parallel`]:
///
/// - **Inner (distance) level:** the combined index is split into 2*ᵇ* contiguous
///   value intervals by its top *b* bits, visited in increasing order. Within an
///   interval each thread fills a disjoint sub-region of one buffer from its own
///   orbit segment; the blocks are compacted into a single contiguous interval,
///   sorted, and kept read-only. Its spacings (`interval[i] − interval[i−1]`,
///   the first non-empty interval's first element taken against `prev_max`, the
///   previous interval's maximum) are computed on the fly, and the global
///   minimum's wrap-around (`cells` − max + min) is applied once, so these are
///   exactly the spacings of the global sorted sequence.
///
/// - **Outer (counting) level:** spacings are classified by their *low* *b* bits,
///   not their top bits (spacings cluster near zero, so a top-bit split would
///   dump almost everything into one class, whereas the low bits are balanced).
///   The matching spacings are compacted into the current class's buffer, sorted,
///   and counted; equal spacings share all bits hence the same class, so the
///   per-class counts sum to the exact total. With `b == 0` this degenerates to a
///   plain parallel birthday test (one interval, one class).
///
/// The result is bit-identical to the sequential [`run_birthday_tradeoff`] for
/// every CPU count and repetition.
///
/// Unlike [`run_birthday_tradeoff`] (which is an inner, single-repetition runner driven
/// by [`crate::common::run_test`]), this is the top-level parallel entry point:
/// it owns the repetition loop, header, orbit partition, and λ accumulation, and
/// resolves the decimation/output-width/dimension modes at run time rather than
/// as the const generics the sequential runner is monomorphized over.
///
/// Returns the total collision count and the summed per-repetition Poisson
/// means, each conditioned on the points the repetition actually kept (see
/// [`crate::common::run_test`]).
pub fn run_birthday_parallel<T: Cell>(
    args: &Args,
    points: usize,
    cells: &BigUint,
    lambda: f64,
    num_cpus: usize,
) -> (u128, f64) {
    let seed = args.seed;
    eprintln!("Seed: {:#018x}", seed);

    let d = args.decimate.unwrap_or(0);
    let b = args.tradeoff_bits();
    let num_passes: u64 = 1u64 << b;
    let t = args.t;
    let decimating = d > 0;
    let full = args.u == 64 && args.s == 0;
    let partition_bits = t * d + b;
    let spacing_mask = T::low_bits_mask(b);
    // cells itself may be unrepresentable (2ᴺ in N-bit storage); cells − 1 always
    // is, and the wrap-around spacing is evaluated through it.
    let cells_m1 = T::from_u128((cells - BigUint::from(1u8)).to_u128().unwrap());

    let scan_total = scan_samples(points, t, d);
    let mut partition = OrbitPartition::new(seed, num_cpus, scan_total, t);
    let num_cpus = partition.num_cpus;
    let base_chunk = partition.base_chunk;
    let rem = partition.rem;
    let chunk = |i: usize| base_chunk + if i < rem { 1 } else { 0 };

    let block_cap = |i: usize| buffer_size(chunk(i), partition_bits).max(1);
    // Per-thread sub-region capacities of the one contiguous interval buffer; their
    // prefix sums are the sub-region starts gen_unit_contiguous writes and compacts.
    let caps: Box<[usize]> = (0..num_cpus).map(block_cap).collect();
    let interval_cap: usize = caps.iter().sum();
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
        // The birthday tradeoff is two-level: 2ᵇ value intervals (inner sweep) by
        // 2ᵇ spacing classes (outer sweep).
        mode_parts.push(format!(
            "tradeoff on {} top bits over {} value intervals x {} spacing classes",
            b, num_passes, num_passes
        ));
    }
    if d > 0 {
        mode_parts.push(decimation_desc(d, t));
    }
    let mode_suffix = join_mode_parts(&mode_parts);

    // Live memory: the one contiguous interval buffer plus the class buffer, both
    // resident together within a repetition.
    let live_elems: usize = interval_cap + class_cap;

    eprintln!(
        "Running a {t}-dimensional parallel birthday-spacings test ({} CPUs, {}) on the upper {} bits of the {} \
         ({} points, {}-bit cells, {} memory locations, {:.3} GiB RAM{})",
        num_cpus,
        split_desc,
        args.u,
        output_type,
        points,
        size_of::<T>() * 8,
        points >> b,
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
        // The per-class "combined:" suffix only adds information when more than one
        // spacing-class runs (a real tradeoff, not plain mode or a single --pass class).
        let multi_pass = pass_hi - pass_lo > 1;
        eprintln!(
            "Rep {}/{}: {} value intervals x {} spacing classes",
            rep, args.reps, num_passes, num_passes
        );
        // Per-class progress heartbeat (collision-style): each spacing-class sweeps
        // all 2ᵇ value-intervals, so without this the rep is silent for hours.
        let mut class_sw = Stopwatch::new();
        // Nominal per-class Poisson mean (lambda_total / 2ᵇ) for the progress
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
                // Phase 1: faithful parallel generation of interval k into one
                // contiguous buffer. Threads fill disjoint sub-regions, then the gaps
                // left by under-filled sub-regions are compacted away.
                let unit: &mut [T] = bytemuck::try_cast_slice_mut(&mut interval_buf).unwrap();
                let total = gen_unit_contiguous::<T>(
                    unit, &caps, &snapshots, &params, &chunk, k, b, decimating, full,
                );
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
                eprint!("[{:.3}s] sort...", isw.lap());
                let interval: &mut [T] = &mut unit[..total];
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
                // count-then-write two-pass over num_cpus chunks.
                let start = if prev_max.is_none() { 1 } else { 0 };
                let interval: &[T] = interval;
                let span = total - start;
                // Pass 1: count matching spacings per chunk.
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
                // Phase 2: write each chunk matches into its own disjoint slot
                // of the class buffer (prefix-sum offsets), recomputing the
                // same spacings.
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
            // It equals cells (possibly unrepresentable) iff gmin == gmax, i.e., all
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
            let elapsed = class_sw.lap();
            let class_p = format_p_value(p_value(class_coll as f64, lambda_class), args.pretty_p);
            if multi_pass {
                eprintln!(
                    "  Class {}/{} done: [{elapsed:.3}s], {class_len} spacings, {class_coll} collisions, p={class_p}; combined: {rep_coll} collisions, p={}",
                    j + 1,
                    num_passes,
                    format_p_value(
                        p_value(rep_coll as f64, classes_done * lambda_class),
                        args.pretty_p
                    ),
                );
            } else {
                eprintln!(
                    "  Class {}/{} done: [{elapsed:.3}s], {class_len} spacings, {class_coll} collisions, p={class_p}",
                    j + 1,
                    num_passes,
                );
            }
        }

        tot += rep_coll as u128;
        // Condition the per-rep Poisson mean on the points actually kept.
        let lambda_rep = test_lambda(rep_points, cells_f64, true);
        lambda_sum += lambda_rep;
        let elapsed = psw.lap();
        let rep_p = format_p_value(p_value(rep_coll as f64, lambda_rep), args.pretty_p);
        if args.reps > 1 {
            eprintln!(
                "[{elapsed:.3}s] {rep_coll}\tp={rep_p}\tcombined: {tot}\tp={}",
                format_p_value(p_value(tot as f64, lambda_sum), args.pretty_p)
            );
        } else {
            eprintln!("[{elapsed:.3}s] {rep_coll}\tp={rep_p}");
        }
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, lambda_sum)
}

// Direct tests of the wrap-around arithmetic in compute_spacings at the
// cells == 2ᴺ storage boundary; no PRNG involved, so no feature gate.
#[cfg(test)]
mod spacing_tests {
    use super::*;
    use num::BigUint;

    // Non-degenerate at the boundary: sorted points {3, 10, 2⁶⁴ − 1} on a circle
    // of 2⁶⁴ cells have spacings {7, 2⁶⁴ − 11} and wrap 2⁶⁴ − (2⁶⁴ − 1) + 3 = 4,
    // none of which overflow u64 despite cells itself being unrepresentable.
    #[test]
    fn wrap_at_width_boundary_is_exact() {
        let cells = BigUint::from(1u8) << 64;
        let mut v = [3u64, 10, u64::MAX];
        compute_spacings(&mut v, &cells);
        assert_eq!(v[0], 4, "wrap-around spacing");
        assert_eq!(v[1], 7);
        assert_eq!(v[2], u64::MAX - 10);
    }

    // Degenerate case at the boundary: all points coincide, so the wrap-around
    // would equal cells == 2⁶⁴. It is replaced by a nonzero stand-in; the n − 1
    // zero spacings then yield exactly n − 2 collisions, the true count.
    #[test]
    fn degenerate_equal_points_at_width_boundary() {
        let cells = BigUint::from(1u8) << 64;
        let mut v = [7u64, 7, 7, 7];
        compute_spacings(&mut v, &cells);
        v.sort_unstable();
        assert_eq!(count_adjacent_equals(&v), 2);
    }
}
