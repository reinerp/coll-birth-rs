/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! The collision test: sequential runners (plain, tradeoff, decimation) and the
//! faithful parallel runner [`run_test_parallel`].

use std::mem::size_of;

use num::BigUint;
use num::traits::ToPrimitive;

use crate::cell::Cell;
use crate::cli::Args;
use crate::common::{
    GridParams, OrbitPartition, alloc_mmap, bin_overflow, bits_read_desc, buffer_size,
    count_adjacent_equals, decimation_desc, effective_cells_suffix, gen_unit_contiguous,
    join_mode_parts, merge_into, scan_samples, test_lambda,
};
use crate::prng::Prng;
use crate::stats::{expected_collisions, format_p_value, p_value};
use crate::util::Stopwatch;

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
    // The cumulative "combined:" suffix only carries new information when more than
    // one pass actually runs (a real tradeoff, not a single --pass unit).
    let multi_pass = pass_hi - pass_lo > 1;

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
        let elapsed = sw.lap();
        let local_p = format_p_value(p_value(c as f64, lambda_pass), pretty_p);
        if multi_pass {
            let lambda_so_far =
                expected_collisions(total_len as f64, (k + 1) as f64 * cells_per_pass);
            eprintln!(
                "[{elapsed:1.3}s], {len} points, {c} collisions, p={local_p}; combined: {total_len} points, {total_coll} collisions, p={}",
                format_p_value(p_value(total_coll as f64, lambda_so_far), pretty_p)
            );
        } else {
            eprintln!("[{elapsed:1.3}s], {len} points, {c} collisions, p={local_p}");
        }
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
        // Uniform per-thread sub-region capacities of the one contiguous stage buffer.
        let stage_caps: Box<[usize]> = vec![thread_cap; num_cpus].into_boxed_slice();
        let stage_buf_len = thread_cap * num_cpus;
        for rep in 1..=args.reps {
            let mut acc = alloc_mmap::<T>(acc_cap);
            let mut acc_len = 0usize;
            let mut stage_mmap = alloc_mmap::<T>(stage_buf_len);
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

                let stage_buf: &mut [T] = bytemuck::try_cast_slice_mut(&mut stage_mmap).unwrap();

                // Phase 1 — generate this stage into one contiguous buffer (decimation,
                // no tradeoff): threads fill disjoint sub-regions, gaps compacted away.
                let stage_len = gen_unit_contiguous::<T>(
                    stage_buf,
                    &stage_caps,
                    &snapshots,
                    &params,
                    &schunk,
                    0,
                    0,
                    true,
                    full,
                );
                eprint!("[{:.3}s] sort...", psw.lap());

                // Phase 2 — sort the contiguous stage run.
                T::sort_mt(&mut stage_buf[..stage_len]);
                eprint!("[{:.3}s] merge...", psw.lap());

                // Fold the sorted stage run into the cumulative sorted accumulator with a
                // two-way merge (no heap); the accumulator stays sorted so each
                // checkpoint count is a linear scan.
                let acc_slice: &mut [T] = bytemuck::try_cast_slice_mut(&mut acc).unwrap();
                // The cumulative kept count is itself headroom-bounded; check before
                // the merge writes past the end of the accumulator.
                if acc_len + stage_len > acc_slice.len() {
                    bin_overflow("the checkpoint accumulator");
                }
                merge_into(acc_slice, acc_len, &stage_buf[..stage_len]);
                acc_len += stage_len;
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
        return (tot, lambda_sum);
    }

    // Per-thread sub-region capacities of the one big buffer; their prefix sums
    // are the sub-region starts that gen_unit_contiguous writes into and compacts.
    let caps: Box<[usize]> = (0..num_cpus).map(buf_len).collect();

    for rep in 1..=args.reps {
        // One contiguous buffer (sized Σ caps == total_buf), reused across every
        // pass of this repetition: threads fill disjoint sub-regions, the gaps are
        // compacted away, then one sort + one linear count serve the whole pass.
        let mut buf_mmap = alloc_mmap::<T>(total_buf);

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
        // The per-pass "combined:" suffix only adds information when more than one
        // pass runs (a real tradeoff, not plain mode or a single --pass unit).
        let multi_pass = pass_hi - pass_lo > 1;
        for pass in pass_lo..pass_hi {
            // Generate / sort / count are run as separate, independently-timed phases,
            // matching the sequential run_collision_tradeoff progress line.
            let mut psw = Stopwatch::new();
            eprint!("Pass {}/{}: gen...", pass + 1, num_passes);

            let buf: &mut [T] = bytemuck::try_cast_slice_mut(&mut buf_mmap).unwrap();

            // Phase 1 — generate into one contiguous buffer: each thread fills its
            // own disjoint sub-region from its orbit snapshot, then the gaps left by
            // under-filled sub-regions are compacted away (a no-op in plain mode,
            // where every thread keeps exactly its chunk).
            let pass_points = gen_unit_contiguous::<T>(
                buf, &caps, &snapshots, &params, &chunk, pass, tradeoff_b, decimating, full,
            );
            eprint!("[{:.3}s] sort...", psw.lap());

            // Phase 2 — one sort over the whole contiguous unit.
            T::sort_mt(&mut buf[..pass_points]);
            eprint!("[{:.3}s] count...", psw.lap());

            // Phase 3 — one linear scan; the union is already contiguous and sorted,
            // so collisions spanning former thread boundaries are counted too.
            let c = count_adjacent_equals(&buf[..pass_points]);

            // Per-pass and cumulative statistics, formatted exactly like the
            // sequential run_collision_tradeoff per-pass line.
            total_points += pass_points;
            rep_coll += c;
            let lambda_pass = expected_collisions(pass_points as f64, cells_per_pass);
            let elapsed = psw.lap();
            let local_p = format_p_value(p_value(c as f64, lambda_pass), args.pretty_p);
            if multi_pass {
                let lambda_so_far =
                    expected_collisions(total_points as f64, (pass + 1) as f64 * cells_per_pass);
                eprintln!(
                    "[{elapsed:1.3}s], {pass_points} points, {c} collisions, p={local_p}; combined: {total_points} points, {rep_coll} collisions, p={}",
                    format_p_value(p_value(rep_coll as f64, lambda_so_far), args.pretty_p),
                );
            } else {
                eprintln!("[{elapsed:1.3}s], {pass_points} points, {c} collisions, p={local_p}");
            }
        }

        tot += rep_coll as u128;
        // Condition the per-rep Poisson mean on the points actually kept.
        let lambda_rep = test_lambda(total_points, cells_f64, false);
        lambda_sum += lambda_rep;
        let rep_p = format_p_value(p_value(rep_coll as f64, lambda_rep), args.pretty_p);
        if args.reps > 1 {
            eprintln!(
                "{rep_coll}\tp={rep_p}\tcombined: {tot}\tp={}",
                format_p_value(p_value(tot as f64, lambda_sum), args.pretty_p)
            );
        } else {
            eprintln!("{rep_coll}\tp={rep_p}");
        }
    }
    eprintln!("Test completed in {:.2} seconds", sw.lap());
    (tot, lambda_sum)
}
