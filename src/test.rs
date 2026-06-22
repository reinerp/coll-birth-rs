/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Variants of the collision and birthday-spacings tests.

use num::BigUint;
use num::traits::ToPrimitive;
use rayon::prelude::*;

use crate::cell::{Cell, cell_index, decimate_once};
use crate::prng::Prng;
use crate::stats::{expected_collisions, format_p_value, p_value};
use crate::util::{Stopwatch, parallelism};

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
    let scan_len = crate::scan_samples(points, t, d);

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
                    .unwrap_or_else(|| crate::bin_overflow("a collision tradeoff pass")) = x;
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
    let scan_len = crate::scan_samples(points, t, d);

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
                    .unwrap_or_else(|| crate::bin_overflow("a decimated collision run")) = x;
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
    let aux_cap = crate::buffer_size(scan_len.div_ceil(num_checkpoints), t * d).max(1);
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
                    .unwrap_or_else(|| crate::bin_overflow("a decimation checkpoint stage")) = x;
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
            crate::bin_overflow("the decimation checkpoint accumulator");
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

/// Runs a birthday-spacings test.
///
/// With decimation (`DECIMATE`), this uses the same fixed-sample model as the
/// collision tests: scan `points · 2^(t·d)` candidate tuples and keep the
/// ~`points` accepted (dense) indices. Without decimation it fills exactly
/// `points` cells. The parallel counterpart is `run_birthday_parallel` in
/// `main.rs`, which is bit-identical to this runner for every CPU count.
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
        let scan_len = crate::scan_samples(points, params.t, params.d);
        let mut len = 0usize;
        for _ in 0..scan_len {
            if let Some(x) = params.draw_decimate_once::<T, DIM, FULL>(prng) {
                *buf.get_mut(len)
                    .unwrap_or_else(|| crate::bin_overflow("a decimated birthday run")) = x;
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
/// `predecessor` is the largest value strictly below this slice — the previous
/// value interval maximum — so the first element receives `v[0] - predecessor`.
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
/// The wrap-around lies in `[1..=cells]` — one value beyond the cell indices — and
/// reaches `cells` exactly when `min == max`, which may not be representable
/// (`cells` can be 2^N in N-bit storage). It is therefore evaluated through the
/// always-representable `cells - 1`, and in the degenerate case replaced by a
/// nonzero stand-in: all other spacings are then zero, so the collision count is
/// unaffected.
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
/// - **Outer (counting) level:** spacings are classified by their *low* *b* bits,
///   not their top bits — spacings cluster near zero (≈ exponential), so a top-bit
///   split would dump almost everything into one class, whereas the low bits are
///   balanced. Only the current class's spacings are kept, sorted, and counted;
///   equal spacings share all bits hence the same class, so the per-class counts
///   sum to the exact total while only ~`points` / 2*ᵇ* spacings (and one
///   interval's ~`points` / 2*ᵇ* points) are ever resident.
///
/// The point multiset is identical to a single sweep, so the total equals the
/// plain [`run_birthday`] count.
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
    let elem_width = if DECIMATE { u - d } else { u };
    let point_key_shift = t * elem_width - b; // top b bits select the value interval
    let spacing_mask = T::low_bits_mask(b); // low b bits select the spacing class
    let scan_len = crate::scan_samples(points, t, d);
    // cells itself may be unrepresentable (2^N in N-bit storage); cells − 1 always
    // is, and the wrap-around spacing is evaluated through it.
    let cells_m1 = T::from_u128((params.cells - BigUint::from(1u8)).to_u128().unwrap());

    // One value interval keeps ~points / 2^b points (with balls-into-bins headroom
    // over the t·d + b selectivity bits).
    let mut scratch: Vec<T> = vec![T::ZERO; crate::buffer_size(scan_len, t * d + b)];

    let snapshot = *prng;
    let mut end_state = snapshot;
    let mut total_coll = 0usize;
    let mut total_points = 0usize;
    let mut sw = Stopwatch::new();
    eprintln!("Birthday tradeoff over {num_passes} spacing-classes");
    // Per-class progress heartbeat (collision-style): each spacing-class sweeps all
    // 2^b value-intervals, so without this the run is silent for the whole sweep.
    let mut class_sw = Stopwatch::new();
    let cells_f64 = params.cells.to_f64().unwrap();
    // Nominal per-class Poisson mean (lambda_total / 2^b) for the progress p-values.
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
                        .unwrap_or_else(|| crate::bin_overflow("a birthday value interval")) = x;
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
                        .unwrap_or_else(|| crate::bin_overflow("a birthday-spacings class")) = s;
                    class_len += 1;
                }
            }
            eprintln!("[{:.3}s], {len} points", isw.lap());
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
                    *class_buf
                        .get_mut(class_len)
                        .unwrap_or_else(|| crate::bin_overflow("a birthday-spacings class")) = wrap;
                    class_len += 1;
                }
            }
        }

        T::sort_st(&mut class_buf[..class_len]);
        let class_coll = count_adjacent_equals(&class_buf[..class_len]);
        total_coll += class_coll;
        let classes_done = (j - pass_lo + 1) as f64;
        eprintln!(
            "  Class {}/{} done: [{:.3}s], {class_len} spacings, {class_coll} collisions, p={}; combined: {total_coll} collisions, p={}",
            j + 1,
            num_passes,
            class_sw.lap(),
            format_p_value(p_value(class_coll as f64, lambda_class), pretty_p),
            format_p_value(
                p_value(total_coll as f64, classes_done * lambda_class),
                pretty_p
            ),
        );
    }
    *prng = end_state;
    eprintln!("[{:.3}s] done.", sw.lap());
    (total_coll, total_points)
}

// These integration tests need a reasonably uniform PRNG: the tradeoff buffer is
// sized assuming points spread ~evenly over the 2^(t·b) passes. A degenerate
// generator (e.g. incr, which maps every point to cell 0) would overflow it, so
// the module is gated to splitmix.
#[cfg(all(test, feature = "splitmix"))]
mod tests {
    use super::*;
    use crate::prng::Prng;
    use num::BigUint;

    fn grid(u: usize, t: usize, d: usize, cells: &BigUint) -> GridParams<'_> {
        GridParams {
            u,
            t,
            s: 0,
            d,
            cells,
        }
    }

    // Summing collisions over 2^(t·b) tradeoff passes equals a single run on the
    // same point stream (decimation off).
    #[test]
    fn tradeoff_sum_equals_single_run_no_decimation() {
        let (u, t, b, d) = (8usize, 2usize, 4usize, 0usize);
        let cells = BigUint::from(1u128 << (u * t)); // 2^16
        let g = grid(u, t, d, &cells);

        let passes = 1usize << b; // 2^4 = 16
        let m = 1_500usize;
        let points = m * passes; // 24_000 > sqrt(2^16) → collisions occur

        let start = Prng::new(2024);

        let mut single = start;
        let mut buf_single = vec![0u64; points];
        let c_single = run_collision::<u64, 0, false>(&mut single, &g, &mut buf_single);

        let mut traded = start;
        let cap = crate::buffer_size(points, b);
        let mut buf_traded = vec![0u64; cap];
        let cells_per_pass = cells.to_f64().unwrap() / passes as f64;
        let c_traded = run_collision_tradeoff::<u64, 0, false>(
            &mut traded,
            &g,
            &mut buf_traded,
            points,
            b,
            cells_per_pass,
            false,
            None,
        );

        assert_eq!(
            c_single, c_traded,
            "tradeoff passes must sum to the single-run count"
        );
    }

    // Same exactness property with decimation on (d > 0): both runs decimate the
    // same stream, so kept points and their collisions match.
    #[test]
    fn tradeoff_sum_equals_single_run_with_decimation() {
        let (u, t, b, d) = (10usize, 2usize, 4usize, 2usize);
        // effective cell space after decimation: 2^((u-d)*t) = 2^16
        let cells = BigUint::from(1u128 << ((u - d) * t));
        let g = grid(u, t, d, &cells);

        let passes = 1usize << b;
        let m = 1_500usize;
        let points = m * passes;

        let start = Prng::new(7);

        // Fixed-sample: both runs scan scan_len = points · 2^(t·d) samples; the
        // kept count is variable, so buffers carry balls-into-bins headroom.
        let scan_len = points << (t * d);

        let mut single = start;
        let mut buf_single = vec![0u64; crate::buffer_size(scan_len, t * d)];
        let c_single = run_collision_decimate::<u64, 0, false>(
            &mut single,
            &g,
            &mut buf_single,
            points,
            cells.to_f64().unwrap(),
            false,
            false,
        );

        let mut traded = start;
        let cap = crate::buffer_size(scan_len, t * d + b);
        let mut buf_traded = vec![0u64; cap];
        let cells_per_pass = cells.to_f64().unwrap() / passes as f64;
        let c_traded = run_collision_tradeoff::<u64, 0, true>(
            &mut traded,
            &g,
            &mut buf_traded,
            points,
            b,
            cells_per_pass,
            false,
            None,
        );

        assert_eq!(c_single, c_traded);
    }

    // The two-level birthday tradeoff visits the same point multiset as a single
    // sweep, so its summed per-class spacing-collision count equals plain birthday.
    #[test]
    fn birthday_tradeoff_matches_plain_no_decimation() {
        let (u, t, b, d) = (8usize, 2usize, 4usize, 0usize);
        let cells = BigUint::from(1u128 << (u * t)); // 2^16
        let g = grid(u, t, d, &cells);
        let points = 4_000usize;
        let start = Prng::new(12_345);

        let mut plain = start;
        let mut buf_plain = vec![0u64; points];
        let c_plain = run_birthday::<u64, 0, false, false>(&mut plain, &g, &mut buf_plain, points);

        let mut traded = start;
        let mut class_buf = vec![0u64; crate::buffer_size(points, b)];
        let c_traded = run_birthday_tradeoff::<u64, 0, false, false>(
            &mut traded,
            &g,
            &mut class_buf,
            points,
            b,
            false,
            None,
        );
        assert_eq!(
            c_plain, c_traded,
            "birthday tradeoff must match the plain birthday count"
        );
    }

    // The wrap-around is computed via cells − 1, so an N-bit type works even when
    // cells == 2^N exactly (no strictly-wider type needed). At the 2^32 boundary
    // u32 and u64 storage must produce identical counts.
    #[test]
    fn birthday_width_boundary_u32_matches_u64() {
        let (u, t) = (16usize, 2usize);
        let cells = BigUint::from(1u128 << 32); // exactly 2^32
        let g = grid(u, t, 0, &cells);
        let points = 50_000usize; // λ = n³/4k ≈ 7276, so collisions occur

        let mut p32 = Prng::new(42);
        let mut buf32 = vec![0u32; points];
        let c32 = run_birthday::<u32, 0, false, false>(&mut p32, &g, &mut buf32, points);

        let mut p64 = Prng::new(42);
        let mut buf64 = vec![0u64; points];
        let c64 = run_birthday::<u64, 0, false, false>(&mut p64, &g, &mut buf64, points);

        assert!(c32.0 > 0, "boundary test should observe spacing collisions");
        assert_eq!(c32, c64, "u32 and u64 storage must agree at cells == 2^32");
    }

    // cells == 2^128 in u128 storage: the largest possible grid. The tradeoff
    // runner used to panic materializing cells itself; both runners must now
    // complete and agree (the counts are ~0 at this density — the value of the
    // test is exercising the wrap arithmetic at the type limit).
    #[test]
    fn birthday_2_pow_128_cells_tradeoff_matches_plain() {
        let (u, t, b) = (64usize, 2usize, 2usize);
        let cells = BigUint::from(1u8) << 128;
        let g = grid(u, t, 0, &cells);
        let points = 4_000usize;
        let start = Prng::new(7);

        let mut plain = start;
        let mut buf_plain = vec![0u128; points];
        let c_plain = run_birthday::<u128, 0, false, true>(&mut plain, &g, &mut buf_plain, points);

        let mut traded = start;
        let mut class_buf = vec![0u128; crate::buffer_size(points, b)];
        let c_traded = run_birthday_tradeoff::<u128, 0, false, true>(
            &mut traded,
            &g,
            &mut class_buf,
            points,
            b,
            false,
            None,
        );
        assert_eq!(c_plain, c_traded);
    }

    // Same property with decimation on: both runs decimate the same stream, so the
    // accepted points (and therefore the spacings) match.
    #[test]
    fn birthday_tradeoff_matches_plain_with_decimation() {
        let (u, t, b, d) = (10usize, 2usize, 4usize, 2usize);
        let cells = BigUint::from(1u128 << ((u - d) * t)); // 2^16 effective
        let g = grid(u, t, d, &cells);
        let points = 4_000usize;
        let scan_len = points << (t * d);
        let start = Prng::new(99);

        let mut plain = start;
        let mut buf_plain = vec![0u64; crate::buffer_size(scan_len, t * d)];
        let c_plain = run_birthday::<u64, 0, true, false>(&mut plain, &g, &mut buf_plain, points);

        let mut traded = start;
        let mut class_buf = vec![0u64; crate::buffer_size(points, b)];
        let c_traded = run_birthday_tradeoff::<u64, 0, true, false>(
            &mut traded,
            &g,
            &mut class_buf,
            points,
            b,
            false,
            None,
        );
        assert_eq!(c_plain, c_traded);
    }
}

// Direct tests of the wrap-around arithmetic in compute_spacings at the
// cells == 2^N storage boundary; no PRNG involved, so no feature gate.
#[cfg(test)]
mod spacing_tests {
    use super::*;
    use num::BigUint;

    // Non-degenerate at the boundary: sorted points {3, 10, 2^64 − 1} on a circle
    // of 2^64 cells have spacings {7, 2^64 − 11} and wrap 2^64 − (2^64 − 1) + 3 = 4,
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
    // would equal cells == 2^64. It is replaced by a nonzero stand-in; the n − 1
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
