/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! The [`Cell`] trait abstracts over the integer types used to store cell indices
//! (`u32`, `u64`, `u128`) for the collision and birthday-spacings tests.

use voracious_radix_sort::{RadixSort, ska_sort};

use crate::prng::Prng;
use crate::util::parallelism;

/// Storage and operations required for a cell-index integer type.
pub trait Cell:
    Copy
    + Eq
    + Ord
    + Send
    + Sync
    + bytemuck::Pod
    + std::ops::ShlAssign<usize>
    + std::ops::ShrAssign<usize>
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitXorAssign
    + std::ops::Sub<Output = Self>
    + std::ops::SubAssign
    + std::ops::AddAssign
{
    const ZERO: Self;

    fn from_u64(x: u64) -> Self;
    fn from_u128(x: u128) -> Self;

    /// Mask of the low *b* bits, saturating to all-ones when *b* reaches the type width.
    /// Used by the birthday spacing-count level, which keys on the low bits of the
    /// spacing (balanced, unlike the top bits which cluster near zero).
    fn low_bits_mask(b: usize) -> Self;

    /// Sorts `v` in place using a multithreaded radix sort.
    fn sort_mt(v: &mut [Self]);

    /// Sorts `v` in place using a single-threaded radix sort.
    fn sort_st(v: &mut [Self]);
}

macro_rules! impl_cell {
    ($t:ty, $bits:literal) => {
        impl Cell for $t {
            const ZERO: Self = 0;

            #[inline(always)]
            fn from_u64(x: u64) -> Self {
                x as Self
            }
            #[inline(always)]
            fn from_u128(x: u128) -> Self {
                x as Self
            }
            #[inline(always)]
            fn low_bits_mask(b: usize) -> Self {
                if b >= $bits {
                    !0
                } else {
                    (1 as $t).wrapping_shl(b as u32).wrapping_sub(1)
                }
            }
            fn sort_mt(v: &mut [Self]) {
                v.voracious_mt_sort(parallelism());
            }
            fn sort_st(v: &mut [Self]) {
                // In-place American-flag (MSD) radix sort. Unlike voracious_sort,
                // which dispatches to the diverting-LSD dlsd_radixsort and allocates
                // a full size-n scratch buffer, ska_sort permutes within v using
                // only O(radix-range) bookkeeping. This matters in parallel tradeoff
                // mode, where num_cpus threads each sort concurrently: an out-of-place
                // sort would double peak RSS (mmap buffers + per-thread scratch).
                ska_sort(v, 8);
            }
        }
    };
}

impl_cell!(u32, 32);
impl_cell!(u64, 64);
impl_cell!(u128, 128);

/// Draws a single (non-decimated) cell index from `prng`.
///
/// The index concatenates *t* chunks of *u* bits taken from the top of
/// [`next_u64()`] (after a left shift by *s*), combined left-to-right with
/// XOR-shifts. Decimation is handled separately by [`decimate_once`].
///
/// Const generics select hot-loop specializations (see the design spec):
/// - `DIM`: the dimension *t*. `DIM > 0` makes the trip count a compile-time
///   constant so the draw loop unrolls; `DIM == 0` falls back to the runtime
///   *t* argument.
/// - `FULL`: when true (*u* = 64 and *s* = 0) the extraction is a
///   compile-time identity.
///
/// [`next_u64()`]: crate::prng::Prng::next_u64
#[inline]
pub fn cell_index<T: Cell, const DIM: usize, const FULL: bool>(
    prng: &mut Prng,
    t_rt: usize,
    u: usize,
    s: usize,
) -> T {
    let combined_shift = 64 - u - s;
    let extract_mask: u64 = if u >= 64 { !0 } else { (1u64 << u) - 1 };

    #[inline(always)]
    fn extract<const FULL: bool>(raw: u64, combined_shift: usize, extract_mask: u64) -> u64 {
        if FULL {
            raw
        } else {
            (raw >> combined_shift) & extract_mask
        }
    }

    let t = if DIM == 0 { t_rt } else { DIM };

    let mut x = T::from_u64(extract::<FULL>(
        prng.next_u64(),
        combined_shift,
        extract_mask,
    ));
    for _ in 1..t {
        x <<= u;
        x ^= T::from_u64(extract::<FULL>(
            prng.next_u64(),
            combined_shift,
            extract_mask,
        ));
    }
    x
}

/// One decimating attempt: draws exactly *t* PRNG outputs (a full candidate
/// tuple) and returns `Some(x)`, with `x` the assembled dense index (each element
/// compacted to *u* − *d* bits), iff every element's low *d* bits are zero; else
/// `None`. The full tuple's *t* draws are always consumed, accepted or rejected,
/// so the generator advances by exactly *t* and sample *j* sits at orbit offset
/// *j* · *t*. This is the fixed-sample counterpart to the loop-until-accept path:
/// the caller scans a fixed number of samples and keeps the `Some` values.
#[inline]
pub fn decimate_once<T: Cell, const DIM: usize, const FULL: bool>(
    prng: &mut Prng,
    t_rt: usize,
    u: usize,
    s: usize,
    d: usize,
) -> Option<T> {
    let combined_shift = 64 - u - s;
    let extract_mask: u64 = if u >= 64 { !0 } else { (1u64 << u) - 1 };

    #[inline(always)]
    fn extract<const FULL: bool>(raw: u64, combined_shift: usize, extract_mask: u64) -> u64 {
        if FULL {
            raw
        } else {
            (raw >> combined_shift) & extract_mask
        }
    }

    let t = if DIM == 0 { t_rt } else { DIM };
    let dec_mask: u64 = (1u64 << d) - 1;
    let width = u - d;

    let first = extract::<FULL>(prng.next_u64(), combined_shift, extract_mask);
    let mut rejected = first & dec_mask != 0;
    let mut x = T::from_u64(first >> d);
    for _ in 1..t {
        let raw = extract::<FULL>(prng.next_u64(), combined_shift, extract_mask);
        rejected |= raw & dec_mask != 0;
        x <<= width;
        x ^= T::from_u64(raw >> d);
    }
    if rejected { None } else { Some(x) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::Prng;

    // decimate_once consumes exactly t draws per call (accepted or rejected),
    // returns Some only when every element's low d bits are zero, and the dense
    // index is in [0, (2ᵘ⁻ᵈ)ᵗ). Both arms must be exercise, which needs a
    // non-degenerate generator (the incr counter maps everything to ~cell 0, so it
    // never rejects), hence the gate.
    #[cfg(not(feature = "incr"))]
    #[test]
    fn decimate_once_consumes_t_draws_and_is_dense() {
        let (t, u, s, d) = (3usize, 12usize, 0usize, 3usize);
        let bound = 1u128 << ((u - d) * t);
        let mut a = Prng::new(0xABCD_1234);
        let (mut accepted, mut rejected) = (0usize, 0usize);
        for _ in 0..50_000 {
            let b = a; // Copy: capture position before the attempt
            let got: Option<u128> = decimate_once::<u128, 3, false>(&mut a, 0, u, s, d);
            // A reference advanced by exactly t draws must be at the same position:
            // compare the next outputs of throwaway clones (does not disturb a).
            let mut aref = b;
            for _ in 0..t {
                aref.next_u64();
            }
            let (mut a2, mut b2) = (a, aref);
            for _ in 0..4 {
                assert_eq!(
                    a2.next_u64(),
                    b2.next_u64(),
                    "decimate_once must consume exactly t draws"
                );
            }
            match got {
                Some(x) => {
                    assert!(x < bound, "dense index {x} not below {bound}");
                    accepted += 1;
                }
                None => rejected += 1,
            }
        }
        assert!(
            accepted > 0 && rejected > 0,
            "expected both accepts and rejects (acc={accepted}, rej={rejected})"
        );
    }

    // (Decimation density is covered by decimate_once_consumes_t_draws_and_is_dense.)

    #[test]
    fn dim_specialization_matches_runtime_fallback() {
        let (u, s) = (12usize, 0usize);
        let mut a = Prng::new(12_345);
        let mut b = a; // Prng: Copy → identical stream
        for _ in 0..2_000 {
            let spec: u64 = cell_index::<u64, 3, false>(&mut a, 0, u, s);
            let rt: u64 = cell_index::<u64, 0, false>(&mut b, 3, u, s);
            assert_eq!(spec, rt);
        }
    }

    #[test]
    fn full_matches_shift_when_u_is_64() {
        let (u, s) = (64usize, 0usize);
        let mut a = Prng::new(999);
        let mut b = a;
        for _ in 0..2_000 {
            let full: u64 = cell_index::<u64, 1, true>(&mut a, 1, u, s);
            let shift: u64 = cell_index::<u64, 1, false>(&mut b, 1, u, s);
            assert_eq!(full, shift);
        }
    }
}
