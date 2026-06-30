/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Integration tests for the birthday-spacings runners' public API. Gated to
//! splitmix for the same buffer-uniformity reason as the collision tests.
#![cfg(feature = "splitmix")]

use num::BigUint;

use coll_birth::birthday::{run_birthday, run_birthday_tradeoff};
use coll_birth::common::{GridParams, buffer_size};
use coll_birth::prng::Prng;

fn grid(u: usize, t: usize, d: usize, cells: &BigUint) -> GridParams<'_> {
    GridParams {
        u,
        t,
        s: 0,
        d,
        cells,
    }
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
    let mut class_buf = vec![0u64; buffer_size(points, b)];
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
// complete and agree (the counts are ~0 at this density; the value of the
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
    let mut class_buf = vec![0u128; buffer_size(points, b)];
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
    let mut buf_plain = vec![0u64; buffer_size(scan_len, t * d)];
    let c_plain = run_birthday::<u64, 0, true, false>(&mut plain, &g, &mut buf_plain, points);

    let mut traded = start;
    let mut class_buf = vec![0u64; buffer_size(points, b)];
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
