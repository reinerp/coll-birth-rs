/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Integration tests for the collision runners' public API.
//!
//! These need a reasonably uniform PRNG: the tradeoff buffer is sized assuming
//! points spread ~evenly over the 2^(t·b) passes. A degenerate generator (e.g.
//! incr, which maps every point to cell 0) would overflow it, so the file is
//! gated to splitmix.
#![cfg(feature = "splitmix")]

use num::BigUint;
use num::traits::ToPrimitive;

use coll_birth::collision::{run_collision, run_collision_decimate, run_collision_tradeoff};
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
    let cap = buffer_size(points, b);
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
    let mut buf_single = vec![0u64; buffer_size(scan_len, t * d)];
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
    let cap = buffer_size(scan_len, t * d + b);
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
