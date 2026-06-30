/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Faithfulness of the parallel runners (and the single-pass recombination)
//! against the sequential runner.
//!
//! Gated away from the degenerate incr counter: these compare a parallel run to
//! the sequential one in tradeoff/decimation modes, whose buffer headroom assumes
//! ~uniform spread across residue bins. A counter that maps every sample to ~one
//! cell overflows that headroom. Every real generator (splitmix, wyrand,
//! MSWS-CTR, LCG, MWC, Romu) exercises both the jump and pre-scan snapshot paths.
#![cfg(not(feature = "incr"))]

use num::BigUint;
use num::traits::ToPrimitive;

use coll_birth::birthday::run_birthday_parallel;
use coll_birth::cli::Args;
use coll_birth::collision::run_test_parallel;
use coll_birth::common::{compute_lambda_and_points, run_test, test_lambda};

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
        seed,
        pretty_p: false,
        parallel: Some(0),
        pass: None,
    }
}

fn cells_for(args: &Args) -> BigUint {
    BigUint::from(2u32).pow(args.u as u32).pow(args.t as u32)
}

// Single-pass (--pass K) runs one of the 2ᵇ summable units; the per-unit
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

    // The per-pass nominal lambda shares (lambda_total / 2ᵇ) sum back to the
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

// Parallel birthday-spacings (plain, b = 0) must equal the sequential run for
// any CPU count: the gathered interval is the same point multiset, so the
// spacings and their collisions match.
#[test]
fn faithful_birthday_plain_matches_sequential() {
    let seed = 0x0B17_4DA9_0000_0001;
    let mut args = make_args(20, 2, 40_000, None, seed);
    args.birthday_spacings = true;
    let cells = cells_for(&args); // 2⁴⁰
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par1 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 1);
    let par3 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 3);
    assert_eq!(seq, par1, "P=1 parallel birthday must equal sequential");
    assert_eq!(seq, par3, "P=3 parallel birthday must equal sequential");
}

// Regression: parallel birthday-spacings with decimation and no tradeoff (b = 0).
// Under decimation the kept count (= number of spacings, all in the single class
// when b = 0) is random with mean `points`; with this seed it lands at 40196 >
// 40000, which overflowed the old zero-headroom `buffer_size(points, 0)` class
// buffer. The class buffer must carry the full t·d + b headroom, so the run
// completes and still matches the sequential one.
#[test]
fn faithful_birthday_decimation_matches_sequential() {
    let seed = 3;
    let mut args = make_args(30, 2, 40_000, None, seed);
    args.birthday_spacings = true;
    args.decimate = Some(2);
    let cells = BigUint::from(2u32)
        .pow((args.u - 2) as u32)
        .pow(args.t as u32); // effective cells 2^((u−d)·t)
    let (lambda, points) = compute_lambda_and_points(&args, &cells);
    let seq = run_test::<u64>(&args, points, &cells, lambda);
    let par1 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 1);
    let par3 = run_birthday_parallel::<u64>(&args, points, &cells, lambda, 3);
    assert_eq!(
        seq, par1,
        "P=1 parallel birthday decimation must equal sequential"
    );
    assert_eq!(
        seq, par3,
        "P=3 parallel birthday decimation must equal sequential"
    );
}

// Same, with the two-level top-bit tradeoff (b > 0).
#[test]
fn faithful_birthday_tradeoff_matches_sequential() {
    let seed = 0x0B17_4DA9_0000_0002;
    let mut args = make_args(20, 2, 10_000, Some(2), seed);
    args.birthday_spacings = true;
    let cells = cells_for(&args); // 2⁴⁰
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
    let points = 50_000usize; // m · 2ᵇ
    let lambda = (points as f64).powi(3) / (4.0 * cells.to_f64().unwrap());
    let seq = run_test::<u32>(&args, points, &cells, lambda);
    let par3 = run_birthday_parallel::<u32>(&args, points, &cells, lambda, 3);
    assert_eq!(
        seq, par3,
        "boundary-width parallel birthday must equal sequential"
    );
}
