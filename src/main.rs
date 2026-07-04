/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Command-line entry point.

use clap::Parser;
use num::BigUint;
use num::traits::ToPrimitive;

use coll_birth::birthday::run_birthday_parallel;
use coll_birth::cli::Args;
use coll_birth::collision::run_test_parallel;
use coll_birth::common::{compute_lambda_and_points, run_test, test_lambda};
use coll_birth::prng::Prng;
use coll_birth::stats::{format_p_value, p_value};

/// Picks the smallest cell integer type that can hold every stored value.
///
/// Cell indices are below `cells`; birthday spacings additionally store the
/// wrap-around `cells - max + min`, which is evaluated through the
/// representable `cells - 1` and fits the same width (the degenerate `min ==
/// max` case, where it would equal `cells`, is dropped without affecting the
/// collision count—see `compute_spacings`). Both tests therefore use `u32` when
/// `cells` ≤ 2³², `u64` when ≤ 2⁶⁴, and `u128` otherwise.
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

fn main() {
    let args = Args::parse();
    args.validate();

    eprintln!("Generator: {}", Prng::NAME);

    // Clear any inherited per-process THP disable (prctl PR_SET_THP_DISABLE,
    // set by some sandboxes/harnesses and preserved across fork+exec). With it
    // set, MADV_HUGEPAGE is ignored, every large buffer is base-paged, and the
    // generate/sort phases become page-fault-bound: 4 KiB faults through a
    // contended mmap_lock cost more than the work itself (measured 6x on the
    // generation phase at 360 threads).
    #[cfg(target_os = "linux")]
    unsafe {
        const PR_SET_THP_DISABLE: libc::c_int = 41;
        libc::prctl(PR_SET_THP_DISABLE, 0u64, 0u64, 0u64, 0u64);
    }

    // Report the kernel's transparent-huge-page policy once: if it reads "[never]",
    // MADV_HUGEPAGE is ignored system-wide and the large buffers stay base-paged
    // regardless of what alloc_mmap requests (a separate, system-level cause).
    // Also report the per-process flag so an environment that re-disables THP
    // (or a kernel that rejects the prctl) is visible in the log.
    #[cfg(target_os = "linux")]
    {
        if let Ok(thp) = std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled") {
            eprint!("Transparent huge pages: {}", thp.trim());
        }
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            if let Some(line) = status.lines().find(|l| l.starts_with("THP_enabled")) {
                eprint!(" (process: {})", line.split_whitespace().nth(1).unwrap_or("?"));
            }
        }
        eprintln!();
    }

    let cells = BigUint::from(2u32)
        .pow((args.u - args.decimation_bits.unwrap_or(0)) as _)
        .pow(args.t as u32);
    if cells > BigUint::from(2u32).pow(128) {
        Args::die("you cannot have more than 2¹²⁸ cells");
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
        // lambda share (lambda_total / 2ᵇ, summed over repetitions).
        let num_passes = 1u64 << args.tradeoff_bits();
        let lambda_k = args.reps as f64
            * test_lambda(points, cells.to_f64().unwrap(), args.birthday_spacings)
            / num_passes as f64;
        eprintln!(
            "Single pass {k} of {num_passes}: recombine the 2ᵇ runs via p_value(Σ counts, Σ lambdas)"
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
