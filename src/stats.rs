/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Statistical helpers: expected collision count, adjusted *p*-values, and formatting.

use std::borrow::Cow;

use crate::cdf;

/// Expected number of collisions when throwing `points` balls into `cells` bins.
///
/// For sparse regimes (`points / cells <= 0.1`) the routine evaluates a truncated Maclaurin
/// expansion converging in at most 64 terms; otherwise it falls back to the closed form
/// `points - cells + cells * (1 - 1/cells)^points`, with `(1 - 1/cells)^points` evaluated as
/// `exp(points * log(1 - 1/cells))` and `log(1 - 1/cells)` itself expanded as a Maclaurin
/// series for accuracy when `cells` is large.
pub fn expected_collisions(points: f64, cells: f64) -> f64 {
    // Fewer than two points cannot collide; returning early also avoids a 0/0 NaN
    // in the sparse series below (a tradeoff/decimation pass may keep 0 or 1 point).
    if points <= 1.0 {
        return 0.0;
    }
    if points / cells <= 0.1 {
        // Sparse regime: lambda = sum_{i>=2} (-1)^i * C(points, i) * cells^(1-i).
        let mut u = points - 1.0;
        let mut v = 2.0;
        let mut t = (points * u) / (2.0 * cells);
        let mut lambda = t;

        let mut i = 3;
        while (t / lambda).abs() > f64::EPSILON && i < 64 {
            u -= 1.0;
            v += 1.0;
            t = -t * u / (cells * v);
            lambda += t;
            i += 1;
        }
        debug_assert!((t / lambda).abs() <= f64::EPSILON);
        lambda
    } else {
        // Dense regime: lambda = points - cells + cells * (1 - 1/cells)^points, with the
        // power evaluated as exp(-points · neg_log) where neg_log = -log(1 - 1/cells) is
        // computed by its Maclaurin series for numerical stability when cells is large.
        let mut t = 1.0 / cells;
        let mut neg_log = t;
        for i in 2..10 {
            t *= 1.0 / cells;
            neg_log += t / i as f64;
        }
        (points - cells) + cells * f64::exp(-(points * neg_log))
    }
}

/// TestU01-style two-sided adjusted *p*-value for the Poisson distribution
/// (cf. Chapter 3 of the long TestU01 guide). Returns `f64::NAN` if CDFLIB reports an error.
pub fn p_value(coll: f64, lambda: f64) -> f64 {
    let Some(tails) = cdf::poisson_tails(coll, lambda) else {
        return f64::NAN;
    };
    if tails.p_right < tails.p_left {
        tails.p_right
    } else if tails.p_left < 0.5 {
        1.0 - tails.p_left
    } else {
        0.5
    }
}

/// Formats a *p*-value, optionally rendering values close to 1 as 1 – ε for
/// readability.
pub fn format_p_value(mut p: f64, pretty_p: bool) -> Cow<'static, str> {
    if p == 0.0 {
        "0".into()
    } else if p == 1.0 {
        "1".into()
    } else {
        let mut prefix = "";
        if pretty_p && 1.0 - p < 1e-4 {
            p = 1.0 - p;
            prefix = "1-";
        }
        format!("{}{:?}", prefix, p).into()
    }
}
