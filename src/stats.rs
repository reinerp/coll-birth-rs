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

/// A TestU01-style two-sided adjusted *p*-value, kept in a form that survives the
/// approach to 1.
///
/// A left-tail anomaly (too few collisions) has *p*-value `1 − p_left` with
/// `p_left = Pr[X ≤ coll]` astronomically small. Storing that as a single `f64`
/// would round it to exactly `1.0` as soon as `p_left` drops below the machine
/// epsilon (~1.1·10⁻¹⁶), discarding the tail. We therefore keep `p_left` itself
/// in [`PValue::NearOne`] and defer the lossy `1 − ε` subtraction to formatting,
/// where it is only performed in the plain (non-pretty) rendering.
#[derive(Clone, Copy, Debug)]
pub enum PValue {
    /// The *p*-value, stored directly (near 0 or in the middle of the range; also
    /// carries `f64::NAN` when CDFLIB reports an error).
    Direct(f64),
    /// A near-1 *p*-value equal to `1 − eps`, with `eps = p_left` retained at full
    /// precision so that pretty mode can print it as `1 − eps`.
    NearOne(f64),
}

/// TestU01-style two-sided adjusted *p*-value for the Poisson distribution
/// (cf. Chapter 3 of the long TestU01 guide). Returns [`PValue::Direct`]`(f64::NAN)`
/// if CDFLIB reports an error.
pub fn p_value(coll: f64, lambda: f64) -> PValue {
    let Some(tails) = cdf::poisson_tails(coll, lambda) else {
        return PValue::Direct(f64::NAN);
    };
    if tails.p_right < tails.p_left {
        // Right-tail anomaly (too many collisions): the p-value is the small
        // p_right, representable directly.
        PValue::Direct(tails.p_right)
    } else if tails.p_left < 0.5 {
        // Left-tail anomaly (too few collisions): the p-value is 1 − p_left.
        // Keep p_left (the small quantity) rather than the cancelled difference.
        PValue::NearOne(tails.p_left)
    } else {
        PValue::Direct(0.5)
    }
}

/// Formats a *p*-value, optionally rendering values close to 1 as `1 − ε` for
/// readability (and at full precision, however small ε is).
pub fn format_p_value(p: PValue, pretty_p: bool) -> Cow<'static, str> {
    match p {
        PValue::Direct(v) => {
            if v == 0.0 {
                "0".into()
            } else if v == 1.0 {
                "1".into()
            } else {
                format!("{v:?}").into()
            }
        }
        // Value is 1 − eps. When eps underflowed to 0 the left tail is below the
        // f64 range, yet the count is still a left-tail anomaly: print a sentinel
        // just below 1 (`1 − 1e-307`, regardless of pretty mode) rather than a bare
        // "1", which would read as a perfect/degenerate value. Otherwise pretty
        // mode prints the tail exactly, and plain mode falls back to the decimal
        // 1 − eps (which may itself round to "1" for tiny eps).
        PValue::NearOne(eps) => {
            if eps == 0.0 {
                "1 − 1e-307".into()
            } else if pretty_p && eps < 1e-4 {
                format!("1 − {eps:?}").into()
            } else {
                let v = 1.0 - eps;
                if v == 1.0 {
                    "1".into()
                } else {
                    format!("{v:?}").into()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A count deep in the left tail (here X = 0 against mean 700) has
    // p_left = e^-700 ≈ 9.9e-305 — tiny but representable. Its two-sided p-value
    // is 1 - p_left, which must render in pretty mode as `1-<that tiny eps>`, not
    // collapse to "1": computing `1 - p_left` in an f64 rounds to exactly 1.0 once
    // p_left < machine epsilon, destroying the tail. This pins the precision.
    #[test]
    fn deep_left_tail_renders_as_one_minus_eps() {
        let s = format_p_value(p_value(0.0, 700.0), true);
        assert!(
            s.starts_with("1 − ") && s.len() > "1 − ".len(),
            "deep left tail must render as `1 − eps`, got {s:?}"
        );
    }

    // Two distinct deep-left-tail counts must produce distinct pretty strings:
    // the old `1 - p_left` collapse mapped both to "1".
    #[test]
    fn distinct_deep_tails_render_distinctly() {
        let a = format_p_value(p_value(0.0, 700.0), true);
        let b = format_p_value(p_value(0.0, 600.0), true);
        assert_ne!(a, b, "different left-tail depths must format differently");
    }

    // When the left tail underflows f64 to 0 (here e^-750 = 0), the p-value is 1
    // to machine precision but is still a left-tail anomaly: print the sentinel
    // `1 − 1e-307` (in both pretty and plain mode) rather than a bare "1".
    #[test]
    fn underflowed_left_tail_prints_sentinel() {
        assert_eq!(format_p_value(p_value(0.0, 750.0), true), "1 − 1e-307");
        assert_eq!(format_p_value(p_value(0.0, 750.0), false), "1 − 1e-307");
    }

    // A small right-tail p-value (too many collisions) is unaffected and prints
    // as a plain tiny decimal.
    #[test]
    fn right_tail_small_pvalue_is_plain() {
        let s = format_p_value(p_value(700.0, 1.0), false);
        assert!(
            !s.starts_with("1-") && s != "1",
            "right-tail anomaly must be a small p-value, got {s:?}"
        );
    }
}
