/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Safe wrapper around the Poisson CDF routine from the [`cdflib`] crate.
//!
//! Adjusted *p*-values and the expected-collision mean are implemented on top in
//! [`crate::stats`].

use cdflib::Poisson;
use cdflib::traits::DiscreteCdf;

/// Lower and upper Poisson tail probabilities at a given count.
#[derive(Clone, Copy, Debug)]
pub struct PoissonTails {
    /// `Pr[X <= coll]`
    pub p_left: f64,
    /// `Pr[X >= coll]`
    pub p_right: f64,
}

/// Computes the Poisson lower and upper tail probabilities at `coll` given mean `lambda`.
///
/// Returns `None` when `lambda` is not a valid Poisson rate or `coll` is not a
/// non-negative integer representable as `u64`.
///
/// # Implementation notes
///
/// [`DiscreteCdf::ccdf`] returns `Pr[X > s]`; computing `Pr[X >= coll]` therefore
/// evaluates the complementary CDF at `coll - 1`. The `coll == 0` case is
/// special-cased to avoid underflowing the `u64` argument.
pub fn poisson_tails(coll: f64, lambda: f64) -> Option<PoissonTails> {
    if !coll.is_finite() || coll < 0.0 || coll.fract() != 0.0 {
        return None;
    }
    let coll_u = coll as u64;

    // `Poisson::new` panics on an invalid rate (negative or non-finite); the
    // fallible constructor lets us honour the documented `None` contract instead.
    let poi = Poisson::try_new(lambda).ok()?;
    let p_left = poi.cdf(coll_u);
    let p_right = if coll_u == 0 {
        1.0
    } else {
        poi.ccdf(coll_u - 1)
    };

    Some(PoissonTails { p_left, p_right })
}
