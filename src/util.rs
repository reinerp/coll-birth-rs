/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Utilities.

use std::num::NonZeroUsize;
use std::thread::available_parallelism;
use std::time::Instant;

/// Number of usable threads, defaulting to 1 if [`available_parallelism`] fails.
pub fn parallelism() -> usize {
    available_parallelism()
        .unwrap_or(NonZeroUsize::new(1).unwrap())
        .into()
}

/// Records elapsed time between successive [`Stopwatch::lap`] calls.
///
/// Used to print progress lines of the form `... [X.XXXs] next-step...`.
pub struct Stopwatch(Instant);

impl Stopwatch {
    pub fn new() -> Self {
        Self(Instant::now())
    }

    /// Seconds elapsed since the previous lap (or since construction), resetting the clock.
    pub fn lap(&mut self) -> f64 {
        let elapsed = self.0.elapsed().as_secs_f64();
        self.0 = Instant::now();
        elapsed
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::new()
    }
}
