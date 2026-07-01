/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Utilities.

use std::fmt::Display;
use std::time::Instant;

/// Renders an integer as a Unicode superscript string, so exponents such as 2⁶⁴
/// can be printed inline; e.g. `superscript(64)` returns `"⁶⁴"`.
///
/// The value is taken through [`Display`], so every integer type works without a
/// cast. Decimal digits map to their superscript code points and `-` to the
/// superscript minus `⁻`; any other character is passed through unchanged.
pub fn superscript(n: impl Display) -> String {
    n.to_string()
        .chars()
        .map(|c| match c {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            '-' => '⁻',
            other => other,
        })
        .collect()
}

/// Number of threads in the Rayon global thread pool.
///
/// This is the single thread-count source for every parallel phase—generation
/// fan-out, sorting, and counting—so all of them honour `RAYON_NUM_THREADS`
/// (which itself defaults to the number of available cores).
pub fn parallelism() -> usize {
    rayon::current_num_threads()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superscript() -> anyhow::Result<()> {
        assert_eq!(superscript(0u32), "⁰");
        assert_eq!(superscript(64usize), "⁶⁴");
        assert_eq!(superscript(1234567890u64), "¹²³⁴⁵⁶⁷⁸⁹⁰");
        assert_eq!(superscript(-5i32), "⁻⁵");
        Ok(())
    }
}
