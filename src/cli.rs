/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Command-line argument definitions.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Runs a collision or birthday-spacings test. Please use RAYON_NUM_THREADS to customize the number of threads.",
    next_line_help = true,
    max_term_width = 100
)]
pub struct Args {
    /// Number of upper bits used, that is, log₂ of the number of subdivisions per dimension.​
    pub u: usize,

    /// The dimension of the test; the cell count is (2ᵘ ⁻ ᵈ)ᵗ.​
    pub t: usize,

    /// Number of memory locations. The number of points is m · 2ᵇ (approximate
    /// when decimating), the number of samples is m · 2ᵇ · 2ᵗᵈ, and the number
    /// of calls to the generator is t · m · (2ᵇ)² · 2ᵗᵈ (the birthday-spacings
    /// test adds a further factor of 2ᵇ for its second level).​
    pub m: Option<usize>,

    /// Left-shift the PRNG output by this many bits before extracting cell indices.​
    #[arg(short, long = "shift", default_value_t = 0)]
    pub s: usize,

    /// Use a time-space tradeoff on the top b bits of the combined index: uses 2ᵇ
    /// times less space, slower by (2ᵇ)² for collisions and (2ᵇ)³ for birthdays.​
    #[arg(short = 'b', long, value_name = "b")]
    pub tradeoff_bits: Option<usize>,

    /// Use decimation to keep only tuples in which every coordinate's lowest d
    /// bits are zero; a fixed number m · 2ᵇ · 2ᵗᵈ samples are scanned, keeping
    /// ≈ m · 2ᵇ.​
    #[arg(short = 'd', long, value_name = "d")]
    pub decimation_bits: Option<usize>,

    /// Print progressive p-values at ⌊√(2ᵗᵈ)⌋ uniform checkpoints; works with -P.​
    #[arg(
        short = 'c',
        long,
        requires = "decimation_bits",
        conflicts_with = "tradeoff_bits"
    )]
    pub checkpoints: bool,

    /// Run the birthday-spacings test instead of the collision test.​
    #[arg(short = 'B', long)]
    pub birthday_spacings: bool,

    /// Number of independent repetitions.​
    #[arg(short, long, default_value_t = 1)]
    pub reps: usize,

    /// PRNG seed. Accepts decimal, or a 0x/0o/0b prefix for hexadecimal, octal, or binary; underscores may separate digits.​
    #[arg(short = 'S', long, default_value_t = 0, value_parser = parse_u64)]
    pub seed: u64,

    /// Print p-values close to 1 in the form `1 − ε`.​
    #[arg(short = 'p', long)]
    pub pretty_p: bool,

    /// Generate data in parallel: the orbit is split into contiguous segments,
    /// one generated per thread. Jump-capable generators jump to each segment
    /// start, others reach it with a sequential pre-scan. This setting is
    /// detrimental if the generator is not jump-capable and there are no
    /// tradeoff bits.
    #[arg(short = 'P', long)]
    pub parallel: bool,

    /// Run only one of the 2ᵇ tradeoff units (0-based) and print its raw count and
    /// its λ share, so the 2ᵇ units can be distributed across invocations and
    /// recombined. Collision: value-interval K. Birthday: spacing-class K. Requires -b.​
    #[arg(long, value_name = "K")]
    pub pass: Option<u64>,
}

impl Args {
    /// The number of top tradeoff bits b (0 when `--tradeoff` is absent). Used by
    /// both the collision and birthday-spacings tests.
    pub fn tradeoff_bits(&self) -> usize {
        self.tradeoff_bits.unwrap_or(0)
    }

    /// Validates argument combinations, reporting inconsistencies through
    /// [`Args::die`] in clap's own error style.
    pub fn validate(&self) {
        if self.t < 1 {
            Self::die("t must be at least 1");
        }
        if self.u < 1 {
            Self::die("u must be at least 1");
        }
        // Guard before 64 - self.s so an out-of-range shift cannot underflow the
        // usize subtraction (which would silently accept the value and later panic
        // or produce garbage extractions).
        if self.s >= 64 {
            Self::die(&format!("shift ({}) must be less than 64", self.s));
        }
        if self.u > 64 - self.s {
            Self::die(&format!(
                "u ({}) can be at most 64 - shift ({})",
                self.u,
                64 - self.s
            ));
        }
        let d = self.decimation_bits.unwrap_or(0);
        // Validate d before using u - d below, so an out-of-range d cannot
        // underflow the tradeoff bound.
        if self.decimation_bits.is_some() {
            if self.m.is_none() {
                Self::die("--decimate requires explicit -m");
            }
            if d < 1 {
                Self::die("--decimate d must be at least 1");
            }
            if d >= self.u {
                Self::die(&format!(
                    "--decimate d ({}) must be less than u ({})",
                    d, self.u
                ));
            }
        }
        if let Some(b) = self.tradeoff_bits {
            if b < 1 {
                Self::die("--tradeoff b must be at least 1");
            }
            if b > self.t * (self.u - d) {
                Self::die(&format!(
                    "--tradeoff b ({}) must be at most t·(u - d) ({})",
                    b,
                    self.t * (self.u - d)
                ));
            }
            // The number of passes is 2ᵇ and must fit in a u64; b >= 64 also
            // describes a wholly infeasible run (≥ 2⁶⁴ passes). Capping here keeps
            // every `1u64 << b` site (validation, the --pass share, the runners)
            // from shift-overflowing.
            if b >= 64 {
                Self::die(&format!("--tradeoff b ({b}) must be less than 64"));
            }
        }
        if self.checkpoints && self.birthday_spacings {
            Self::die("--checkpoints is incompatible with --birthday-spacings");
        }
        if let Some(k) = self.pass {
            match self.tradeoff_bits {
                None => Self::die("--pass requires -b (--tradeoff)"),
                Some(b) => {
                    let num_passes = 1u64 << b;
                    if k >= num_passes {
                        Self::die(&format!(
                            "--pass K ({}) must be less than 2ᵇ ({})",
                            k, num_passes
                        ));
                    }
                }
            }
        }
    }

    /// Reports an argument error in clap's own style (message plus usage footer)
    /// and exits with status 2, indistinguishable from clap's native diagnostics.
    /// Also used by `main` for constraints that involve derived quantities (point
    /// counts, cell counts) rather than raw arguments.
    pub fn die(msg: &str) -> ! {
        use clap::CommandFactory;
        Args::command()
            .error(clap::error::ErrorKind::ValueValidation, msg)
            .exit()
    }

    /// Resolved parallel generation width: `None` for sequential generation,
    /// `Some(n)` with `n` the Rayon pool size (governed by `RAYON_NUM_THREADS`)
    /// when `--parallel` is set.
    pub fn parallel_cpus(&self) -> Option<usize> {
        if self.parallel {
            Some(crate::util::parallelism())
        } else {
            None
        }
    }
}

/// Parses an unsigned 64-bit integer in decimal, or in hexadecimal, octal, or
/// binary when prefixed with `0x`, `0o`, or `0b` (case-insensitive).
/// Underscores are allowed as digit separators (e.g. `0xDEAD_BEEF`).
fn parse_u64(value: &str) -> Result<u64, String> {
    let trimmed = value.trim();
    let (radix, digits) = match trimmed.get(..2) {
        Some("0x") | Some("0X") => (16, &trimmed[2..]),
        Some("0o") | Some("0O") => (8, &trimmed[2..]),
        Some("0b") | Some("0B") => (2, &trimmed[2..]),
        _ => (10, trimmed),
    };
    let digits: String = digits.chars().filter(|&c| c != '_').collect();
    if digits.is_empty() {
        return Err(format!("invalid integer: {value:?}"));
    }
    u64::from_str_radix(&digits, radix).map_err(|e| format!("invalid integer {value:?}: {e}"))
}
