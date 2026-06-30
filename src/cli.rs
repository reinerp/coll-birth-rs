/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Command-line argument definitions.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None, next_line_help = true, max_term_width = 100)]
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
    pub tradeoff: Option<usize>,

    /// Decimate: keep only tuples in which every coordinate's lowest d bits are zero;
    /// a fixed m · 2ᵇ · 2ᵗᵈ samples are scanned, keeping ~m · 2ᵇ.​
    #[arg(short = 'd', long, value_name = "d")]
    pub decimate: Option<usize>,

    /// Print progressive p-values at ⌊√(2ᵈ)⌋ uniform checkpoints; works with -P.​
    #[arg(short = 'c', long, requires = "decimate", conflicts_with = "tradeoff")]
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

    /// Run in parallel on P CPUs; bare `-P` uses all available, or pass a count as `-P=P`. The
    /// single sequential orbit is split into P contiguous segments, so the result is
    /// identical to a sequential run for every generator and mode: jump-capable generators
    /// jump to each segment start, others reach it with a sequential pre-scan.​
    #[arg(short = 'P', long, value_name = "P", num_args = 0..=1, require_equals = true, default_missing_value = "0")]
    pub parallel: Option<usize>,

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
        self.tradeoff.unwrap_or(0)
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
        let d = self.decimate.unwrap_or(0);
        // Validate d before using u - d below, so an out-of-range d cannot
        // underflow the tradeoff bound.
        if self.decimate.is_some() {
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
        if let Some(b) = self.tradeoff {
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
            match self.tradeoff {
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

    /// Resolved parallel CPU count: `None` for sequential, `Some(p)` for parallel.
    pub fn parallel_cpus(&self) -> Option<usize> {
        self.parallel.map(|p| {
            if p == 0 {
                crate::util::parallelism()
            } else {
                p
            }
        })
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

/// Initializes the `env_logger` logger with a custom format including
/// timestamps with elapsed time since initialization.
pub fn init_env_logger() -> anyhow::Result<()> {
    use jiff::{
        SpanRound,
        fmt::friendly::{Designator, Spacing, SpanPrinter},
    };
    use std::io::Write;

    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));

    let start = std::time::Instant::now();
    let printer = SpanPrinter::new()
        .spacing(Spacing::None)
        .designator(Designator::Compact);
    let span_round = SpanRound::new()
        .largest(jiff::Unit::Day)
        .smallest(jiff::Unit::Millisecond)
        .days_are_24_hours();

    builder.format(move |buf, record| {
        let Ok(ts) = jiff::Timestamp::try_from(std::time::SystemTime::now()) else {
            return Err(std::io::Error::other("Failed to get timestamp"));
        };
        let style = buf.default_level_style(record.level());
        let elapsed = start.elapsed();
        let span = jiff::Span::new()
            .seconds(elapsed.as_secs() as i64)
            .milliseconds(elapsed.subsec_millis() as i64);
        let span = span.round(span_round).expect("Failed to round span");
        writeln!(
            buf,
            "{} {} {style}{}{style:#} [{:?}] {} - {}",
            ts.strftime("%F %T%.3f"),
            printer.span_to_string(&span),
            record.level(),
            std::thread::current().id(),
            record.target(),
            record.args()
        )
    });
    builder.init();
    Ok(())
}
