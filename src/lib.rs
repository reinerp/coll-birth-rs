/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

pub mod birthday;
pub mod cdf;
pub mod cell;
#[doc(hidden)]
pub mod cli;
pub mod collision;
pub mod common;
// `try_skip` returns `Result<(), ()>` as a capability probe (Ok = skip-capable,
// Err = not), an intentional internal signal rather than a real error type; the
// signature is shared across every per-generator impl, so allow the public-API
// lint module-wide instead of annotating each one.
#[allow(dead_code, clippy::result_unit_err)]
pub mod prng;
pub mod stats;
pub mod util;
