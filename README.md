# Large-scale collision and birthday-spacings tests for pseudorandom number generators

The tests draw the _u_ ≤ 64 − _s_ upper bits from the 64-bit PRNG output
shifted to the left by _s_, _t_ times, forming _t_-tuples of cell indices. If
_d_ < _u_ decimation bits are specified, only those tuples in which all elements
have the lower _d_ bits equal to zero are kept: the cell space has thus size
(2*ᵘ* ⁻ *ᵈ*)_ᵗ_.

If _b_ ≤ _t_ (_u_ − _d_) tradeoff bits are specified, the combined cell index is
partitioned into 2*ᵇ* contiguous value intervals by its top _b_ bits, and each interval is processed
in a pass, iterating multiple times over the PRNG output. Both tests use the same
top-bit partition: for collisions, equal points share their top bits and so land in
the same pass; for birthday spacings, a contiguous interval yields the correct
distances within a pass (with a fix at each interval border), and the spacing
collisions are then counted by a second level keyed on the _low_ _b_ bits of the
spacings (balanced, since spacings cluster near zero) — so the birthday test runs
(2*ᵇ*)² passes.

The argument _m_ is the number of memory locations. The number of
points is then approximately _m_ · 2*ᵇ*. The number of samples from the orbit of the generator
is _m_ · 2*ᵇ* · 2*ᵗᵈ*. For the collision test the number of calls to the generator
is _t_ · _m_ · (2*ᵇ*)² · 2*ᵗᵈ* (each _t_-tuple costs _t_ calls); the birthday test
adds a further factor of 2*ᵇ* for its second level.

The actual allocation will be larger than _m_ by a few percents
in tradeoff mode because the number of points with given upper bits will
slightly vary.

The driver generates points from the selected PRNG (concatenating the _t_
extracted coordinates into a single cell index), then sorts the resulting cell
indices and either counts collisions or measures the distribution of birthday
spacings. _p_-values are
computed against a Poisson reference via [`cdflib`], with the mean conditioned
on the number of points actually kept (relevant under decimation, where the
kept count is random).

The generator to test is selected at compilation time using Cargo features.
For example,

```
cargo build --release --features splitmix64
```

will test SplitMix64. To add a new generator, add a feature in `Cargo.toml`
and a corresponding implementation in the [`prng`] module.

[`cdflib`]: https://crates.io/crates/cdflib
[`prng`]: https://docs.rs/coll/latest/coll/prng/index.html
