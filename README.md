# Large-scale collision and birthday-spacings tests for pseudorandom number generators

This crate implements two empirical tests for pseudorandom number generators
(PRNGs), the _collision test_ and the _birthday-spacings test_. While such tests
are implemented by batteries of tests such as TestU01 (and, in fact, are very
easy to implement in a naive way), we implement new algorithmic techniques that
open the way to very large scale execution of these tests, even with a limited
about of memory.

The tests draw the _u_ ≤ 64 − _s_ upper bits from the 64-bit PRNG output
shifted to the left by _s_, _t_ times, forming _t_-tuples of cell indices. If
_d_ < _u_ decimation bits are specified, only those tuples in which all elements
have the lower _d_ bits equal to zero are kept: the cell space has thus size
(2*ᵘ* ⁻ *ᵈ*)_ᵗ_.

# Space-time tradeoffs

The main novelty of this crate is the implementation of _space-time tradeoffs_
for the computation of the tests, borrowing standard techniques in database
and stream counting.

If _b_ ≤ _t_ (_u_ − _d_) tradeoff bits are specified, the combined cell index is
partitioned into 2*ᵇ* contiguous value intervals by its top _b_ bits, and each
interval is processed in a pass, iterating multiple times over the PRNG output.
Both tests use the same top-bit partition: for collisions, equal points share
their top bits and so land in the same pass; for birthday spacings, a contiguous
interval yields the correct distances within a pass (with a fix at each interval
border), and the spacing collisions are then counted by a second level of
tradeoff, this time based on the the _lower_ _b_ bits of the spacings (balanced,
since spacings cluster near zero); thus, the birthday-specings test runs
(2*ᵇ*)² passes.

For simplicity of interation with the tool, the main parameter to the tests is
_m_, the (approximate) number of memory locations to use. Then,

- the number of points is approximately _m_ · 2*ᵇ*;

- the number of samples from the orbit of the generator is _m_ · 2*ᵇ* · 2*ᵗᵈ* (each _t_-tuple costs _t_ calls);

- for the collision test, the number of calls to the generator is _t_ · _m_ ·
  (2*ᵇ*)² · 2*ᵗᵈ*;

- the birthday-spacings test adds a further factor of 2*ᵇ* for its second level.

The actual allocation will be larger than _m_ by a few percents in tradeoff mode
because the number of points with given upper bits will slightly vary.

The driver generates points from the selected PRNG, then sorts the resulting
cell indices and either counts collisions or measures the distribution of
birthday spacings. _p_-values are computed against a Poisson reference via
[`cdflib`], with the mean conditioned on the number of points actually kept
(relevant under decimation, where the kept count is random).

If parallel cores are available, the generation of the output will happen in
parallel: the part of the orbit that need to be generated is split into
segments, and each core generate a segment. If the PRNG supports skipping, the
starting states for each segment are computed using skipping. If no skipping is
available, parallel generation is used only in case of tradeoffs, as even with 2
processors and one tradeoff bits enumerating the part of the orbit to find the
initial state of each segment breaks even, and with more processor it becomes
competitive.

# Usage

The generator to test is selected at compilation time using Cargo features.
For example,

```
cargo run --release --features splitmix -- 64 1 4000000000 -b 3 -P
```

will test SplitMi using about 29.9 GiB of RAM, using four tradeoff bits and parallel generations.

. To add a new generator, add a feature in `Cargo.toml`
and a corresponding implementation in the [`prng`] module.

# Example: WyRand

WyRand is a simple 64-bit generator with 64 bits of state. It increments a counter and
apply a hash using ideas from [Wyhash]. While the generator passes all common statistical
tests, the hash is not sufficient to hide the bias towards collisions:

```bash

```

# Example: an affine congruential generator that is _too good_

Multipliers for affine congruential generators (AGC, erroneously called linear,
so, LCG, since ever) are judged on the basis of the _spectral test_, which
computes the distance between hyperplanes spanned by vectors of consecutive
outputs. It is a staple of the literature on the topic since the 60's that you
should strive for the smallest possible distance, to which one associates a
large _figure of merit_. A large body of research has studied spectral scores,
and studied how to obtain multipliers with large figures of merit. Less known is
that figures of merit have nothing to do with the randomness of the output of
the generator—they just describe its _uniformity_. If a multiplier is not
uniform enough, it will fail collision test because too many outputs end up in
the same cell.

However, if you can run large-scale collision test, a multiplier that is _too
good_ will fail, too:

```bash
Generator: LCG64 (0xa5b9ee81534fa94d)
Seed: 0x18ba2f5e32070688
Running a parallel collision test (10 CPUs, faithful split (jump-ahead)) on the upper 32 bits of the full 64-bit output using 64000000000 points (64-bit cells, 29.904 GiB live, tradeoff on 4 top bits over 16 passes)
u: 32 t: 2 cells: 18446744073709551616 expected collisions: 111.0223023323856
Pass 1/16: gen...[22.816s] sort...[9.282s] count...[12.440s], 3999999761 points, 0 collisions, p=0.9990306579978518; combined: 3999999761 points, 0 collisions, p=0.9990306579978518
[...]
26      p=1     combined: 26    p=1
Test completed in 609.44 seconds
26      p=1
```

The multiplier, for 64-bit ACGs with 64 bits of state, has been found during the
large-scale search that [I conducted with Guy Steele to improve spectral
coefficients]. Its *f*₂ figure of merit is a whopping 0.977689—almost perfect.
As a result, the generator fails catastrophically to reproduce the right number
of collisions for pairs of consecutive outputs. Note that without space-time
tradeoffs the test would require half a terabyte of RAM.

# Example: multiply-with-carry generators

Marsaglia's multiply-with-carry generators are very.

While they pass most statistical tests, it is known that their output is tightly
couple with that of a linear congruential generator with large prime modulus (an
actual linear congruential generator, sometimes called a _multiplicative_
generator because of the confusion between linear and affine generators
discussed above). Spectral analysis shows that such generators have inherently
bad figures of merit *f*₃, but obtaining concrete failures in statistical test is
not easy due to the large state space. However, we can find bias using birthday
spacing in a 64-bit MWC with 128 bits of state:

```bash

```

The multiplier has excellent scores, but our tests can detect its bias on a
standard workstation. In this case, running the test in a naïve way would
require terabytes of RAM.

[`cdflib`]: https://crates.io/crates/cdflib
[`prng`]: https://docs.rs/coll/latest/coll/prng/index.html

[I conducted with Guy Steele]:
