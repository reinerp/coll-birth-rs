# Large-scale collision and birthday-spacings tests for pseudorandom number generators

This crate implements two empirical tests for pseudorandom number generators
(PRNGs), the _collision test_ and the _birthday-spacings test_. While such tests
are implemented by batteries of tests such as TestU01 (and, in fact, are very
easy to implement in a naive way), we implement new parallel algorithmic
techniques that open the way to very large scale execution of these tests, even
with a limited amount of memory.

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
tradeoff, this time based on the _lower_ _b_ bits of the spacings (balanced,
since spacings cluster near zero); thus, the birthday-spacings test runs
(2*ᵇ*)² passes.

For simplicity of interaction with the tool, the main parameter to the tests is
_m_, the (approximate) number of memory locations to use. Then,

- the number of points is approximately _m_ · 2*ᵇ*;

- the number of samples from the orbit of the generator is _m_ · 2*ᵇ* · 2*ᵗᵈ*
  (each _t_-tuple costs _t_ calls);

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

If multiple cores are available, an option can make the generation of the output
happen in parallel: the part of the orbit that needs to be generated is split
into segments, and each core generates a segment. If the PRNG supports skipping,
the starting states for each segment are computed using skipping. If no skipping
is available, parallel generation is used only in case of tradeoffs, as even
with 2 processors and one tradeoff bit enumerating the relevant part of the
orbit to find the initial state of each segment breaks even, and with more
processors it becomes competitive.

# Decimation

To the best of my knkowledge, Melissa O'Neill's has been the first to [propose
the use of decimation] to make a collision test most powerful (but note that her
new “birthday test” in the quoted reference is just the standard collision test
you can find in Knuth's TAoCP). Decimation increases the number of expected collisions
because the approximate mean of the distribution in the sparse case is given by the square
of the number of points divided by the number of cells. Since numbers with the _d_
lower bits equal to zero can cause collisions only among themselves, the mean is multiplied
by 2*ᵈ* with respect to an undecimated test.

The idea can be extended to any subset of bits being equal to any set of bit
pattern, or more general mappings.

# Repetitions

The tests can be repeated multiple times: the statistics can be added together,
from which a single _p_-value is computed.

# Comparison of techniques

Given a budget of _m_ memory locations, a space-time tradeoff on _b_ bits,
decimation with 2*b* bits, or performing 2*²ᵇ* repetitions will use the same
number of calls to the generator, and will end up comparing against the same
mean.

If there are no collisions, the _p_-values will be the same. However, if there
are too many collisions, different techniques will yield different statistics
and different _p_-values.

Empirically, it seems that the most powerful technique is space-time tradeoffs
(i.e., a standard collision test), followed by decimation, and finally
repetitions. However, this might depend on the generator, because space-time
tradeoffs use the whole orbit under examination, but decimation looks farther
into the orbit. Moreover, space-time tradeoffs are a technique to implement the
collision and the birthday-spacings tests, whereas decimation is a variant on
the test.

# Usage

The generator to test is selected at compilation time using Cargo features.
For example,

```text
cargo run -r -F splitmix -- 64 1 4000000000 -b 4 -p -P
Generator: SplitMix
Seed: 0x18bb6e430a1df210
Running a parallel collision test (10 CPUs, jump-ahead) on the upper 64 bits of the full 64-bit output using 48000000000 points (64-bit cells, 22.439 GiB RAM, tradeoff on 4 top bits over 16 passes)
u: 64 t: 1 cells: 18446744073709551616 expected collisions: 62.45004507969723
Pass 1/16: gen...[10.207s] sort...[4.607s] count...[9.962s], 3000005913 points, 0 collisions, p=0.9798216130914219; combined: 3000005913 points, 0 collisions, p=0.9798216130914219
Pass 2/16: gen...[9.916s] sort...[3.741s] count...[10.261s], 2999960789 points, 0 collisions, p=0.979819243690045; combined: 5999966702 points, 0 collisions, p=0.99959278489142
[...]
Pass 16/16: gen...[12.642s] sort...[4.773s] count...[12.193s], 4000072070 points, 0 collisions, p=0.9990309011505323; combined: 64000000000 points, 0 collisions, p=1 − 6.0761254074375494e-49
0	p=1 − 6.0761254074375494e-49	combined: 0	p=1 − 6.0761254074375494e-49
Test completed in 471.27 seconds
0	p=1 − 6.0761254074375494e-49
```

will test SplitMix using about 22.4 GiB of RAM, using four tradeoff bits and
parallel generations. Note that each tradeoff pass is interpretable as a
decimation, and each prefix of tradeoff passes as a multi-target decimation, so
corresponding _p_-values are output, helping to see where the computation is
going. Since we were expecting _p_-values close to one, we used the pretty-printing option
`-p` to switch to a more accurate display.

# Adding

To add a new generator, add a feature in `Cargo.toml` and a corresponding
implementation in the [`prng`] module. If skipping is possible, you can
implement the `try_skip` method.

# Example: WyRand

[WyRand] is a simple 64-bit generator with 64 bits of state. It increments a counter and
applies a hash using ideas from [Wyhash]. While the generator passes all common statistical
tests, the hash is not sufficient to hide the bias from a large-scale collision test:

```text
cargo run --release --features wyrand -- 64 1 5000000000 -b 4 -P
Generator: wyrand
Seed: 0x18bb7fa97c0acbb0
Running a parallel collision test (10 CPUs, jump-ahead) on the upper 64 bits of the full 64-bit output using 80000000000 points (64-bit cells, 37.366 GiB RAM, tradeoff on 4 top bits over 16 passes)
u: 64 t: 1 cells: 18446744073709551616 expected collisions: 173.47234734474017
Pass 1/16: gen...[13.229s] sort...[10.799s] count...[14.386s], 4999944051 points, 20 collisions, p=0.008035262486159036; combined: 4999944051 points, 20 collisions, p=0.008035262486159036
Pass 2/16: gen...[10.049s] sort...[5.861s] count...[16.074s], 4999999826 points, 18 collisions, p=0.0286001462015577; combined: 9999943877 points, 38 collisions, p=0.0009458204930775911
[...]
Pass 16/16: gen...[10.290s] sort...[5.745s] count...[15.337s], 4999930072 points, 20 collisions, p=0.008034809669851041; combined: 80000000000 points, 309 collisions, p=1.2406616831619787e-20
309	p=1.2406616831619787e-20	combined: 309	p=1.2406616831619787e-20
Test completed in 511.66 seconds
309	p=1.2406616831619787e-20
```

# Example: an affine congruential generator that is _too good_

Multipliers for affine congruential generators—ACGs, commonly known as linear
congruential generators (LCGs), even if their map is _affine_, not _linear_—are
judged on the basis of the _spectral test_, which computes the distance between
hyperplanes spanned by vectors of consecutive outputs. It is a staple of the
literature on the topic since the 60's that you should strive for the smallest
possible distance, to which one associates a large _figure of merit_. A large
body of research has studied spectral scores, and studied how to obtain
multipliers with large figures of merit. Less known is that figures of merit
have nothing to do with the randomness of the output of the generator—they just
describe its _uniformity_. If a multiplier is not uniform enough, it will fail
collision test because too many outputs end up in the same cell.

However, if you can run large-scale collision test, a multiplier that is _too
good_ will fail, too, as the hyperplanes are still there:

```text
cargo run -r -F lcg_64_64_0xa5b9ee81534fa94d -- 32 2 4000000000 -b 4 -p -P
Generator: LCG64 (0xa5b9ee81534fa94d)
Seed: 0x18bb783e2bfe78f0
Running a parallel collision test (10 CPUs, jump-ahead) on the upper 32 bits of the full 64-bit output using 64000000000 points (64-bit cells, 29.904 GiB RAM, tradeoff on 4 top bits over 16 passes)
u: 32 t: 2 cells: 18446744073709551616 expected collisions: 111.0223023323856
Pass 1/16: gen...[18.022s] sort...[5.342s] count...[11.741s], 3999990502 points, 1 collisions, p=0.992304281430322; combined: 3999990502 points, 1 collisions, p=0.992304281430322
Pass 2/16: gen...[18.263s] sort...[4.610s] count...[12.012s], 4000000906 points, 1 collisions, p=0.9923045242213231; combined: 7999991408 points, 2 collisions, p=0.9998955354592226
[...]
Pass 16/16: gen...[17.052s] sort...[4.359s] count...[11.932s], 4000103062 points, 1 collisions, p=0.9923069078008872; combined: 64000000000 points, 19 collisions, p=1 − 4.384243826101139e-27
19	p=1 − 4.384243826101139e-27	combined: 19	p=1 − 4.384243826101139e-27
Test completed in 544.79 seconds
19	p=1 − 4.384243826101139e-27
```

The multiplier, for 64-bit ACGs with 64 bits of state, has been found during the
large-scale search that [I conducted with Guy Steele to improve spectral
coefficients]. Its *f*₂ figure of merit is a whopping 0.977689—almost perfect.
As a result, the generator fails catastrophically to reproduce the right number
of collisions for pairs of consecutive outputs. Note that without space-time
tradeoffs the test would require half a terabyte of RAM.

# Example: multiply-with-carry generators

Marsaglia's multiply-with-carry generators are very fast generator with
arbitrarily large periods. While they pass most typical statistical tests (in
fact, with sufficient state, all tests), it is known that their output is
tightly coupled with that of a linear congruential generator with large prime
modulus (an actual _linear_ congruential generator, not an _affine_ one,
sometimes called a _multiplicative_ generator because of the confusion between
linear and affine generators discussed above). Spectral analysis shows that such
generators have inherently bad figures of merit *f*₃, but obtaining concrete
failures in statistical test is not easy due to the large state space. However,
we can find bias using birthday spacings in a 64-bit MWC with 128 bits of state:

```bash
cargo run -r -F mwc_128_64_0xffebb71d94fcdaf9 36 3 2000000000 -B -P -b 6
Generator: MWC128 (0xffebb71d94fcdaf9)
Transparent huge pages: always [madvise] never
Seed: 0x0000000000000000
Running a parallel birthday-spacings test (20 CPUs, jump-ahead) on the upper 36 bits of the full 64-bit output (128000000000 points, 128-bit cells, 2000000000 memory locations, 59.853 GiB RAM, tradeoff on 6 top bits over 64 value intervals x 64 spacing classes)
u: 36 t: 3 cells: 324518553658426726783156020576256 expected collisions: 1.6155871338926322
Rep 1/1: 64 value intervals x 64 spacing classes
  Class 1/64, interval 1/64: gen...[90.838s] sort...[10.731s] filter...[1.067s], 2000034155 points
  Class 1/64, interval 2/64: gen...[90.889s] sort...[10.882s] filter...[1.076s], 1999968926 points
[...]
 Class 64/64, interval 64/64: gen...[57.182s] sort...[13.009s] filter...[1.105s], 2000023578 points
 Class 64/64 done: [6064.541s], 2000055256 spacings, 2 collisions, p=0.00031330676200791153; combined: 65 collisions, p=8.595346190145383e-79
[392078.211s] 65	p=8.595346190145383e-79	combined: 65	p=8.595346190145383e-79
Test completed in 392079.89 seconds
65	p=8.595346190145383e-79
```

The multiplier has excellent scores, but our tests can detect its bias on a
standard workstation. In this case, running the test in a naïve way would
require terabytes of RAM.

# Acknowledgments

I would like to thank the GitHub user `alvoskov` for a [very interesting
discussion] that stimulated me to publish this create.

[`cdflib`]: https://crates.io/crates/cdflib
[`prng`]: https://docs.rs/coll-birth/latest/coll_birth/prng/index.html
[I conducted with Guy Steele]: https://doi.org/10.1002/spe.3030
[WyRand]: https://github.com/wangyi-fudan/wyhash
[WyHash]: https://github.com/wangyi-fudan/wyhash
[very interesting discussion]: https://github.com/alvoskov/SmokeRand/issues/24
[propose the use of decimation]: https://www.pcg-random.org/posts/birthday-test.html
