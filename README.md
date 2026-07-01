# Large-scale collision and birthday-spacings tests for pseudorandom number generators

This crate implements two empirical tests for pseudorandom number generators
(PRNGs), the _collision test_ and the _birthday-spacings test_. While such tests
are implemented by batteries of tests such as [TestU01] (and, in fact, are very
easy to implement in a naive way), we implement new algorithmic techniques that
open the way to very large scale execution of these tests, even with a limited
amount of memory.

The tests draw the _u_ ≤ 64 − _s_ upper bits from the 64-bit PRNG output
shifted to the left by _s_, _t_ times, forming _t_-tuples of cell indices. If
_d_ < _u_ decimation bits are specified, only those tuples in which all elements
have the lower _d_ bits equal to zero are kept: the cell space has thus size
(2*ᵘ* ⁻ *ᵈ*)_ᵗ_.

The implementation uses scans and radix sorting to process the
output of the PRNG, using multiple cores if available.

# Space-time tradeoffs

The main novelty of this crate is the implementation of [_space-time tradeoffs_]
for the computation of the tests, borrowing standard techniques in database
and stream counting.

If _b_ ≤ _t_ · (_u_ − _d_) tradeoff bits are specified, the combined cell index is
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
_m_, the (approximate) number of memory locations, which, depending on _t_ ·
(_u_ − _d_), will be `u32`, `u64`, or `u128`. Then,

- the number of points is ≈ _m_ · 2*ᵇ*;

- the number of samples from the orbit of the generator is ≈ _m_ · 2*ᵇ* · 2*ᵗᵈ*
  (each _t_-tuple costs _t_ calls);

- the number of calls to the generator for the collision test is ≈ _t_ · _m_ ·
  (2*ᵇ*)² · 2*ᵗᵈ*; the birthday-spacings test adds a further factor of 2*ᵇ* for
  its second level.

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
the use of decimation] to make a one-dimensional collision test most powerful
(but note that her new “birthday test” in the quoted reference is just the
standard collision test you can find in Knuth's TAoCP). Decimation increases the
number of expected collisions because the approximate mean of the distribution
in the sparse case is given by the square of the number of points divided by the
number of cells.

We extend the idea to _t_-dimensional tests. We consider only tuples in which
each of the _t_ numbers has its _d_ lower bits equal to zero, so tuples can
collide only among themselves, and the mean is multiplied by 2*ᵗᵈ* with respect
to an undecimated test. We also extend decimation to the birthday-spacings test.

Decimation can also in principle be applied by considering any subset of bits
being equal to any set of bit pattern, or more general mappings.

# Repetitions

The tests can be repeated multiple times: the statistics can be added together,
and then a single _p_-value is computed.

# Comparison of techniques

Given a budget of _m_ memory locations, a space-time tradeoff on _b_ bits,
decimation with 2*b*/_t_ bits per dimension, or performing 2*²ᵇ* repetitions will use the same
number of calls to the generator, and will end up comparing against the same
mean. The amount of sorted data is the number of memory locations for the
decimated test; the space-time tradeoff adds a factor of 2*ᵇ*, and repetitions
add a factor of 2*²ᵇ*.

If there are no collisions, the _p_-values will be the same. However, if there
are too many collisions, different techniques will yield different statistics
and different _p_-values.

Empirically, it seems that the most powerful technique is space-time tradeoffs
(i.e., a standard collision test), followed by decimation, and finally
repetitions, in the sense that with the same amount of space and number of calls
to the generator one gets better _p_-values in this order. However, this might
depend on the generator, because space-time tradeoffs use the whole orbit under
examination, but decimation looks farther into the orbit. Moreover, space-time
tradeoffs are a technique to implement the collision and the birthday-spacings
tests, whereas decimation is a variant on the tests.

What is pretty clear is that repeating the tests is the most expensive and less
effective technique.

# Usage

The generator to test is selected at compilation time using Cargo features.
For example,

```text
cargo run -r -F splitmix -- 64 1 8000000000 -b 2 -p -P
Generator: SplitMix
Transparent huge pages: always [madvise] never
Seed: 0x0000000000000000
Running a 1-dimensional parallel collision test (20 CPUs, jump-ahead) on the upper 64 bits of the full 64-bit output (32000000000 points, 64-bit cells, 8000000000 memory locations, 59.807 GiB RAM, tradeoff on 2 top bits over 4 passes)
u: 64 t: 1 cells: 18446744073709551616 expected collisions: 27.755575598712134
Pass 1/4: gen...[12.575s] sort...[25.045s] count...[1.047s], 7999983191 points, 0 collisions, p=0.9990306305409877; combined: 7999983191 points, 0 collisions, p=0.9990306305409877
Pass 2/4: gen...[12.543s] sort...[21.221s] count...[1.048s], 7999927485 points, 0 collisions, p=0.9990305368624458; combined: 15999910676 points, 0 collisions, p=1 − 9.39767956526281e-7
Pass 3/4: gen...[12.678s] sort...[21.999s] count...[1.048s], 8000029665 points, 0 collisions, p=0.999030708688019; combined: 23999940341 points, 0 collisions, p=1 − 9.109089151127042e-10
Pass 4/4: gen...[12.585s] sort...[21.952s] count...[1.050s], 8000059659 points, 0 collisions, p=0.9990307591204716; combined: 32000000000 points, 0 collisions, p=1 − 8.828901577425957e-13
0	p=1 − 8.828901577425957e-13	combined: 0	p=1 − 8.828901577425957e-13
Test completed in 146.49 seconds
0	p=1 − 8.828901577425957e-13
```

will test SplitMix on 32 billion points using about 60 GiB of RAM, two tradeoff
bits and parallel generation (we could have obtained the same result using a
fourth of the RAM by choosing four tradeoff bits). Note that each tradeoff pass
is interpretable as a decimation, and each prefix of tradeoff passes as a
multi-pattern decimation, so corresponding _p_-values are output, helping to see
where the computation is going. Since we were expecting _p_-values close to one,
we used the pretty-printing option `-p` to switch to a more accurate display
when the result is close to one.

As we mentioned, since there are no collisions using decimation on 2*b*/_t_ bits
(here _t_ = 1, so 4 bits) per dimension (i.e., 64 billion undecimated points)
would give an essentially equivalent result:

```text
cargo run -r -F splitmix -- 64 1 8000000000 -d 4 -p -P
Generator: SplitMix
Transparent huge pages: always [madvise] never
Seed: 0x0000000000000000
Running a 1-dimensional parallel collision test (20 CPUs, jump-ahead) on the upper 64 bits of the full 64-bit output (8000000000 points, 64-bit cells, 8000000000 memory locations, 59.807 GiB RAM, decimating 4 bits per dimension (~2⁴ candidate samples per kept sample))
u: 64 t: 1 cells: 1152921504606846976 expected collisions: 27.755575547961804 (effective cells after decimation: 2⁶⁰)
Pass 1/1: gen...[18.642s] sort...[26.335s] count...[1.048s], 7999985821 points, 0 collisions, p=1 − 8.829770712901635e-13; combined: 7999985821 points, 0 collisions, p=1 − 8.829770712901635e-13
0	p=1 − 8.829770712901635e-13	combined: 0	p=1 − 8.829770712901635e-13
Test completed in 47.73 seconds
0	p=1 − 8.829770712901635e-13
```

The minuscule difference in the _p_-value is due to the discrepancy between the
target number of decimated points (8 billions) and the actual number of
decimated points (7999985821) obtained from 64 billion points.

Also repeating the test 16 times on 8 billion points would be equivalent (as
1.7347 · 16 = 27.7552):

```text
cargo run -r -F splitmix -- 64 1 8000000000 -r 16 -p -P
Generator: SplitMix
Transparent huge pages: always [madvise] never
Seed: 0x0000000000000000
Running a 1-dimensional parallel collision test (20 CPUs, jump-ahead) on the upper 64 bits of the full 64-bit output (8000000000 points, 64-bit cells, 8000000000 memory locations, 59.605 GiB RAM)
u: 64 t: 1 cells: 18446744073709551616 expected collisions: 1.7347234755091947
Pass 1/1: gen...[1.605s] sort...[24.610s] count...[1.048s], 8000000000 points, 0 collisions, p=0.8235510139931854; combined: 8000000000 points, 0 collisions, p=0.8235510139931854
0	p=0.8235510139931854	combined: 0	p=0.8235510139931854
Pass 1/1: gen...[1.609s] sort...[25.191s] count...[1.050s], 8000000000 points, 0 collisions, p=0.8235510139931854; combined: 8000000000 points, 0 collisions, p=0.8235510139931854
0	p=0.8235510139931854	combined: 0	p=0.968865755337167
[...]
Pass 1/1: gen...[1.609s] sort...[26.398s] count...[1.049s], 8000000000 points, 0 collisions, p=0.8235510139931854; combined: 8000000000 points, 0 collisions, p=0.8235510139931854
0	p=0.8235510139931854	combined: 0	p=1 − 8.828901494125378e-13
Test completed in 465.96 seconds
0	p=1 − 8.828901494125378e-13
```

# Adding your own generator

To add a new generator, add a feature in `Cargo.toml` and a corresponding
implementation in the [`prng`] module. If skipping is possible, you can
implement the `try_skip` method.

# Example: WyRand

[WyRand] is a simple 64-bit generator with 64 bits of state. It increments a counter and
applies a hash using ideas from [Wyhash]. While the generator passes all common statistical
tests, the hash is not sufficient to hide the bias from a large-scale collision test:

```text
cargo run -r -F wyrand -- 64 1 8000000000 -b 3 -P
Generator: wyrand
Transparent huge pages: always [madvise] never
Seed: 0x0000000000000000
Running a 1-dimensional parallel collision test (20 CPUs, jump-ahead) on the upper 64 bits of the full 64-bit output (64000000000 points, 64-bit cells, 8000000000 memory locations, 59.807 GiB RAM, tradeoff on 3 top bits over 8 passes)
u: 64 t: 1 cells: 18446744073709551616 expected collisions: 111.0223023323856
Pass 1/8: gen...[13.200s] sort...[25.677s] count...[1.049s], 8000083095 points, 28 collisions, p=0.0005575160148976676; combined: 8000083095 points, 28 collisions, p=0.0005575160148976676
Pass 2/8: gen...[13.214s] sort...[21.772s] count...[1.047s], 8000011384 points, 20 collisions, p=0.07162240780462378; combined: 16000094479 points, 48 collisions, p=0.0003042040178853015
[...]
Pass 8/8: gen...[13.298s] sort...[22.828s] count...[1.049s], 8000119446 points, 22 collisions, p=0.026591939215130665; combined: 64000000000 points, 222 collisions, p=1.2959913978087028e-20
222	p=1.2959913978087028e-20	combined: 222	p=1.2959913978087028e-20
Test completed in 298.37 seconds
222	p=1.2959913978087028e-20
```

# Example: an affine congruential generator that is _too good_

Multipliers for affine congruential generators—ACGs, commonly known as linear
congruential generators (LCGs), even if their map is _affine_, not _linear_—are
judged on the basis of the _spectral test_, which computes the distance between
hyperplanes spanned by vectors of consecutive outputs. It is a staple of the
literature on the topic since the 60's that you should strive for the smallest
possible distance, to which one associates a large _figure of merit_. A large
body of research has studied spectral scores, and studied how to obtain
multipliers with large figures of merit.

Much less known is that figures of merit have nothing to do with the randomness of
the output of the generator—they just describe its _uniformity_. If a multiplier
is not uniform enough, it will fail collision test because too many outputs end
up in the same cell.

However, if you can run large-scale collision test, a multiplier that is _too
good_ will fail, too, as the hyperplanes are still there:

```text
cargo run -r -F lcg_64_64_0xa5b9ee81534fa94d -- 32 2 8000000000 -b 3 -p -P
Generator: LCG64 (0xa5b9ee81534fa94d)
Transparent huge pages: always [madvise] never
Seed: 0x0000000000000000
Running a 2-dimensional parallel collision test (20 CPUs, jump-ahead) on the upper 32 bits of the full 64-bit output (64000000000 points, 64-bit cells, 8000000000 memory locations, 59.807 GiB RAM, tradeoff on 3 top bits over 8 passes)
u: 32 t: 2 cells: 18446744073709551616 expected collisions: 111.0223023323856
Pass 1/8: gen...[20.506s] sort...[26.332s] count...[1.050s], 8000076053 points, 3 collisions, p=0.9994770836776423; combined: 8000076053 points, 3 collisions, p=0.9994770836776423
Pass 2/8: gen...[20.129s] sort...[21.184s] count...[1.053s], 8000194725 points, 4 collisions, p=0.9980257653489827; combined: 16000270778 points, 7 collisions, p=1 − 2.9280185095831245e-6
[...]
Pass 8/8: gen...[19.392s] sort...[22.802s] count...[1.050s], 8000186794 points, 2 collisions, p=0.9998955967799726; combined: 64000000000 points, 16 collisions, p=1 − 1.804698905648335e-29
16	p=1 − 1.804698905648335e-29	combined: 16	p=1 − 1.804698905648335e-29
Test completed in 353.10 seconds
16	p=1 − 1.804698905648335e-29
```

The multiplier, for 64-bit ACGs with 64 bits of state, has been found during the
large-scale search that [Guy Steele and I conducted to improve spectral
coefficients]. Its *f*₂ figure of merit is a whopping 0.977689—almost perfect.
As a result, the generator fails catastrophically to reproduce the right number
of collisions for pairs of consecutive outputs. Note that without space-time
tradeoffs the test would require half a terabyte of RAM.

# Example: multiply-with-carry generators

Marsaglia's multiply-with-carry generators are very fast generators with
arbitrarily large periods. While they pass most typical statistical tests (in
fact, with sufficient state, all tests), it is known that [their output is
tightly coupled with that of a linear congruential generator with large prime
modulus] (an actual _linear_ congruential generator, not an _affine_ one,
sometimes called a _multiplicative_ generator because of the confusion between
linear and affine generators discussed above). Spectral analysis shows that such
generators have inherently bad figures of merit *f*₃, but obtaining concrete
failures in statistical test is not easy due to the large state space. However,
we can find bias using birthday spacings in a 64-bit MWC with 128 bits of state:

```text
cargo run -r -F mwc_128_64_0xffebb71d94fcdaf9 36 3 2000000000 -B -P -b 6
Generator: MWC128 (0xffebb71d94fcdaf9)
Transparent huge pages: always [madvise] never
Seed: 0x0000000000000000
Running a 3-dimensional parallel birthday-spacings test (20 CPUs, jump-ahead) on the upper 36 bits of the full 64-bit output (128000000000 points, 128-bit cells, 2000000000 memory locations, 59.853 GiB RAM, tradeoff on 6 top bits over 64 value intervals x 64 spacing classes)
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

Our test can detect the bias on a standard workstation. In this case, running
the test in a naïve way would require at least a terabyte of RAM.

# Acknowledgments

I would like to thank the GitHub user `alvoskov` for a [very interesting
discussion] that stimulated me to investigate space-time tradeoffs and publish
this crate.

[`cdflib`]: https://crates.io/crates/cdflib
[`prng`]: https://docs.rs/coll-birth/latest/coll_birth/prng/index.html
[WyRand]: https://github.com/wangyi-fudan/wyhash
[WyHash]: https://github.com/wangyi-fudan/wyhash
[very interesting discussion]: https://github.com/alvoskov/SmokeRand/issues/24
[propose the use of decimation]: https://www.pcg-random.org/posts/birthday-test.html
[Guy Steele and I conducted to improve spectral coefficients]: https://doi.org/10.1002/spe.3030
[their output is tightly coupled with that of a linear congruential generator with large prime modulus]: https://www.jstor.org/stable/2153884
[_space-time tradeoffs_]: https://doi.org/10.1137/0220017
[TestU01]: https://doc.org/10.1145/1268776.1268777
