# Single-buffer + in-place compaction for the parallel runners

**Date:** 2026-06-23
**Status:** approved, implementing

## Problem

The parallel collision runner counts cross-thread collisions with a k-way merge
over the `num_cpus` per-thread sorted buffers (`merge_count_collisions` →
`merge_count_segment`, a quaternary min-heap). On large inputs this `count`
phase dominates: for 2×10⁹ points on 10 CPUs it took ~6.1 s — far more than the
sort (~2.7 s) — because each element pays a heap push/pop whose next memory load
is data-dependent (latency-bound), and the work is only O(n log k) but with poor
ILP and cache behaviour. The heap is the *right asymptotics* (a linear "min over
k heads" scan would be O(n·k), a disaster as `num_cpus` grows), so the fix is not
a different merge but to **avoid the merge entirely**.

If all kept points were in one contiguous buffer, counting collapses to a single
`sort_mt` + a flat `count_adjacent_equals` linear scan — the cheap, prefetch-
friendly path the *sequential* runner already uses, and the path the *birthday*
parallel runner already uses per value-interval (it gathers per-thread blocks
into one `interval_buf`, sorts, and scans — no heap).

## Approach

Introduce one shared primitive: **generate a work-unit's points into a single
contiguous buffer, in place.** Threads generate into disjoint sub-regions of one
buffer (carved by their `buffer_size` caps); a compaction step then closes the
gaps left by under-filled sub-regions, leaving the kept points contiguous in
`buf[..total_used]`. One `sort_mt` + one linear count follows.

### Why in-place (not gather-into-a-second-buffer)

The birthday runner gathers into a *separate* `interval_buf` (disjoint
destination slots → trivially parallel). That is fine when the unit is small
(`points/2^b` per interval), but the collision plain unit is *all* the points; a
second full-size buffer would double peak RSS (~+15 GiB in the motivating run).
In-place compaction keeps memory equal to today's (one buffer of `Σ caps`
instead of `k` buffers summing to the same), at the cost of the moves being
*leftward and overlapping*.

### Feasibility boundary (why compaction, not exact offsets or sentinels)

To make the kept points contiguous for one linear count we need one of:

- **Exact offsets / gap-free direct write** — only possible in *plain* mode,
  where each thread keeps exactly `chunk(i)` points (counts known a priori).
- **Sentinel-pad + sort** — needs a value that is never a legal cell; impossible
  when the cell space saturates the storage width (`cells == 2^(8·sizeof(T))`,
  i.e. `(u−d)·t ∈ {32,64,128}` — e.g. `u=64, t=1` → `2^64` cells in `u64`).
- **Compaction** — universal: works for variable per-thread counts (decimation,
  tradeoff) *and* at cell-space saturation, in place, no spare value needed.

Compaction is therefore the mechanism that makes single-buffer work in **every**
mode, including the decimated `2^64`-cell runs that motivate this change. In
*plain* mode it is a **no-op**: counts are exact so each block's destination
equals its source and nothing is copied.

### Compaction algorithm (sequential-block, in place)

Per work-unit, after generation returns per-thread used counts `used[i]`:

```
bases[i] = prefix sum of caps   (sub-region starts; where thread i wrote)
dsts[i]  = prefix sum of used   (compacted destinations)
for i in 0..k:                  // sequential, left to right
    if dsts[i] != bases[i]:
        buf.copy_within(bases[i] .. bases[i]+used[i], dsts[i])   // memmove
total_used = dsts[k-1] + used[k-1]
```

**Correctness:** `dsts[i] ≤ bases[i]` for all `i` (since `Σ used ≤ Σ caps`), so
every move is leftward. Going left-to-right, when block `i` is moved block `i−1`
is already at its (more-leftward) destination, so block `i`'s source is free to
read and its destination ends at `dsts[i]+used[i] = dsts[i+1] ≤ bases[i+1]`, the
start of block `i+1`'s still-untouched source — no inter-block clobber.
`copy_within` (memmove semantics) handles the intra-block leftward overlap.

Sequential per-block copies (each a single `copy_within`) are the agreed
baseline; total work is one O(n) pass and it only runs off the plain path. A
tiled/parallel variant is a possible later optimization, not required for
correctness.

## The shared helper (`common.rs`)

```rust
/// Faithfully generate one work-unit into a single contiguous buffer.
/// Threads fill disjoint sub-regions of `buf` (carved by `caps`), then the gaps
/// are closed by an in-place left-compaction. Returns `total_used`; on return
/// `buf[..total_used]` holds the kept points (unsorted).
pub(crate) fn gen_unit_contiguous<T: Cell>(
    buf: &mut [T],
    caps: &[usize],            // per-thread sub-region capacities; Σ caps ≤ buf.len()
    snapshots: &[Prng],        // per-thread orbit starts (len == caps.len())
    params: &GridParams,
    chunk: &(dyn Fn(usize) -> usize + Sync), // stream_len per thread
    pass: u64,
    tradeoff_b: usize,
    decimating: bool,
    full: bool,
) -> usize
```

Implementation: `split_at_mut` `buf` into the `caps`-sized sub-regions, spawn one
thread per sub-region calling `gen_pass_dispatch` (returns `(used, _prng)`),
collect `used[i]`, then run the compaction above. Threads write disjoint
sub-slices (safe); compaction runs after the join (sequential).

## Call-site changes

### Collision plain / tradeoff (`collision::run_test_parallel`, main loop)
- Allocate one `buf` of `total_buf = Σ buffer_size(chunk(i), partition_bits)`
  (already computed as `total_buf`) instead of `k` per-thread mmaps.
- Per pass: `let total = gen_unit_contiguous(buf, &caps, &snapshots, …, pass, …);`
  `T::sort_mt(&mut buf[..total]); let c = count_adjacent_equals(&buf[..total]);`
- Delete the per-pass three-phase gen/sort + `merge_count_collisions`.

### Collision checkpoint (`-c`)
- One stage buffer of `buffer_size(max_stage, partition_bits)` instead of `k`
  thread buffers; `acc` accumulator unchanged.
- Per stage: `gen_unit_contiguous` → `sort_mt` (this is the sorted stage run,
  replacing `merge_sorted`) → `merge_into(acc, acc_len, stage_run)` (**kept**,
  two-way, no heap) → `count_adjacent_equals(acc[..acc_len])`.

### Birthday (`run_birthday_parallel`)
- One `buf` of `Σ buffer_size(chunk(i), partition_bits)` instead of separate
  `blocks` + `interval_buf` (drops the `blocks` allocation and the gather copy).
- Per interval `k`: `let total = gen_unit_contiguous(buf, &caps, …, pass=k, b, …);`
  `T::sort_mt(&mut buf[..total]);` then the existing spacing/classify logic over
  `buf[..total]` (predecessor `prev_max`, `global_min/max`, class buffer, wrap)
  is unchanged.

## Deletions
`merge_count_collisions`, `merge_count_collisions_segmented`,
`merge_count_segment`, `merge_sorted`, the `dary_heap::QuaternaryHeap` and
`std::cmp::Reverse` imports, and the inline `tests` module's
`segmented_merge_matches_brute_force` / `segmented_merge_edge_cases`.

## Kept
`merge_into`, `count_adjacent_equals`, `OrbitPartition` and all faithfulness
machinery, all statistics (`test_lambda`, p-values, per-pass/rep means).

## Testing
- **New inline unit test** for `gen_unit_contiguous` compaction: with the `incr`
  generator (deterministic), assert that for several `caps`/`used` patterns
  (including all-full = plain no-op, some-empty, first-empty, last-empty) the
  returned `buf[..total]` is exactly the concatenation of the kept prefixes and
  `total == Σ used`. A pure compaction-only helper can also be unit-tested with
  synthetic `used` counts (no PRNG) for the overlap edge cases.
- **Existing `tests/parallel.rs`** (parallel == sequential, bit-identical, for
  plain / tradeoff / decimation / decimation+tradeoff / checkpoints / birthday)
  is the primary regression guard and must pass unchanged.
- Replace the deleted segmented-merge inline tests; the brute-force intent is now
  covered by `tests/parallel.rs` faithfulness + the compaction unit test.

## Statistics / faithfulness
Unchanged. Collision count = adjacent-equal pairs in the sorted union, identical
regardless of how the union is assembled. Birthday spacings come from the same
sorted-by-interval order with the same border-predecessor carry. Same orbit
split ⇒ same multiset ⇒ bit-identical to the sequential runner for every CPU
count.

## Out of scope
Parallelizing the per-block compaction copy (sequential baseline is the agreed
choice). No change to sequential runners, CLI, or statistics.
