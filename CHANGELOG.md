# Change Log

## [0.2.0] - 2026-07-01

### Changed

- Removed unused logging code and dependencies.

- `-P`/`--parallel` is now a boolean flag: it enables parallel generation on the
  Rayon global thread pool, and the number of threads for every phase
  (generation, sorting, counting) is controlled by `RAYON_NUM_THREADS`.

- Improved option names.

## [0.1.0] - 2026-07-01

### New

- First release.
