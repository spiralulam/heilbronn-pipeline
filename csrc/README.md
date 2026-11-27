# heil: the C search engine

One binary for the ruin & recreate searches of the pipeline: place n points in the unit square (`s`), in the
triangle (0,0),(1,0),(0,1) (`t`) or in the convex variant (`c`, value relative to the hull area) so that the
smallest triangle is as large as possible. `heilbronn/run.py` calls it; it can also be used on its own.

## Build

    make                      # cc -O3 -mcpu=native -Wall -Wextra -> ./heil
    make ARCH=-march=native   # x86 / gcc

If the macOS linker fails with a `tapi` error about the SDK, point `cc` at an installed SDK:
`make CC="cc -isysroot $(xcrun --show-sdk-path)"` or an explicit path under
`/Library/Developer/CommandLineTools/SDKs/`.

## Command line

    heil V n [options]

| option | default | meaning |
|---|---|---|
| `V` | | `s`, `t` or `c` |
| `n` | | number of points (4..100) |
| `-g`, `--group G` | `C1` | symmetry group. `s`: C1 C2 C4 D1 D1d D2 D2d D4; `t`: C1 C3 D1 D3; `c`: Ck, Dk (k <= 12) |
| `-s`, `--seed S` | 1 | random seed |
| `-t`, `--time SECS` | 60 | search time (counted after the setup) |
| `-o`, `--out PREFIX` | `<V><n>` | the best configuration is written to `PREFIX.txt` |
| `-i`, `--init FILE` | | start configuration in site coordinates |
| `-w`, `--walk` | | hot walk from `--init` instead of ruin & recreate |
| `--vacancies K` | 0 | cold symmetric search where K generic orbits miss one image |
| `--free M` | 0 | cold symmetric search where M generic orbits keep only one point |

## Modes

| call | mode | engine |
|---|---|---|
| `heil V n -i SEED` | warm free RR: ruin & recreate on single points from a seed with n points | `free.c` |
| `heil V n -g G` | cold symmetric RR: every 30 s tree draws a random orbit structure and builds it greedily | `sym.c` |
| `heil V n -g G -i SEED` | warm symmetric RR from a G-symmetric seed; if the seed has m != n points, one or two whole orbits are added or removed first (best of all options) | `sym.c` |
| `heil s n -g D4 --vacancies K --free M` | cold symmetric RR in a defect class | `sym.c` |
| `heil V n [-g G] -i SEED --walk` | hot basin-hopping walk (C1 runs it on single points) | `sym.c` |

All modes run until the time is up and keep the best configuration found.

**How this relates to the algorithm.** The value (smallest triangle, or smallest triangle / hull area) is
maximised through a smooth soft-min of all triangle areas, sharpened in stages (beta 8 ... 8000) and
optimised with L-BFGS (`common.c`). Points stay in the region through their type (interior, on a side,
at a corner) and a quadratic penalty. The search around this polish is ruin & recreate: delete the most
stressed points (or orbits), re-insert them greedily at the best of many random candidates, polish softly,
accept if better or with probability 0.15. A new incumbent gets the sharp polish; in the free engine for `s`
and `t` it is then finished by the LP relocation of single points plus a min-norm ascent on the nearly active
triangles (`reloc.c`). The symmetric engine works on orbit representatives (generic, on a mirror axis, on a
side, centre, corner) so that every configuration it sees is exactly G-symmetric.

## Output

- `PREFIX.txt`: header line, then one `x y` line per point (`%.17g`, site coordinates: square [0,1]^2,
  triangle (0,0),(1,0),(0,1), convex unscaled with hull area 1). Written atomically on every improvement.
  - free engine: `# <V> n=<n> value=<value>`
  - symmetric engine: `# <V> n=<n> value=<value> group=<G> reps=<kinds>[ vacancies=<k>]` where the kinds
    are G (generic), A (axis), S (side), X (centre), K (corner); vacant images are not written.
- stdout: the final best value (`%.15g`).
- stderr: one line per improvement and a final summary.
- exit codes: 0 done, 1 usage or unknown group, 3 no orbit structure for this n and group, 4 seed problem
  (unreadable, wrong number of points, not G-symmetric, no way to reach n).

## Files

`heil.h` shared declarations; `main.c` command line; `common.c` triangle list, hull, soft-min, L-BFGS,
file I/O; `free.c` free engine; `sym.c` symmetric engine and walk; `reloc.c` LP relocation and
min-norm ascent.
