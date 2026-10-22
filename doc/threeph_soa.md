# Three-phonon storage and bulk HDF5 0.5

Copyright 2026 The ALMA Project Developers

Three-phonon processes are owned by `alma::Threeph_processes`, not a vector
of process objects. The layout follows the row-based SoA organization in
ShengBTE trunk (`master`, reference revision
`1f158a8e3918133117145c4ca22099a0a64f8e0b`), retaining almaBTE's Gaussian
broadening, process multiplicities, units, and numerical equations.

## Storage and views

Each rank owns a contiguous range of incoming modes. Global row `r` denotes
irreducible class `r / nbranches` and branch `r % nbranches`. Emission and
absorption have separate tables. Each table contains terminal-inclusive,
zero-based `uint64` row offsets and contiguous partner and numeric columns.
A partner is encoded as `q * nbranches + branch`. The partner columns use
`uint32` when the entire grid fits; larger grids use `uint64`, with checked
arithmetic and no silent truncation.

```cpp
auto processes = alma::find_allowed_threeph(grid, comm);
for (auto p : processes) {
    p.compute_vp2(cell, grid, thirdorder); // writes the owning columns
}
const auto& read_only = processes;
for (auto p : read_only) {
    // p.q, p.alpha, p.c, p.type and read-only computations are available.
}
```

A view is returned by value. Copying it preserves its connection to the
owner; use `auto p` or `auto&& p`, rather than `auto& p`. Metadata is decoded
into the short-lived view, while computed values remain in the columns.
`to_value()` creates an independent `Threeph_process` snapshot. Conversion to
an owning value is also available for value-oriented standard algorithms.
The old scalar value class remains a reference implementation and an API
boundary type; no vector of those values is stored by production paths.

Views are invalidated by structural changes, moving, replacing or destroying
the owner. Const views never write through to its caches. Iterators traverse
emission, then absorption, in row order, and advance over row boundaries
without a binary search for every process. They support input iteration;
reordering algorithms such as `std::sort` are deliberately unsupported.

To construct a table manually, append processes in nondecreasing incoming-row
order within each sign and call `finalize()` once before accessing it. Columns
installed through `set_table()` are validated immediately. Empty rows and
ranks with no assigned work are valid.

`Gamma_grid::equivalent_qpoints()` borrows a const reference when called on
an lvalue grid. That reference remains valid while the grid is alive and
unmoved. Calling it on a temporary returns an owning vector instead; plain
`auto` also makes an owning copy from an lvalue.

## Kernels and temperature lifetime

`Threeph_scattering` prepares the temperature-independent rate prefactors.
Its `rates(T)` evaluates Bose occupations once per full-grid mode and returns
separate rate arrays for the two signs. It can be reused across a temperature
sweep. The grid and processes must remain alive and unchanged while a
prepared object is in use; rebuild it after changing frequencies or matrix
elements.

`Threeph_collision` prepares rates for one temperature and groups candidate
symmetry operations by incoming image. `apply(F)` computes the local
three-phonon collision contribution; the caller performs the MPI reduction.
Alternate operations are retained when a partner image is unavailable on a
non-cubic grid. This preserves the old first-valid-operation rule.

The iterative solver prepares this operator once per temperature. Its hot
loop reads only two partner indices and one rate per process (16 bytes with
32-bit indices, excluding row metadata). It does not reconstruct process
objects, copy spectra, compute occupations, or allocate per-process maps.
Rows update disjoint output modes and can run with OpenMP. Set
`OMP_NUM_THREADS` to control threads, or configure `-DALMA_OPENMP=OFF` for a
serial build. This change does not replace the separate dense beyond-RTA
solver or the storage of elastic processes.

## HDF5 process columns (0.4 and 0.5)

Root attributes remain `major_version = 0`, `minor_version = 5`, `description`,
and `timestamp`. Crystal, symmetry, harmonic and scattering datasets retain
their meanings. The process group contains:

```
/threeph_processes
    @mode_encoding = "q*nbranches+branch; zero-based"
    @row_encoding  = "class*nbranches+branch; zero-based"
    representatives : uint64 [nclasses]
    emission/
        offsets     : uint64 [nclasses*nbranches + 1]
        second      : uint32 or uint64 [nprocesses]
        third       : uint32 or uint64 [nprocesses]
        domega      : float64 [nprocesses]
        sigma       : float64 [nprocesses]
        gaussian    : float64 [nprocesses]
        vp2         : float64 [nprocesses]
        flags       : uint8   [nprocesses]
    absorption/
        ... identical columns, independently sized ...
```

Fixed-width file types are little-endian. `domega` and `sigma` are in rad/ps,
`gaussian` in ps/rad, and `vp2` in THz^4/(kg nm^2), as before. In `flags`,
bit 0 marks a computed Gaussian and bit 1 marks a computed matrix element.
Unset values are initialized to zero and must not be interpreted as computed
zeroes. Thus both partially prepared and completed process tables can be
round-tripped. Temperature-dependent rates are not persisted.

The reader accepts **0.1, 0.2, 0.4 and 0.5**, checking both version attributes. Other
versions are rejected explicitly. Version 0.4 replaces the 0.2 process schema;
it is not an additive minor revision. Old readers that ignore the minor
version cannot read 0.4 files.

The legacy 0.1/0.2 reader converts columns directly into SoA storage in two
bounded passes, preserving each row's process order. It never creates a global AoS
array. On input, whole incoming rows are distributed by cumulative process
counts, independently of the MPI rank count that wrote the file. Empty tables
fall back to an even row distribution. The 0.4 reader validates dimensions,
unsigned index types, offsets, column lengths, flags, representatives and
index bounds before computation. Root I/O errors are communicated before the
next MPI transfer so all ranks report invalid input.

I/O uses the serial HDF5 library on rank 0 and hyperslabs of up to 65,536
processes per column. Legacy redistribution is also chunked. Native-format
MPI transfers currently move one rank's columns at a time, so rank 0 may hold
its own partition and one additional partition; parallel HDF5 is not required.

Convert an existing file without recomputing matrix elements:

```sh
build/src/convert_bulk_hdf5 old.h5 new.h5
```

The output must be a new filename. The converter preserves additional
scattering/user objects and non-version root attributes. It can also be run
under MPI; old files remain readable directly without prior conversion.

## Wigner velocities in 0.5

Version 0.5 retains these process columns and adds an explicit Wigner-availability
marker and complex band-pair velocity datasets under `/qpoint_grid`. Original
legacy files without this operator still support population transport. Conversion
preserves available Wigner data and user metadata but cannot synthesize missing
coherence information. See [updates_integration.md](updates_integration.md) for the
complete layout, legacy fork format, phase conventions and validation.

## Dependencies and checks

The build requires CMake 3.25+, C++17+, Boost 1.75+, MPI, HDF5 and spglib.
Installed spglib is discovered through its CMake package, pkg-config or a
header/library search. `-DALMA_FETCH_SPGLIB=ON` downloads and builds the pinned
2.7.0 release if no installation is found. There is no bundled spglib archive
or autotools build. Use `CMAKE_PREFIX_PATH` for a non-system installation.

`threeph_soa_test` compares SoA rates and collision applications against the
scalar process equations, exercises views and packed indices, checks MPI and
HDF5 round trips, covers partially prepared/empty tables and zero-work ranks,
and verifies collective rejection of malformed files. CTest also registers a
two-rank execution.

Build the optional benchmark and compare the same fixture, compiler, build
type, MPI ranks and thread count against the pre-migration revision:

```sh
cmake --build build --target benchmark_threeph
OMP_NUM_THREADS=1 build/src/benchmark_threeph material.h5 5 reference.txt
```

The benchmark reports load time, RTA time, iteration preparation time, time
per iteration and numerical checksums. Its iterative calculation intentionally
omits elastic processes to isolate the three-phonon path. The optional text
output contains every RTA rate followed by the conductivity tensor after the
requested iteration count. Storage and kernel changes must be assessed
separately from changes to numerical integration.

For the bundled Si fixture (175,718 processes), a single-rank, single-thread
RelWithDebInfo comparison with three repetitions measured:

| Operation | Original AoS | SoA |
| --- | ---: | ---: |
| RTA evaluation | 21.7 ms | 1.38 ms |
| Iterative step | 270 ms | 49.1 ms |
| Legacy file load | 1.30 s | 40.8 ms |
| Native 0.4 file load | — | 31.8 ms |

The baseline was revision `7f6c01b8`, with its spglib adapter rebuilt against
the same spglib 2.7 library as the new code. All RTA rates and conductivity
tensor entries agreed within a maximum scaled difference of `5.4e-15`.
These timings include the accompanying caching, allocation and I/O changes;
they do not measure the storage layout in isolation or predict larger-grid
performance.
