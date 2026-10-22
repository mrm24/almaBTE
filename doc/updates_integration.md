# Integration of the almaBTE updates branch

The fork improvements are by **Martí Raya-Moreno
<marti.raya.moreno@physik.hu-berlin.de>**. This integration preserves the fork's
history and combines `mrm24/almaBTE:updates` at `d87d7137` with prealma master
`480899d8` (21 October 2026). It includes the earlier `corrections` branch,
already merged into the fork's main. Four-phonon and GPU branch work is outside
this integration.

## Numerical features

- RTA and small-grain conductivity tensors respect crystal symmetry. The
  directional RTA API projects the same symmetrized tensor.
- First-Brillouin-zone mapping retains a fixed search origin, searches positive
  and negative lattice translations, and includes all equivalent boundary images.
- Polar calculations support Wang (default) and Gonze nonanalytic corrections.
  Both the dynamical-matrix API and spectrum API include the chosen correction.
  The derivative of Wang's normalized wavevector includes its inverse norm.
- Spectra include the complex Wigner velocity operator, in nm/ps, with the
  smooth phase convention of Phys. Rev. X **12**, 041011. All components use the
  final stored eigenvector basis. Scalar projection by the little group is not
  applied to off-diagonal band operators, which also transform in band space.
- `kappa_Tsweep` accepts `<computeCoherenceTerm/>`. It writes coherence to a
  separate output with suffix `_coherence`, using the same tensor/directional
  selection as the population result. Missing Wigner data is an error for this
  request, rather than an implicit zero result. Zero linewidth contributes zero
  between distinct frequencies. Degenerate blocks retain the existing population
  treatment; coherence couples distinct frequency subspaces. The population
  velocities retain their per-axis definition and need not equal the diagonal of
  the common-basis Wigner operator inside a degenerate block. Coherence invariance
  under rotations within a degenerate block assumes equal linewidths there;
  general unequal-linewidth covariance requires a matrix collision model.
- `phononinfo` adds absorption/emission phase space and weighted phase space,
  plus a small-grain conductivity summary. Phase-space normalization is applied
  once to each contribution. The mean-matrix-element diagnostic handles empty
  process tables without division by zero.
- Phonopy old/full/compact text force constants are accepted. Cartesian POSCAR
  coordinates include both the Angstrom-to-nm conversion and the POSCAR scale.

All three-phonon production paths retain the row-oriented SoA tables and prepared
scattering/collision kernels. A rank with no local processes is valid. Numerical
CSV data uses sufficient precision for double values, while readable engineering
labels and filenames retain their established formatting.

## Applications

`VCAbuilder` and `gruneisen` share their XML parser, material loading, alloy
mixing, grid construction and polar-method selection:

```sh
VCAbuilder input.xml [grid_density_override]
gruneisen input.xml [temperature_K]
```

The Gruneisen temperature defaults to 300 K and must be positive. It produces
mode-resolved dimensionless parameters and a heat-capacity-weighted mean. It uses
the same numerical library functions as the optional ShengBTE emulator and does
not enumerate scattering processes. Single crystals, alloys and parametric
alloys use the same input forms as VCAbuilder.

Builders accept `<broadening scale_factor="0.1"/>` and
`<nonanalytic_treatment method="wang"/>` or `method="gonze"`.

The `interfaces` executable reads an `<interface_conductance>` XML document and
computes transmission/conductance with constant transmission, DMM, or the
experimental pseudo-AMM model. Its library API checks direction, temperature,
and probability bounds. The AMM implementation retains its experimental status.

## Bulk HDF5 0.5 and migration

The process schema introduced in 0.4 is retained without reverting to process
objects: representatives, separate emission/absorption offsets, packed partner
indices, numeric columns and preparation flags. See [threeph_soa.md](threeph_soa.md).

Format 0.5 adds:

```
/qpoint_grid
    @wigner_available : uint8 scalar, 0 or 1
    re_wigner_v       : float64 [qpoint, cartesian, bra_band, ket_band]
    im_wigner_v       : float64 [qpoint, cartesian, bra_band, ket_band]
```

Both datasets are present exactly when the marker is 1. Their attributes specify
units (`nm / ps`), axes, and smooth phase convention. File floats are IEEE
little-endian. The flattened in-memory column is `bra + nbranches * ket`; this
is deliberately decoded into explicit band axes on disk. I/O transfers blocks,
not individual scalar matrix elements.

The reader accepts original 0.1/0.2 files, the fork's files labelled 0.2 with
flattened `[qpoint, cartesian, bra+nbranches*ket]` Wigner datasets, existing 0.4
files, and 0.5. Missing operators are represented explicitly. Such files remain
usable for population transport. The converter preserves an available operator
but cannot reconstruct missing off-diagonal information from frequencies and
group velocities; regenerate from force constants when coherence is required.

```sh
convert_bulk_hdf5 old.h5 new.h5
```

The converter requires a new output path. It retains IFC/scattering/user groups
and extra metadata nested inside core groups as well as at the root. It replaces
the old process schema with validated SoA tables. IFC subgroup helpers preserve
harmonic cell positions and matrices and third-order atomic indices, translations,
and all 27 Cartesian components. Their I/O failures are reported collectively.

## Validation

`updates_test` and `updates_mpi_test` exercise Wigner finite-difference derivatives,
Hermiticity and basis consistency, a two-mode coherence reference, phase and
exactly degenerate-subspace invariance, zero linewidth and unavailable data,
0.4/0.5/fork-legacy input, conversion and metadata preservation, malformed inputs
under MPI, IFC round trips, phase-space normalization, Cartesian/Phonopy formats,
Gonze commensurate reconstruction/acoustic translations, and interface/Gruneisen
limits. Existing regressions and SoA scalar-reference tests remain applicable.

The existing historical tabulations of CSV results have six significant digits;
comparison tolerances account for their rounding rather than reducing output
precision. Physical changes from corrected Brillouin-zone handling and polar
wavevector derivatives must be distinguished from storage/kernel changes when
comparing conductivities. The benchmark on an unchanged legacy Si file isolates
the latter and retains the original rates and iterative checksum.

The optional application checks use only the Python standard library and run via
uv:

```sh
uv run --no-project test/updates_cli_test.py BUILD_DIRECTORY SOURCE_DIRECTORY
```

An optional independent harmonic comparison uses Phonopy 4.5.0, h5py and NumPy:

```sh
uv run --with h5py --with phonopy==4.5.0 test/phonopy_reference_test.py BUILD_DIRECTORY SOURCE_DIRECTORY
```

On the InAs 3x3x3 check grid, Gonze dynamical eigenvalues agree with Phonopy to a
maximum scaled difference of approximately `1.2e-8` with matching masses and
physical constants. The Wang comparison retains almaBTE's established convention
of omitting the correction on first-BZ faces, where Phonopy applies it; its maximum
scaled difference on that grid is approximately `1.5e-4`. Analytic derivatives of
both almaBTE methods are separately checked against finite differences.

The corrected GaN small-grain `zz` reference is `2.09919747 W/(m K nm)`, replacing
`2.12192`. The direct/iterative beyond-RTA `zz` reference is approximately
`189.807 W/(m K)`, replacing `190.0217`. These changes follow the corrected BZ and
Wang derivative paths. The original Fortran reader's 2018 per-element-array fix
is retained, with a bounds-checked regression where the atom count exceeds the
element count; the fork's older snapshot must not reintroduce that overflow.
