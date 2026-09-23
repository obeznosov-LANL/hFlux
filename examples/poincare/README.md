# Poincare example

End-to-end workflow for reconstructing a 3D magnetic field from sampled `B`
values, tracing field lines, and producing a Poincare section plus a
divergence-cleaned field dump on a fine mesh.

## Executables

Built from `src/` (see top-level `CMakeLists.txt`); binaries land in
`build/src/`.

- `dump_field <out.txt>` — evaluates the resonant analytic field on the
  hardcoded grid (NR=64, NZ=128, Nphi=9; `q0=1.98`, resonant `m=2/n=1`) and
  writes it in the B-field input format below.
- `poincare_tool <bfield_file> <seed_file> <output_txt>` — reads the B-field
  and seed files, reconstructs the field (Fourier + Hermite interpolation,
  divergence cleaning), traces each seed, and writes:
  - `<output_txt>` — the Poincare section.
  - `<output>.fine.txt` — the reconstructed field on the `fourier_test` fine
    mesh (160 x 320 x 7, phi offset 0.37).

Interpolation order parameters are fixed at compile time (`m=2`, `swidth=7`),
matching the tests.

## File formats

### B-field input (`dump_field` output / `poincare_tool` input)

```
NR NZ Nphi R0 Z0 dR dZ            # header line
B_R B_phi B_Z                     # NR*NZ*Nphi lines; i (R) outer, j (Z), iphi inner
...
```

Values are physical `B` components at `(R0 + i*dR, Z0 + j*dZ, iphi*2pi/Nphi)`.
`poincare_tool` multiplies by `R` internally.

### Seed file (`poincare_tool` input)

```
n_turn                            # integer toroidal turns per seed
R Z                               # one seed per line
...
```

### Poincare output

```
# trace turn R Z
<trace> <turn> <R> <Z>
```

### Fine-mesh field output (`<output>.fine.txt`)

```
# R Z phi B_R B_phi B_Z
<R> <Z> <phi> <B_R> <B_phi> <B_Z>
```

## Plotting

Both scripts take the data file as an argument (uses matplotlib + numpy).

```
python plot_poincare.py <poincare_file> [-o out.png]
python plot_field.py    <fine_file>     [--phi-index N] [-o out.png]
```

`plot_field.py` draws `B_R`, `B_phi`, `B_Z` on the R-Z plane at one toroidal
plane, with contour lines overlaid.

## Full workflow

```sh
cmake --build build

# 1. Generate a sample B-field in the input format.
./build/src/dump_field field.txt

# 2. Define seeds: 1000 turns from three starting points.
printf "1000\n3.0 0.0\n3.05 0.0\n3.1 0.0\n" > seeds.txt

# 3. Trace and dump.
./build/src/poincare_tool field.txt seeds.txt out.txt

# 4. Plot.
python plot_poincare.py out.txt
python plot_field.py out.fine.txt
```
