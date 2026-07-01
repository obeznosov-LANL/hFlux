# Quickstart

## Native C++ Build

hFlux's native implementation is a C++20 library using Kokkos and HDF5.

```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If dependencies are provided through Spack, use the repository Spack environment before configuring:

```bash
spack -e . concretize
spack -e . install
spack env activate .
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Interfaces

The C++ library is the native implementation.

The Python implementation under `src/python/` is a separate pure-JAX implementation. It follows the C++ interpolation logic and does not call the compiled shared library.

A standalone Julia implementation is planned. It will be self-contained rather than a wrapper around the compiled shared library.

## Python/JAX Example

Run the analytic-field example from the repository root:

```bash
PYTHONPATH=src/python/src python src/python/examples/plot_analytic_field.py \
  --nR 100 --nZ 200 --outdir analytic_plots
```

This builds the analytic test field, interpolates it, computes flux, cleans divergence, finds the magnetic axis, normalizes flux, and writes plots to `analytic_plots/`.

The interpolation layer accepts any number of field components in an array with shape `(nR_data, nZ_data, ncomp)`. The magnetic-field helpers use the `component0` argument to identify a consecutive three-component magnetic-field block:

```text
data[..., component0 + 0] = R * B_R
data[..., component0 + 1] = R * B_phi
data[..., component0 + 2] = R * B_Z
```

For the default `component0=0`, this is simply `(R * B_R, R * B_phi, R * B_Z)` in components `0`, `1`, and `2`.

For example, assume that the first component of the magnetic field, $R B_R$, occupies the 7th component of the input array. Since Python indexing is zero-based, this means the magnetic-field block starts at `component0=6`:

```text
data[..., 6] = R * B_R
data[..., 7] = R * B_phi
data[..., 8] = R * B_Z
```

Then flux computation and divergence cleaning can be called as follows:

```python
field = field.compute_flux(component0=6)
field = field.clean_divergence(component0=6)
```

Only `compute_flux` and `clean_divergence` require this three-component convention. Plain interpolation and `eval_field` preserve and return all components.
