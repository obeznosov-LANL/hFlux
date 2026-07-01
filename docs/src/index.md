# hFlux

hFlux -- lightweight toolkit for tokamak simulation code diagnostics.

![Poincare plots image](assets/index.png)

# Key features

* Poloidal flux function construction.
* Continuous, smooth, divergence-free magnetic field reconstruction.
* Built-in divergence cleaning; input fields do not need to be divergence-free.
* Second-order differentials of flux function, for example $\nabla B$ and $\nabla \times B$.
* Magnetic-axis calculation.
* Safety-factor calculation for arbitrary flux surfaces.
* Field-line calculations supporting Poincare plot data output.
* C++ core implementation with Kokkos.
* Separate pure-JAX Python implementation.
* Future standalone Julia implementation.

## Requirements

- CMake >= 3.10
- A C++ compiler supporting C++20
- Kokkos
- HDF5

You can verify your CMake version by running `cmake --version`.


## Build instructions

```console
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The native implementation is the C++ library.

The Python implementation under `src/python/` is a separate pure-JAX implementation. It follows the C++ interpolation logic but does not call the compiled shared library.

A standalone Julia implementation is planned. It will be self-contained rather than a wrapper around the compiled shared library.

# Release
O4754 hFlux was approved for Open-Source Assertion
