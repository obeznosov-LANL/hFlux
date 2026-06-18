
# hFlux [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://obeznosov-lanl.github.io/hFlux/) 

hFlux -- lightweight toolkit for tokamak simulation code diagnostics.

<img src="https://github.com/obeznosov-LANL/hFlux/blob/main/docs/src/assets/readme.png" width="100%" title="Poincare plots">

# Key features
* Poloidal flux function.
* Continuous, smooth and divergence free magnetic field reconstructionk.
* Built-in divergence cleaning, input fields do not require to be divergence free.
* Second order differentials of flux function (for example $\nabla B$, $\nabla \times B$).
* Calculation of magnetic axis.
* Safety-factor for arbitrary flux surfaces.
* Field line calculations, supporting Poincare plot data output.
* C/C++ interface, Python and Julia Wrappers.

## Requirements

- **CMake ≥ 3.10**  
- A C++ compiler supporting **C++23**
- HDF5 1.12 or newer
- Kokkos 4.4.01
- Cuda 12.6.2 or newer if GPU support is required

Additionally Python interface requires:
- Numpy
- Scipy
- Matplotlib

You can verify your CMake version by running `cmake --version`. Make sure your compiler (e.g., GCC, Clang, MSVC) supports C++23.


## Build instructions
Create suggested directory structure:
```console
hFlux-project/
├── sources/
├── build/
├── install/
```
Clone the repository under `hFlux-project/sources`. From `hFlux-project/build` run:

```console
~$ ../../source/hFlux                           \
    -DCMAKE_BUILD_TYPE=Debug                    \
    -DCMAKE_INSTALL_PREFIX=../../install/hFlux  \
    -DKokkos_DIR=__path_to_KokkosConfig.cmake__ \
    -DHDF5_DIR=__path_to_hdf5_install_dir__
~$ make -j install
```

Now you can access library through Julia interface. Coherent C and C++ interfaces are in the works.
Python interface can be found under `hFlux-project/install/python`. To use hFlux in your python applicaions add the pervious directory to enviroment variable `PYTHONPATH` and import hFlux module, e.g.,

```console
~$ export PYTHONPATH="${PYTHONPATH}:__path_to_project__/hFlux-project/install/python"
~$ python
>>> import hFlux
```

# Release
O4754 hFlux was approved for Open-Source Assertion
