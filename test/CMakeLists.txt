set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)
include("Utils")

add_hFlux_test(hermite_test
  SRCS
    hermite_test.cpp
  INPUT
    hermite_test.input
)

add_hFlux_test(hermite_test_c
  SRCS
    hermite_test.c
  INPUT
    hermite_test.input
)

add_hFlux_test(poincare_test_c
  SRCS
    poincare_test.c
  INPUT
    hermite_test.input
)

find_package(HDF5 COMPONENTS C CXX)
target_link_libraries(hermite_test PUBLIC HDF5::HDF5)
target_link_libraries(hermite_test PUBLIC Kokkos::kokkos)
