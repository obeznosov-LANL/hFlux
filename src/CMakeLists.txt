add_library(hflux
  ${CMAKE_CURRENT_SOURCE_DIR}/hFlux.cpp)

add_library(hflux::hflux ALIAS hflux)

target_include_directories(hflux
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hflux PUBLIC Kokkos::kokkos)

# ------------------------------------------------------------
# Command-line tools
# ------------------------------------------------------------
add_executable(poincare_tool ${CMAKE_CURRENT_SOURCE_DIR}/poincare_main.cpp)
target_link_libraries(poincare_tool PRIVATE hflux::hflux m)
target_compile_features(poincare_tool PRIVATE cxx_std_20)

add_executable(dump_field ${CMAKE_CURRENT_SOURCE_DIR}/dump_field_main.cpp)
target_include_directories(dump_field PRIVATE ${CMAKE_SOURCE_DIR}/test)
target_link_libraries(dump_field PRIVATE hflux::hflux m)
target_compile_features(dump_field PRIVATE cxx_std_20)

include(GNUInstallDirs)

install(TARGETS hflux
    EXPORT hflux-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS poincare_tool dump_field
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/hFlux
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT hflux-targets
    FILE hflux-targets.cmake
    NAMESPACE hflux::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hflux
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/hflux-config.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/hflux-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hflux
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/cmake/hflux-config-version.cmake
    VERSION ${hflux_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        ${CMAKE_BINARY_DIR}/cmake/hflux-config.cmake
        ${CMAKE_BINARY_DIR}/cmake/hflux-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hflux
)

export(EXPORT hflux-targets
    FILE ${CMAKE_BINARY_DIR}/cmake/hflux-targets.cmake
    NAMESPACE hflux::
)
