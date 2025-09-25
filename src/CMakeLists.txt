add_library(hflux
  ${CMAKE_CURRENT_SOURCE_DIR}/hFlux.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/c_wrapper.cpp)

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

include(GNUInstallDirs)

install(TARGETS hflux
    EXPORT hflux-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
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

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/hFlux.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/python
)

export(EXPORT hflux-targets
    FILE ${CMAKE_BINARY_DIR}/cmake/hflux-targets.cmake
    NAMESPACE hflux::
)
