cmake_minimum_required(VERSION 3.10)
project(mpc_standalone CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# acados core: points at the installed tree (same as $ACADOS_SOURCE_DIR
# used by the Python interface / this project's own VM env setup).
# Override with -DACADOS_SOURCE_DIR=... if it lives elsewhere.
if(NOT DEFINED ACADOS_SOURCE_DIR)
    if(DEFINED ENV{ACADOS_SOURCE_DIR})
        set(ACADOS_SOURCE_DIR $ENV{ACADOS_SOURCE_DIR})
    else()
        set(ACADOS_SOURCE_DIR "$ENV{HOME}/acados")
    endif()
endif()

# The codegen'd solver (control/mpc/ocp.py's build_ocp(), run once via
# Python -- this CMakeLists does not invoke Python itself, it expects the
# codegen to already exist). main.cpp's function names are tied to this
# specific build's content hash; regenerating with a different OCP
# structure changes the hash and main.cpp needs updating too (see its own
# header comment).
if(NOT DEFINED ACADOS_CODEGEN_DIR)
    set(ACADOS_CODEGEN_DIR
        "${CMAKE_SOURCE_DIR}/../../../codegen_ocp_lateral_mpc_kinematic_bicycle_a3fa00e4")
endif()
get_filename_component(ACADOS_CODEGEN_DIR "${ACADOS_CODEGEN_DIR}" ABSOLUTE)

if(NOT EXISTS "${ACADOS_CODEGEN_DIR}")
    message(FATAL_ERROR
        "ACADOS_CODEGEN_DIR does not exist: ${ACADOS_CODEGEN_DIR}\n"
        "Generate it first, from the repo root:\n"
        "  python3 -c \"from control.mpc.ocp import build_ocp; from acados_template import AcadosOcpSolver; AcadosOcpSolver(build_ocp())\"\n"
        "or pass -DACADOS_CODEGEN_DIR=<path> if it lives elsewhere.")
endif()

add_executable(mpc_standalone main.cpp)

target_include_directories(mpc_standalone PRIVATE
    ${ACADOS_SOURCE_DIR}/include
    ${ACADOS_SOURCE_DIR}/include/acados
    ${ACADOS_SOURCE_DIR}/include/blasfeo/include
    ${ACADOS_SOURCE_DIR}/include/hpipm/include
    ${ACADOS_CODEGEN_DIR}
)

target_link_directories(mpc_standalone PRIVATE
    ${ACADOS_SOURCE_DIR}/lib
    ${ACADOS_CODEGEN_DIR}
)

target_link_libraries(mpc_standalone PRIVATE
    acados_ocp_solver_ocp_lateral_mpc_kinematic_bicycle_a3fa00e4
    acados
    hpipm
    blasfeo
    m
)

# Bake the library search paths into the binary so it runs standalone --
# no LD_LIBRARY_PATH, no venv activation, no Python interface needed at
# runtime (the whole point of this executable, per M3's deployment-path
# check). --disable-new-dtags: modern linkers emit DT_RUNPATH by default,
# which does NOT propagate to a linked library's OWN transitive
# dependencies (libacados.so needs libhpipm/libblasfeo/libqpOASES_e,
# which would otherwise fail to resolve even with BUILD_RPATH set here).
# The older DT_RPATH tag does propagate transitively -- this flag is what
# makes the linker emit that instead.
target_link_options(mpc_standalone PRIVATE -Wl,--disable-new-dtags)
set_target_properties(mpc_standalone PROPERTIES
    BUILD_RPATH "${ACADOS_SOURCE_DIR}/lib;${ACADOS_CODEGEN_DIR}"
)
