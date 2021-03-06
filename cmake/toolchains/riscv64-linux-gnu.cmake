set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv)

if(NOT PPLNN_TOOLCHAIN_DIR)
    set(PPLNN_TOOLCHAIN_DIR "/usr")
    if(NOT EXISTS ${PPLNN_TOOLCHAIN_DIR})
        message(FATAL_ERROR "`PPLNN_TOOLCHAIN_DIR` not set.")
    endif()
elseif(NOT EXISTS ${PPLNN_TOOLCHAIN_DIR})
    message(FATAL_ERROR "`PPLNN_TOOLCHAIN_DIR`(${PPLNN_TOOLCHAIN_DIR}) not found")
endif()

set(CMAKE_ASM_COMPILER ${PPLNN_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-g++)
set(CMAKE_C_COMPILER ${PPLNN_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${PPLNN_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
