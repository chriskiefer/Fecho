cmake_minimum_required (VERSION 2.6)
set(EXT_NAME fechopd)
project (${EXT_NAME})
set(FECHO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(CMAKE_OSX_ARCHITECTURES $(ARCHS_STANDARD_32_64_BIT))
set(ARMA_USE_LAPACK    false)
set(ARMA_USE_BLAS      false)

if(APPLE)
  set(ARMA_OS macos)
  
  set(ARMA_USE_LAPACK true)
  set(ARMA_USE_BLAS   true)
  
  ## Under MacOS, the current version of ARMA_FindCLAPACK can get confused between two incompatible
  ## versions of "clapack.h" (one provided by the system and one provided by ATLAS).
  ## As such, use of ATLAS under MacOS is disabled for now.
  
else()
  set(ARMA_OS unix)
  
  include(ARMA_FindMKL)
  include(ARMA_FindACMLMP)
  include(ARMA_FindACML)
  
  message(STATUS "   MKL_FOUND   = ${MKL_FOUND}")
  message(STATUS "ACMLMP_FOUND   = ${ACMLMP_FOUND}")
  message(STATUS "  ACML_FOUND   = ${ACML_FOUND}")
  
  if(MKL_FOUND OR ACMLMP_FOUND OR ACML_FOUND)
    
    set(ARMA_USE_BLAS   true)
    set(ARMA_USE_LAPACK true)
    
    message(STATUS "")
    message(STATUS "*** If the MKL or ACML libraries are installed in")
    message(STATUS "*** /opt or /usr/local, make sure the run-time linker can find them.")
    message(STATUS "*** On Linux systems this can be done by editing /etc/ld.so.conf")
    message(STATUS "*** or modifying the LD_LIBRARY_PATH environment variable.")
    message(STATUS "*** On systems with SELinux enabled (eg. Fedora, RHEL),")
    message(STATUS "*** you may need to change the SELinux type of all MKL/ACML libraries")
    message(STATUS "*** to fix permission problems that may occur during run-time.")
    message(STATUS "*** See README.txt for more information")
    message(STATUS "")
    
  else()
    
    include(ARMA_FindLAPACK)
    include(ARMA_FindOpenBLAS)
    include(ARMA_FindBLAS)
    include(ARMA_FindCLAPACK)
    include(ARMA_FindCBLAS)
    
    message(STATUS "  LAPACK_FOUND = ${LAPACK_FOUND}")
    message(STATUS "    BLAS_FOUND = ${BLAS_FOUND}")
    message(STATUS "OpenBLAS_FOUND = ${OpenBLAS_FOUND}")
    message(STATUS " CLAPACK_FOUND = ${CLAPACK_FOUND}")
    message(STATUS "   CBLAS_FOUND = ${CBLAS_FOUND}")
    
    if(LAPACK_FOUND)
      set(ARMA_USE_LAPACK true)
    endif()
    
    if(BLAS_FOUND)
      set(ARMA_USE_BLAS true)
    endif()
    
    if(OpenBLAS_FOUND AND CLAPACK_FOUND AND CBLAS_FOUND)
      message(STATUS "")
      message(STATUS "*** WARNING: both OpenBLAS and ATLAS have been found; ATLAS will not be used")
    endif()
    
    if(OpenBLAS_FOUND)
      
      set(ARMA_USE_BLAS true)
      
      message(STATUS "")
      message(STATUS "*** If the OpenBLAS library is installed in")
      message(STATUS "*** /usr/local/lib or /usr/local/lib64")
      message(STATUS "*** make sure the run-time linker can find it.")
      message(STATUS "*** On Linux systems this can be done by editing /etc/ld.so.conf")
      message(STATUS "*** or modifying the LD_LIBRARY_PATH environment variable.")
      message(STATUS "")
      
    else()
      
      if(CLAPACK_FOUND AND CBLAS_FOUND)
        message(STATUS "CLAPACK_INCLUDE_DIR = ${CLAPACK_INCLUDE_DIR}")
        message(STATUS "  CBLAS_INCLUDE_DIR = ${CBLAS_INCLUDE_DIR}")
        
        if(${CLAPACK_INCLUDE_DIR} STREQUAL ${CBLAS_INCLUDE_DIR})
          set(ARMA_USE_ATLAS true)
          set(ARMA_ATLAS_INCLUDE_DIR ${CLAPACK_INCLUDE_DIR})
        endif()
      endif()
      
    endif()
    
  endif()

endif()


if(APPLE)
  set(ARMA_LIBS ${ARMA_LIBS} "-framework Accelerate")  # or "-framework accelerate" ?
  message(STATUS "MacOS X detected. Added '-framework Accelerate' to compiler flags")
  
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  message(STATUS "Clang compiler on MacOS X detected. Added '-stdlib=libc++' to compiler flags")
  endif()

endif()

message(STATUS "")
message(STATUS "*** The run-time library component of Armadillo will use the following libraries:")
message(STATUS "*** ARMA_LIBS = ${ARMA_LIBS}")
message(STATUS "")


include_directories(${FECHO_SRC})

set(PD_SRC /Users/chris/src/pd-extended/pd/src)
include_directories(${PD_SRC})

file(GLOB ARMA_SRC
    "${FECHO_SRC}/armadillo_bits/*.hpp"
)
add_library(${EXT_NAME} SHARED 
	fecho-pd.cpp
	${PD_SRC}/m_pd.h
	${FECHO_SRC}/armadillo
	${FECHO_SRC}/fecho.hpp
	${FECHO_SRC}/fechoActivations.hpp
	${FECHO_SRC}/fechoUtil.hpp
	${ARMA_SRC}
)

target_link_libraries( ${EXT_NAME}  ${ARMA_LIBS} )

source_group("armadillo" FILES ${ARMA_SRC})
source_group("fecho" FILES 
	${FECHO_SRC}/armadillo
	${FECHO_SRC}/fecho.hpp
	${FECHO_SRC}/fechoActivations.hpp
	${FECHO_SRC}/fechoUtil.hpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
set_target_properties(${EXT_NAME} 
	PROPERTIES
	LINK_FLAGS "-undefined dynamic_lookup"
	LIBRARY_OUTPUT_DIRECTORY_DEBUG "/Library/Pd/"
	LIBRARY_OUTPUT_DIRECTORY_RELEASE "/Library/Pd/"
	LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "/Library/Pd/"
	LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "/Library/Pd/"
)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET ${EXT_NAME} PROPERTY PREFIX "")
    set_property(TARGET ${EXT_NAME} PROPERTY OUTPUT_NAME "fecho.pd_darwin")
    set_property(TARGET ${EXT_NAME} PROPERTY SUFFIX "")
    set_property(TARGET ${EXT_NAME} PROPERTY SOVERSION "0.1")
endif()



