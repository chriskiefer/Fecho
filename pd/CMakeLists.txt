cmake_minimum_required (VERSION 2.6)
set(EXT_NAME fechopd)
project (${EXT_NAME})
set(FECHO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(CMAKE_OSX_ARCHITECTURES $(ARCHS_STANDARD_32_64_BIT))
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



