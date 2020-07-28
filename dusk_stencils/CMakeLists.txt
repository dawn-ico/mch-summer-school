set(STENCIL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/dusk_stencils)

function(generate_codegen_target)
  set(options)
  set(oneValueArgs TEST BACKEND)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Add make target
  set(source_file ${CMAKE_SOURCE_DIR}/dusk_stencils/${ARG_TEST}.py)
  set(output_dir ${CMAKE_BINARY_DIR}/dusk_stencils/generated)
  file(MAKE_DIRECTORY ${output_dir})
  set(generated_file ${output_dir}/${ARG_TEST}_${ARG_BACKEND}.cpp)
  add_custom_target(${ARG_TEST}_${ARG_BACKEND}_codegen DEPENDS ${generated_file})  
  add_custom_command(OUTPUT ${generated_file}
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/calldusk.sh ${ARG_BACKEND} ${source_file} ${generated_file}
    DEPENDS ${source_file}
  )

endfunction()

function(compile_naive_target)
  set(options)
  set(oneValueArgs TEST)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_library( ${ARG_TEST}_cxx-naive
    ${CMAKE_SOURCE_DIR}/wrappers/${ARG_TEST}_wrapper.h 
    ${CMAKE_SOURCE_DIR}/wrappers/${ARG_TEST}_cxx-naive_wrapper.cpp
  )
  target_compile_features(${ARG_TEST}_cxx-naive PUBLIC cxx_std_14)
  target_include_directories(${ARG_TEST}_cxx-naive PUBLIC ${CMAKE_SOURCE_DIR}/wrappers/)
  target_include_directories(${ARG_TEST}_cxx-naive PUBLIC ${CMAKE_BINARY_DIR}/dusk_stencils/)
  target_include_directories(${ARG_TEST}_cxx-naive PUBLIC ${dawn4py_INCLUDE_DIR})
  target_link_libraries(${ARG_TEST}_cxx-naive PUBLIC atlas)

  if (PRECISION STREQUAL "float")
      target_compile_definitions(${ARG_TEST}_cxx-naive PUBLIC DAWN_PRECISION=0)
  else()
      target_compile_definitions(${ARG_TEST}_cxx-naive PUBLIC DAWN_PRECISION=1)
  endif()

  add_dependencies(${ARG_TEST}_cxx-naive ${ARG_TEST}_cxx-naive_codegen)
endfunction()

function(compile_cuda_target)
  set(options)
  set(oneValueArgs TEST LANG)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set_source_files_properties( ${CMAKE_BINARY_DIR}/dusk_stencils/generated/${ARG_TEST}_cuda.cpp PROPERTIES LANGUAGE CUDA )
  
  if ("${ARG_LANG}" STREQUAL "cpp")
    set_source_files_properties(${CMAKE_SOURCE_DIR}/wrappers/${ARG_TEST}_cuda_wrapper.cpp PROPERTIES LANGUAGE CUDA )
  endif()

  if ("${ARG_LANG}" STREQUAL "cpp")
    list(APPEND source_files ${CMAKE_SOURCE_DIR}/wrappers/${ARG_TEST}_cuda_wrapper.cpp ${CMAKE_SOURCE_DIR}/wrappers/${ARG_TEST}_wrapper.h)
  elseif("${ARG_LANG}" STREQUAL "fortran")
    list(APPEND source_files ${CMAKE_BINARY_DIR}/dusk_stencils/generated/${ARG_TEST}_cuda.cpp)
  else()
    message( FATAL_ERROR "compile cuda target only accepts cpp or fortran, CMake will exit." )
  endif() 
   
  add_library(${ARG_TEST}_cuda ${source_files}) 

  set_target_properties(${ARG_TEST}_cuda PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/binding")

  target_compile_features(${ARG_TEST}_cuda PUBLIC cxx_std_14)
  target_include_directories(${ARG_TEST}_cuda PUBLIC ${CMAKE_SOURCE_DIR}/wrappers/)
  target_include_directories(${ARG_TEST}_cuda PUBLIC ${CMAKE_BINARY_DIR}/dusk_stencils/)
  target_include_directories(${ARG_TEST}_cuda PUBLIC ${dawn4py_INCLUDE_DIR})
  target_link_libraries(${ARG_TEST}_cuda PUBLIC atlas)

  if (PRECISION STREQUAL "float")
      target_compile_definitions(${ARG_TEST}_cuda PUBLIC DAWN_PRECISION=0)
  else()
      target_compile_definitions(${ARG_TEST}_cuda PUBLIC DAWN_PRECISION=1)
  endif()

  target_compile_options(${ARG_TEST}_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_ARCH}>)

  add_dependencies(${ARG_TEST}_cuda ${ARG_TEST}_cuda_codegen) 
endfunction()

# both backends for dusk tests
foreach(test ${dusk_tests})
  generate_codegen_target(TEST ${test} BACKEND cxx-naive) 
  generate_codegen_target(TEST ${test} BACKEND cuda)  

  compile_naive_target(TEST ${test})
  compile_cuda_target(TEST ${test} LANG cpp)
endforeach()  