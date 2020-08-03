include(GoogleTest)

function(compile_driver) 
  set(options)
  set(oneValueArgs TEST)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable( ${ARG_TEST}_driver 
    ${CMAKE_SOURCE_DIR}/cpp_drivers/${ARG_TEST}_driver.cpp
    ${CMAKE_SOURCE_DIR}/cpp_drivers/TestMain.cpp
    )
  target_compile_features(${ARG_TEST}_driver PUBLIC cxx_std_17)
  target_include_directories(${ARG_TEST}_driver PUBLIC ${CMAKE_SOURCE_DIR}/wrappers/)
  target_include_directories(${ARG_TEST}_driver PUBLIC ${dawn4py_INCLUDE_DIR})
  target_include_directories(${ARG_TEST}_driver PUBLIC ${atlas_utils_INCLUDE_DIRS})
  target_link_libraries(${ARG_TEST}_driver PUBLIC ${ARG_TEST}_cxx-naive)  
  target_link_libraries(${ARG_TEST}_driver PUBLIC atlas)
  target_link_libraries(${ARG_TEST}_driver PUBLIC atlasUtilsLib)
  target_link_libraries(${ARG_TEST}_driver PUBLIC gtest)  
  # Ugly way to set the precision, TODO: reconsider the approach to setting the precision in dawn
  if (PRECISION STREQUAL "float")
      target_compile_definitions(${ARG_TEST}_driver PUBLIC DAWN_PRECISION=0)
  else()
      target_compile_definitions(${ARG_TEST}_driver PUBLIC DAWN_PRECISION=1)
  endif()

  install(TARGETS ${ARG_TEST}_driver)

  gtest_discover_tests(${ARG_TEST}_driver TEST_PREFIX "CodeGen::" DISCOVERY_TIMEOUT 30)

endfunction(compile_driver)

foreach(test ${dusk_tests})
  compile_driver(TEST ${test})  
endforeach()  