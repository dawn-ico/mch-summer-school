include(FetchContent)

if(NOT FETCHED_DUSK)
  message(STATUS "Fetching dusk...")
endif()
FetchContent_Declare(dusk
  GIT_REPOSITORY git@github.com:dawn-ico/dusk.git
  GIT_TAG 8e52fe29406cbc5ed6fcda89893ee08238262036
)
FetchContent_GetProperties(dusk)
mark_as_advanced(FETCHED_DUSK)
FetchContent_MakeAvailable(dusk)
