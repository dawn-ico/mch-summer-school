include(FetchContent)

if(NOT FETCHED_DUSK)
  message(STATUS "Fetching dusk...")
endif()
FetchContent_Declare(dusk
  GIT_REPOSITORY git@github.com:dawn-ico/dusk.git
  GIT_TAG 480a993a115eb831c92c1e5cc19fec536a03ec4e 
)
FetchContent_GetProperties(dusk)
mark_as_advanced(FETCHED_DUSK)
FetchContent_MakeAvailable(dusk)
