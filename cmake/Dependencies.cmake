# Third-party dependencies fetched at configure time. Pinned release tags only
# (never branches); TLS verification stays ON for every fetch.
include(FetchContent)

# GoogleTest: MSVC adaptation -- no pthreads probing, and the same DLL/MD CRT
# as the rest of the VS2022 build.
set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# nlohmann/json: Safetensors header parsing + config loading.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)
