# User-facing executables. Both talk to the engine exclusively through the
# public API (include/blackwell/), inherited from blackwell_core's PUBLIC
# include dir -- no src/-internal headers.
add_executable(blackwell_llm main.cpp)
target_link_libraries(blackwell_llm PRIVATE blackwell_core)
blackwell_copy_dstorage_dlls(blackwell_llm)

add_executable(blackwell_bench benchmark.cpp)
target_link_libraries(blackwell_bench PRIVATE blackwell_core)
blackwell_copy_dstorage_dlls(blackwell_bench)
