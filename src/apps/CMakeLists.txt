# User-facing executables -- the first true consumers of the COM-style DLL
# boundary: they include ONLY blackwell/iblackwell_engine.h, link the
# blackwell_core import library, and receive blackwell_core.dll (plus the
# DirectStorage runtime it loads) next to the exe.
add_executable(blackwell_llm main.cpp)
target_link_libraries(blackwell_llm PRIVATE blackwell_core)
blackwell_copy_runtime_dlls(blackwell_llm blackwell_core)
blackwell_copy_dstorage_dlls(blackwell_llm)

add_executable(blackwell_bench benchmark.cpp)
target_link_libraries(blackwell_bench PRIVATE blackwell_core)
blackwell_copy_runtime_dlls(blackwell_bench blackwell_core)
blackwell_copy_dstorage_dlls(blackwell_bench)
