add_executable(blackwell_llm 
	main.cpp 
	safetensors.cpp
	memory_pool.cpp)

target_link_libraries(blackwell_llm PRIVATE nlohmann_json::nlohmann_json blackwell_kernels)