# ==============================================================================
# agent_orchestrator: the ReAct (Reason + Act) state machine that ties the LLM
# engine together with agent_core (AST/semantics) and agent_env (sandbox/exec).
#
# This is the INTEGRATION seam. agent_core and agent_env are each deliberately
# isolated from the CUDA inference engine and from one another; this target is
# the first that links BOTH of them. It stays GPU-agnostic too -- it talks to the
# model only through the ILLMGenerator text interface, so nothing here pulls in
# blackwell_core / blackwell_kernels or CUDA. That keeps the orchestrator fully
# unit-testable with a MockLLM and lets us swap the underlying LLM architecture
# without touching the loop, the parser or the tools.
# ==============================================================================

add_library(agent_orchestrator STATIC
    tool_parser.cpp
    tool_registry.cpp
    orchestrator.cpp
    default_tools.cpp
)

target_compile_features(agent_orchestrator PUBLIC cxx_std_20)
target_include_directories(agent_orchestrator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# The default_tools bridge is what justifies linking both subsystems here.
target_link_libraries(agent_orchestrator PUBLIC agent_core agent_env)
