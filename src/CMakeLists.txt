add_subdirectory(kernels)

# 🤖 Autonomous code-agent core (tree-sitter CST + shadow graph). Isolated from
# the CUDA inference engine; see src/agent/CMakeLists.txt. Default ON.
option(BUILD_AGENT_CORE "Build the autonomous-agent code-analysis core (tree-sitter based, no CUDA)" ON)
if(BUILD_AGENT_CORE)
    add_subdirectory(agent)
endif()

# 🧰 Autonomous agent sandbox / "OS" layer (VFS + subprocess + git worktree).
# Also isolated from CUDA; see src/agent_env/CMakeLists.txt. Default ON.
option(BUILD_AGENT_ENV "Build the autonomous-agent sandbox OS layer (filesystem/exec/git, no CUDA)" ON)
if(BUILD_AGENT_ENV)
    add_subdirectory(agent_env)
endif()

# 🤝 Autonomous agent ReAct orchestrator (LLM <-> agent_core <-> agent_env).
# The integration seam: the first target that links BOTH isolated subsystems.
# Still CUDA-free (talks to the model via the ILLMGenerator text interface), so
# it requires both agent libs. Default ON when those are built.
option(BUILD_AGENT_ORCHESTRATOR "Build the autonomous-agent ReAct orchestrator (no CUDA)" ON)
if(BUILD_AGENT_ORCHESTRATOR AND BUILD_AGENT_CORE AND BUILD_AGENT_ENV)
    add_subdirectory(agent_orchestrator)
endif()

# ==============================================================================
# 1. ДВИЖОК И ИСПОЛНЯЕМЫЕ ФАЙЛЫ
# ==============================================================================
add_subdirectory(core)   # blackwell_core (см. src/core/CMakeLists.txt)
add_subdirectory(audio)  # blackwell_audio (Ultravox projector + prompt-injector kernels)
add_subdirectory(audio_rt) # blackwell_audio_rt (lock-free RT audio primitives; header-only)
add_subdirectory(vad)    # blackwell_vad (Silero neural VAD, ONNXRuntime CPU, CUDA-free)
add_subdirectory(tts)    # blackwell_tts (speech synthesis text frontend; CUDA-free)
add_subdirectory(bridge) # blackwell_bridge (app<->engine SPSC audio + prompt splice + C-ABI)

# ☁️ Cloud leg of the Local Router pattern: the Anthropic Messages API client
# the EOS commit gate dispatches to. The option is declared at the ROOT with the
# other feature switches (it defaults ON, and the prerequisite that default
# carries is documented there); this is only where it is consumed. The commit
# gate itself (src/bridge/intent_commit.hpp) is header-only and always built, so
# the local pipeline stays testable with this OFF.
if(BUILD_CLOUD_CLIENT)
    include(${CMAKE_SOURCE_DIR}/cmake/CloudDeps.cmake)
    blackwell_setup_cloud_deps()
    add_subdirectory(cloud)  # blackwell_cloud (CUDA-free, engine-free)
endif()

add_subdirectory(apps)   # blackwell_llm, blackwell_bench

# ==============================================================================
# 4. СТАНДАЛОН-ЭКСПЕРИМЕНТЫ (см. src/experiments/CMakeLists.txt)
# ==============================================================================
# The nvCOMP compression experiment (blackwell_compress) was removed together
# with the vendored external/nvcomp binaries; experiments now live in their own
# subdirectory with per-target scripts.
add_subdirectory(experiments)

# ==============================================================================
# 5. AGENT SIMULATOR / PROMPT PLAYGROUND (live LLM testing GUI)
# ==============================================================================
# A local-HTTP-server GUI for debugging the XML ReAct protocol AND running real
# inference. It links blackwell_core (CUDA engine + tokenizer) and the ReAct
# orchestrator, so it is gated on the orchestrator stack being built.
option(BUILD_AGENT_PLAYGROUND "Build the GUI Agent Simulator / Prompt Playground tool" ON)
if(BUILD_AGENT_PLAYGROUND AND BUILD_AGENT_ORCHESTRATOR AND BUILD_AGENT_CORE AND BUILD_AGENT_ENV)
    add_subdirectory(tools/playground)
endif()

# ==============================================================================
# 6. POC: input-hook / UIA-caret / D2D-overlay OS pipeline
# ==============================================================================
# Isolated feasibility PoC for the overlay-translator UI pipeline. No CUDA, no
# blackwell_core, no tokenizer -- just Win32 + COM + UI Automation + Direct2D.
# See src/tools/poc_overlay/CMakeLists.txt.
option(BUILD_POC_OVERLAY "Build the input-hook/UIA-caret/D2D-overlay OS pipeline PoC" ON)
if(BUILD_POC_OVERLAY AND WIN32)
    add_subdirectory(tools/poc_overlay)
endif()