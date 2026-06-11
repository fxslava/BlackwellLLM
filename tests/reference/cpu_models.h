#pragma once
// Compatibility shim. The CPU reference implementations moved to
// tests/common/cpu_reference.h during the test-suite restructuring; this
// header stays because src/experiments/compress_experiment.cu includes it by
// relative path and the root CMakeLists.txt exposes tests/reference/ on the
// global include path. New code should include "common/cpu_reference.h".
#include "../common/cpu_reference.h"
