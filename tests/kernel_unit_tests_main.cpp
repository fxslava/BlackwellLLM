// =============================================================================
// kernel_unit_tests — consolidated entry point for the audio-frontend per-kernel
// parity checks. Merges what used to be two separate micro-executables
// (ultravox_projector_test + prompt_injector_test) into ONE binary, halving the
// link/discovery overhead while keeping each check's self-contained TU.
//
// Each check lives in its own translation unit and exposes an int run_*(base)
// returning the number of FAILED sub-checks (0 == all passed); this dispatcher
// resolves the golden-dumps directory and aggregates the results.
//
// Dumps dir resolution (first hit wins):
//   argv[1]  ->  $BLACKWELL_ULTRAVOX_DUMPS  ->  tests/integration/golden_dumps/ultravox
// CTest passes the absolute dir as argv[1] (see tests/CMakeLists.txt), so the
// run is CWD-independent.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>

// Defined in test_ultravox_projector.cpp / test_prompt_injector.cpp.
int run_projector_parity(const std::string& base);
int run_prompt_injector_parity(const std::string& base);

int main(int argc, char** argv) {
    std::string base;
    if (argc > 1) {
        base = argv[1];
    } else if (const char* env = std::getenv("BLACKWELL_ULTRAVOX_DUMPS"); env && *env) {
        base = env;
    } else {
        base = "tests/integration/golden_dumps/ultravox";
    }

    std::printf("kernel_unit_tests  (dumps: %s)\n\n", base.c_str());

    int failures = 0;
    std::printf("--- Ultravox projector kernels -----------------------------------\n");
    failures += run_projector_parity(base);
    std::printf("\n--- Prompt injector splice ---------------------------------------\n");
    failures += run_prompt_injector_parity(base);

    std::printf("\n==================================================================\n");
    std::printf("%s: %d failed check(s) across the kernel unit suite.\n",
                failures == 0 ? "SUCCESS" : "FAILURE", failures);
    return failures == 0 ? 0 : 1;
}
