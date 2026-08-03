// -----------------------------------------------------------------------------
// Compile-time contract check for the header-only F5EngineSynthesizer.
//
// The adapter has no out-of-line definitions, so without a TU that includes it
// nothing in the build would ever compile it: it would rot silently until the
// first consumer appeared, which is exactly when a broken seam is most
// expensive. This file exists so `blackwell_tts_f5` proves, on every build, that
// the CUDA engine still satisfies the CUDA-free bridge's interface.
//
// It is also where that binding is enforced in ONE direction only: this TU sees
// both f5_tts_engine.hpp (ONNXRuntime, cudart) and tts_duplex_bridge.hpp, and it
// lives in blackwell_tts_f5. blackwell_tts never sees the engine.
// -----------------------------------------------------------------------------
#include "f5_engine_synthesizer.hpp"

#include <type_traits>

namespace blackwell::tts {

static_assert(std::is_base_of_v<ISynthesizer, F5EngineSynthesizer>,
              "F5EngineSynthesizer must implement the bridge's synthesis seam");
static_assert(!std::is_abstract_v<F5EngineSynthesizer>,
              "F5EngineSynthesizer has an unimplemented ISynthesizer method -- the "
              "engine and the seam have drifted apart");

}  // namespace blackwell::tts
