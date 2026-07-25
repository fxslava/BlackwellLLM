// Parity test for the prompt-injector kernel (splice_audio_embeddings_kernel):
// splice 188 audio embeddings into the text sequence at the <|audio|> position
// and match the PyTorch reference (spliced_embeds_ref.bin) via cosine similarity.
//
// One of the checks compiled into the consolidated `kernel_unit_tests` binary:
// the entry point is run_prompt_injector_parity(base) (see
// tests/kernel_unit_tests_main.cpp), not a main().

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"            // CUDA_CHECK_THROW
#include "device_buffer.h"     // blackwell::DeviceBuffer
#include "audio_test_utils.h"  // load_bin_file, compute_cosine_similarity
#include "prompt_injector.cuh"

using blackwell::DeviceBuffer;
using audio_test::compute_cosine_similarity;
using audio_test::load_bin_file;

static DeviceBuffer<float> to_device(const std::vector<float>& h) {
    DeviceBuffer<float> d(h.size());
    CUDA_CHECK_THROW(cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    return d;
}

// Runs the prompt-injector splice parity check against the dumps under `base`.
// Returns the number of FAILED checks (0 == passed). Called from the
// consolidated kernel_unit_tests dispatcher.
int run_prompt_injector_parity(const std::string& base) {
    // Geometry sidecar written by dump_injector_tensors(): "seq_len audio_pos num_audio hidden".
    int seq_len = 0, audio_pos = 0, num_audio = 0, hidden = 0;
    {
        std::ifstream m(base + "/injector_meta.txt");
        if (!(m >> seq_len >> audio_pos >> num_audio >> hidden)) {
            std::fprintf(stderr, "FATAL: cannot read injector_meta.txt\n");
            return 1;
        }
    }
    const int out_rows = seq_len - 1 + num_audio;

    auto text     = load_bin_file(base + "/text_embeds.bin");
    auto audio    = load_bin_file(base + "/audio_embeds.bin");
    auto expected = load_bin_file(base + "/spliced_embeds_ref.bin");
    if (text.size() != (size_t)seq_len * hidden ||
        audio.size() != (size_t)num_audio * hidden ||
        expected.size() != (size_t)out_rows * hidden) {
        std::fprintf(stderr, "FATAL: input size mismatch\n");
        return 1;
    }

    DeviceBuffer<float> d_text = to_device(text);
    DeviceBuffer<float> d_audio = to_device(audio);
    DeviceBuffer<float> d_out((size_t)out_rows * hidden);

    inject_audio_embeddings(d_text, d_audio, d_out, seq_len, audio_pos, num_audio, hidden);
    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    std::vector<float> got((size_t)out_rows * hidden);
    CUDA_CHECK_THROW(cudaMemcpy(got.data(), d_out.get(), got.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));

    const double cos = compute_cosine_similarity(got, expected);
    const bool ok = cos > 0.999;
    std::printf("[%s] PromptInjector splice [%d,%d]->[%d,%d]  cosine = %.8f  (> 0.999)\n",
                ok ? "PASS" : "FAIL", seq_len, hidden, out_rows, hidden, cos);
    std::printf("%s: prompt-injector parity.\n", ok ? "SUCCESS" : "FAILURE");
    return ok ? 0 : 1;  // # of failed checks; aggregated by kernel_unit_tests
}
