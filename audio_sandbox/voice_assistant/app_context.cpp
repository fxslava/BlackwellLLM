// app_context.cpp — the speech-pipeline callbacks declared in app_context.hpp.
//
// MOVED, NOT CHANGED. Every branch below was inline in main.cpp; the state
// transitions, the barge-in edges and the deliberate omissions (see the IDLE case)
// are byte-for-byte what main() did. If this file contains a behavioural decision,
// the extraction was done wrong.
#include "app_context.hpp"

namespace rt {

void on_token(void* user, const SpeechTokenEvent* event, std::uint64_t gen_id) {
    auto* ctx = static_cast<AppContext*>(user);
    if (ctx == nullptr || ctx->view == nullptr || event == nullptr) return;
    ctx->view->on_local_token(event->text, gen_id);
}

void on_state(void* user, SpeechPipelineState /*prev*/, SpeechPipelineState next,
              std::uint64_t /*gen_id*/) {
    auto* ctx = static_cast<AppContext*>(user);
    if (ctx == nullptr || ctx->view == nullptr) return;
    ctx->view->on_pipeline_state(next);
#if defined(VOICE_ASSISTANT_HAS_TTS)
    if (ctx->tts != nullptr) {
        switch (next) {
            case SPEECH_STATE_PREFILL_SPEAKING:
                // THE BARGE-IN EDGE. The user has started talking. Whether we are
                // mid-answer or idle, anything still unspoken is now unwanted:
                // this aborts the solver mid-step, drops buffered text, and marks
                // queued audio stale so the device drops it on its next pull.
                ctx->tts->BargeIn();
                break;
            case SPEECH_STATE_INTERRUPTION_REWIND:
                // The pipeline's own barge-in path. Idempotent with the above.
                ctx->tts->BargeIn();
                break;
            case SPEECH_STATE_DECODE_TRANSLATING:
                // NOT a Resume point. This state means the TRANSCRIPTION decode
                // has started -- the user's words are being written down. The
                // answer does not exist yet and may never (the commit gate can
                // reject the turn). Clearing the barge-in latch here would arm the
                // speaker for a generation that is not this one; it is done on the
                // dispatcher's dispatch-start edge instead.
                break;
            case SPEECH_STATE_IDLE:
            default:
                break;
        }
    }
#endif
    // The return to IDLE is the turn boundary (SpeechTokenEvent carries no final
    // flag). The gate has already run by then, so the control's published verdict
    // is the authoritative reason this generation ended -- and it is what the UI
    // must show, because BargeIn and Eos both land on IDLE.
    if (next == SPEECH_STATE_IDLE) {
        ctx->view->on_local_final(ctx->control->last_reason());
        // NO EndOfTurn() here. IDLE is the end of the TRANSCRIPTION pass, which
        // happens BEFORE the intent is gated and long before the answer starts
        // streaming. Flushing the chunker at this point would push the tail of
        // whatever was buffered -- in practice nothing, now that the transcript no
        // longer feeds it -- and would then arrive a second time, mid-answer, as a
        // spurious boundary. The answer's real end is the dispatcher's completion
        // edge, which is where the flush lives.
    }
}

#if defined(VOICE_ASSISTANT_HAS_SILERO)
float vad_score(void* user, const float* block, std::size_t count) {
    auto* ctx = static_cast<VadContext*>(user);
    if (ctx == nullptr || ctx->vad == nullptr) return 0.0f;
    // EVERY BLOCK, unconditionally. Silero is recurrent -- it carries an LSTM
    // state across blocks -- so skipping blocks to suppress a verdict leaves that
    // state describing a signal the microphone never produced.
    return ctx->vad->feed(block, count);
}
#endif

}  // namespace rt
