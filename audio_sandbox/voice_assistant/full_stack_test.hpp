#pragma once
// -----------------------------------------------------------------------------
// full_stack_test.hpp — `--test-llm-tts "<prompt>"`: one injected turn on the
// COMPLETE, already-loaded stack, then close the window so the summaries print.
//
// WHY IT EXISTS WHEN --say ALREADY DOES: --say stands in FOR the LLM, so it runs
// F5 on an otherwise idle GPU and therefore cannot reproduce anything caused by
// decode and synthesis sharing the device. This one runs after everything is
// constructed -- engine, arena, audio head, Silero, AEC, F5 -- so it measures the
// loaded machine, which is the entire point.
//
// THE PROMPT GOES IN THROUGH submit_typed_turn, the SAME entry the typed box uses
// and therefore the same gate, dispatcher, routing and answer stream a spoken turn
// takes. Reaching into generate_local_reply() directly would have been fewer lines
// and would have tested a path no user can take -- and would have pinned the test
// to the local leg, which is precisely the coupling this release removed.
//
// Runs on its own thread and swallows everything: a diagnostic must not be able to
// take the app down.
// -----------------------------------------------------------------------------
#include "build_features.hpp"

#include <string>

#include "app_context.hpp"
#include "audio_pipeline_binder.hpp"
#include "conversation_router.hpp"
#include "settings_store.hpp"

namespace rt {

class AssistantWindow;

// `settings` is copied, not borrowed: the hot-swap step below deliberately mutates
// a COPY to name a different endpoint, and it must not disturb what the app is
// running on.
void run_full_stack_test(const std::string& prompt, AssistantSettings settings, bool use_real,
                         AppContext& ctx, AudioPipelineBinder& audio,
                         ConversationRouter& router, AssistantWindow& window);

}  // namespace rt
