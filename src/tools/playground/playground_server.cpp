// agent_playground -- Agent Simulator / Prompt Playground (live LLM edition).
//
// A local HTTP server that serves a single-page web UI for the BlackwellLLM XML
// ReAct protocol. It has two layers:
//
//   1. MANUAL simulator (no GPU): build a multi-turn transcript by hand with a
//      click-to-inject snippet toolbar and live XML highlighting; /api/parse runs
//      the GENUINE production ToolParser so hallucinations (a stray
//      <file_content> instead of <tool_call>) are flagged against ground truth.
//
//   2. LIVE engine (CUDA): load a real .safetensors checkpoint and run the actual
//      BlackwellEngine. A BlackwellLLMAdapter re-frames the transcript into Qwen
//      ChatML and decodes a real completion; an auto-ReAct mode drives the real
//      AgentOrchestrator with real file/code tools.
//
// THREADING MODEL (see ModelService): the CUDA engine is a single, non-thread-safe
// resource. All long operations (model load, generation, ReAct) run on ONE shared
// background worker thread; a std::mutex-guarded state machine
// (Empty/Loading/Ready/Generating/Error) guarantees at most one CUDA operation is
// ever in flight. HTTP handlers never touch the engine -- they only read/mutate a
// small shared status struct under the lock and hand work to the worker. The
// browser polls a cheap /api/status endpoint, so neither the server's other
// handlers nor the UI ever block on CUDA.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "tool_parser.h"
#include "orchestrator.h"
#include "tool_registry.h"
#include "default_tools.h"
#include "../../agent/semantic_engine.h"
#include "../../agent_env/sandbox_fs.h"
#include "blackwell_llm_adapter.h"

#ifdef _WIN32
#include <windows.h>
#endif

using json = nlohmann::json;
using namespace agent::orch;
using playground::BlackwellLLMAdapter;

namespace {

constexpr size_t kMaxSeqLen = 8192;

// Directory containing the running executable, so static UI assets resolve
// relative to the binary rather than the (unpredictable) current working dir.
std::filesystem::path executable_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return std::filesystem::current_path();
    return std::filesystem::path(std::wstring(buf, n)).parent_path();
#else
    std::error_code ec;
    auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::current_path() : exe.parent_path();
#endif
}

// ---- ground-truth parser / tool helpers (manual-mode) ----------------------

// Mirrors the tools wired up in agent_orchestrator/default_tools.cpp. Used to
// build snippet buttons and to flag tool names the registry would reject.
const std::vector<std::pair<std::string, std::string>>& default_tool_manifest() {
    static const std::vector<std::pair<std::string, std::string>> kTools = {
        {"read_file",      "Read a whole file. arg: path (sandbox-relative)."},
        {"write_file",     "Create/overwrite a file. args: path, content (or body)."},
        {"patch_file",     "Replace first occurrence of 'find' with 'replace'. args: path, find, replace."},
        {"list_dir",       "List a sandbox directory (non-recursive). arg: path (default '.')."},
        {"analyze_source", "(Re)build the code graph from C++ source. arg: source (or body)."},
        {"callers_of",     "List functions that call a symbol. arg: symbol."},
        {"callees_of",     "List functions a symbol calls. arg: symbol."},
    };
    return kTools;
}

bool is_known_tool(const std::string& name) {
    for (const auto& t : default_tool_manifest())
        if (t.first == name) return true;
    return false;
}

const char* kind_str(ActionKind k) {
    switch (k) {
        case ActionKind::None:     return "None";
        case ActionKind::ToolCall: return "ToolCall";
        case ActionKind::Finish:   return "Finish";
    }
    return "None";
}

json action_to_json(const ParsedAction& a) {
    json j;
    j["kind"] = kind_str(a.kind);
    if (a.kind == ActionKind::ToolCall) {
        j["tool"]["name"]  = a.tool.name;
        j["tool"]["known"] = is_known_tool(a.tool.name);
        j["tool"]["body"]  = a.tool.body;
        json args = json::object();
        for (const auto& kv : a.tool.args) args[kv.first] = kv.second;
        j["tool"]["args"] = args;
    } else if (a.kind == ActionKind::Finish) {
        j["finish_text"] = a.finish_text;
    }
    return j;
}

const char* role_tag(Message::Role r) {
    switch (r) {
        case Message::Role::System:      return "System";
        case Message::Role::User:        return "User";
        case Message::Role::Assistant:   return "Assistant";
        case Message::Role::Observation: return "Observation";
    }
    return "User";
}

// Build the AgentOrchestrator role-tagged transcript from UI messages, byte-for-
// byte identical to AgentOrchestrator::render_transcript(). This is exactly what
// the adapter (and the orchestrator) consume.
std::string render_transcript(const json& messages) {
    auto tag = [](const std::string& r) -> const char* {
        if (r == "System")      return "SYSTEM";
        if (r == "User")        return "USER";
        if (r == "Assistant")   return "ASSISTANT";
        if (r == "Observation") return "OBSERVATION";
        return "UNKNOWN";
    };
    std::string out;
    for (const auto& m : messages) {
        out += '[';
        out += tag(m.value("role", std::string{}));
        out += "]\n";
        out += m.value("content", std::string{});
        out += "\n\n";
    }
    out += "[ASSISTANT]\n";
    return out;
}

// ---- ModelService: the single owner of the CUDA engine + worker thread -------

class ModelService {
public:
    enum class State { Empty, Loading, Ready, Generating, Error };

    ~ModelService() {
        if (worker_.joinable()) worker_.join();
    }

    json status() {
        std::lock_guard<std::mutex> lk(mu_);
        json j;
        j["state"]      = state_name(state_);
        j["busy"]       = (state_ == State::Loading || state_ == State::Generating);
        j["message"]    = message_;
        j["model_dir"]  = model_dir_;
        j["result_seq"] = result_seq_;
        j["last_kind"]  = last_kind_;
        j["last_output"]= last_output_;
        j["last_chatml"]= last_chatml_;
        j["last_react"] = last_react_;
        j["stream"]     = stream_buffer_;  // partial text of the in-flight job
        return j;
    }

    // Each of these returns "" on success, or a human-readable rejection reason
    // ("busy: ...", "no model loaded") when the request cannot start right now.
    std::string start_load(const std::string& dir) {
        std::lock_guard<std::mutex> lk(mu_);
        if (busy_locked()) return "busy: " + message_;
        join_worker_locked();
        state_ = State::Loading;
        model_dir_ = dir;
        message_ = "Loading model from " + dir + " ...";
        worker_ = std::thread([this, dir] { do_load(dir); });
        return "";
    }

    std::string start_generate(std::string transcript, BlackwellLLMAdapter::Params p) {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ != State::Ready) return ready_reason_locked();
        join_worker_locked();
        state_ = State::Generating;
        message_ = "Generating...";
        stream_buffer_.clear();  // fresh live buffer for this run
        worker_ = std::thread([this, t = std::move(transcript), p] { do_generate(t, p); });
        return "";
    }

    std::string start_react(std::string system, std::string goal, int max_iters,
                            BlackwellLLMAdapter::Params p) {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_ != State::Ready) return ready_reason_locked();
        join_worker_locked();
        state_ = State::Generating;
        message_ = "Running ReAct loop...";
        stream_buffer_.clear();  // fresh live buffer for this run
        worker_ = std::thread([this, system = std::move(system), goal = std::move(goal),
                               max_iters, p] { do_react(system, goal, max_iters, p); });
        return "";
    }

private:
    static const char* state_name(State s) {
        switch (s) {
            case State::Empty:      return "empty";
            case State::Loading:    return "loading";
            case State::Ready:      return "ready";
            case State::Generating: return "generating";
            case State::Error:      return "error";
        }
        return "empty";
    }

    bool busy_locked() const {
        return state_ == State::Loading || state_ == State::Generating;
    }
    std::string ready_reason_locked() const {
        if (busy_locked()) return "busy: " + message_;
        return "no model loaded";
    }
    // Safe because we only reach a start_* spawn when NOT busy, which means any
    // previous worker has already run to completion (it set a terminal state as
    // its last action); join() therefore returns immediately.
    void join_worker_locked() {
        if (worker_.joinable()) worker_.join();
    }

    // --- worker-thread bodies (exactly one runs at a time) --------------------

    void do_load(std::string dir) {
        std::unique_ptr<BlackwellLLMAdapter> ad;
        std::string err;
        try {
            ad = std::make_unique<BlackwellLLMAdapter>(dir, kMaxSeqLen);  // SLOW
        } catch (const std::exception& e) {
            err = e.what();
        } catch (...) {
            err = "unknown error";
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (err.empty()) {
            adapter_ = std::move(ad);
            state_ = State::Ready;
            message_ = "Model ready: " + dir;
        } else {
            adapter_.reset();
            state_ = State::Error;
            message_ = "Load failed: " + err;
        }
    }

    // Appends a freshly decoded token to the shared live buffer under the lock so
    // /api/status can surface generation in progress. Always returns true; a
    // future "Stop" button would return false to abort the decode loop.
    bool append_stream(const std::string& tok) {
        std::lock_guard<std::mutex> lk(mu_);
        stream_buffer_ += tok;
        return true;
    }

    void do_generate(std::string transcript, BlackwellLLMAdapter::Params p) {
        std::string out, chatml, err;
        // adapter_ is stable here: it is only mutated by do_load, which cannot run
        // concurrently with this (single worker thread + state machine).
        try {
            out = adapter_->generate(transcript, p, &chatml,
                                     [this](const std::string& tok) { return append_stream(tok); });
        } catch (const std::exception& e) {
            err = e.what();
        } catch (...) {
            err = "unknown error";
        }
        std::lock_guard<std::mutex> lk(mu_);
        last_kind_ = "generate";
        last_chatml_ = chatml;
        last_react_ = json::array();
        last_output_ = err.empty() ? out : ("[ERROR] " + err);
        message_ = err.empty() ? "Generation complete" : "Generation error";
        ++result_seq_;
        state_ = State::Ready;
    }

    void do_react(std::string system, std::string goal, int max_iters,
                  BlackwellLLMAdapter::Params p) {
        json arr = json::array();
        std::string err, answer;
        try {
            adapter_->set_params(p);
            // Stream every assistant turn of the loop into the live buffer. The
            // orchestrator only calls the single-arg generate(), so the callback
            // has to be installed on the adapter rather than passed per-call.
            adapter_->set_stream_callback(
                [this](const std::string& tok) { return append_stream(tok); });
            // Provision a confined sandbox + a fresh code graph for the tools.
            namespace fs = std::filesystem;
            fs::path ws = fs::temp_directory_path() / "blackwell_playground_ws";
            std::error_code ec;
            fs::create_directories(ws, ec);

            agent::env::SandboxFs sandbox(ws);
            agent::TreeSitterSemanticProvider provider;
            ToolRegistry registry;
            register_fs_tools(registry, sandbox);
            register_code_tools(registry, provider);

            OrchestratorConfig cfg;
            cfg.max_iterations = max_iters > 0 ? max_iters : 8;
            cfg.system_prompt = system;
            AgentOrchestrator orch(*adapter_, registry, cfg);
            RunResult r = orch.run(goal);
            answer = r.answer;
            for (const auto& m : orch.history())
                arr.push_back({{"role", role_tag(m.role)}, {"content", m.content}});
        } catch (const std::exception& e) {
            err = e.what();
        } catch (...) {
            err = "unknown error";
        }
        // Detach the callback (worker-thread only, so no lock needed) so a later
        // bare generate() can't stream into a stale buffer.
        adapter_->set_stream_callback(nullptr);
        std::lock_guard<std::mutex> lk(mu_);
        last_kind_ = "react";
        last_chatml_.clear();
        last_react_ = arr;
        last_output_ = err.empty() ? answer : ("[ERROR] " + err);
        message_ = err.empty() ? "ReAct complete" : "ReAct error";
        ++result_seq_;
        state_ = State::Ready;
    }

    std::mutex mu_;
    State state_ = State::Empty;
    std::string model_dir_;
    std::string message_ = "No model loaded.";
    std::string last_kind_;
    std::string last_output_;
    std::string last_chatml_;
    std::string stream_buffer_;  // partial output of the in-flight job (live UI)
    json last_react_ = json::array();
    long long result_seq_ = 0;
    std::unique_ptr<BlackwellLLMAdapter> adapter_;
    std::thread worker_;
};

BlackwellLLMAdapter::Params params_from_json(const json& b) {
    BlackwellLLMAdapter::Params p;
    if (b.contains("temperature"))    p.temperature = b["temperature"].get<float>();
    if (b.contains("top_p"))          p.top_p = b["top_p"].get<float>();
    if (b.contains("max_new_tokens")) p.max_new_tokens = b["max_new_tokens"].get<int>();
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    int port = (argc > 1) ? std::atoi(argv[1]) : 8080;
    if (port <= 0 || port > 65535) port = 8080;

    ModelService model;
    httplib::Server svr;

    // Serve the static frontend (ui/index.html, ui/style.css, ui/app.js). The
    // directory is resolved against the EXECUTABLE (<exe_dir>/ui), or taken from
    // argv[2] if given, so the tool works regardless of the launch CWD. cpp-httplib
    // checks mount-point files before dynamic GET handlers, but the /api/* routes
    // have no matching file, so they keep working.
    const std::filesystem::path ui_dir =
        (argc > 2) ? std::filesystem::path(argv[2]) : executable_dir() / "ui";
    if (!svr.set_mount_point("/", ui_dir.string())) {
        fprintf(stderr, "WARNING: UI assets not found at '%s'.\n", ui_dir.string().c_str());
        svr.Get("/", [ui_dir](const httplib::Request&, httplib::Response& res) {
            res.set_content(
                "<h1>Agent Playground</h1><p>UI assets not found at <code>" +
                    ui_dir.string() +
                    "</code>.<br>They ship in <code>src/tools/playground/ui</code> "
                    "and are copied next to the executable at build time.</p>",
                "text/html; charset=utf-8");
        });
    }

    svr.Get("/api/tools", [](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (const auto& t : default_tool_manifest())
            arr.push_back({{"name", t.first}, {"description", t.second}});
        res.set_content(arr.dump(), "application/json");
    });

    // --- manual mode: ground-truth parse + transcript render ------------------
    svr.Post("/api/parse", [](const httplib::Request& req, httplib::Response& res) {
        std::string text;
        try { text = json::parse(req.body).value("text", std::string{}); } catch (...) {}
        res.set_content(action_to_json(ToolParser::parse(text)).dump(), "application/json");
    });

    svr.Post("/api/render", [](const httplib::Request& req, httplib::Response& res) {
        json messages = json::array();
        try {
            json body = json::parse(req.body);
            if (body.contains("messages")) messages = body["messages"];
        } catch (...) {}
        res.set_content(json{{"transcript", render_transcript(messages)}}.dump(),
                        "application/json");
    });

    // --- live mode: model lifecycle + async jobs ------------------------------
    svr.Get("/api/status", [&model](const httplib::Request&, httplib::Response& res) {
        res.set_content(model.status().dump(), "application/json");
    });

    svr.Post("/api/model/load", [&model](const httplib::Request& req, httplib::Response& res) {
        std::string path;
        try { path = json::parse(req.body).value("path", std::string{}); } catch (...) {}
        if (path.empty()) {
            res.set_content(json{{"ok", false}, {"message", "model path is empty"}}.dump(),
                            "application/json");
            return;
        }
        const std::string err = model.start_load(path);
        res.set_content(json{{"ok", err.empty()},
                             {"message", err.empty() ? "loading started" : err}}.dump(),
                        "application/json");
    });

    svr.Post("/api/generate", [&model](const httplib::Request& req, httplib::Response& res) {
        json body = json::object();
        try { body = json::parse(req.body); } catch (...) {}
        json messages = body.contains("messages") ? body["messages"] : json::array();
        const std::string err =
            model.start_generate(render_transcript(messages), params_from_json(body));
        res.set_content(json{{"ok", err.empty()},
                             {"message", err.empty() ? "generation started" : err}}.dump(),
                        "application/json");
    });

    svr.Post("/api/react", [&model](const httplib::Request& req, httplib::Response& res) {
        json body = json::object();
        try { body = json::parse(req.body); } catch (...) {}
        const std::string system = body.value("system", std::string{});
        const std::string goal   = body.value("goal", std::string{});
        const int max_iters      = body.value("max_iterations", 8);
        if (goal.empty()) {
            res.set_content(json{{"ok", false}, {"message", "goal is empty"}}.dump(),
                            "application/json");
            return;
        }
        const std::string err = model.start_react(system, goal, max_iters, params_from_json(body));
        res.set_content(json{{"ok", err.empty()},
                             {"message", err.empty() ? "react started" : err}}.dump(),
                        "application/json");
    });

    const std::string url = "http://localhost:" + std::to_string(port);
    printf("\n  Agent Playground (live) running at %s\n", url.c_str());
    printf("  Load a checkpoint dir from the UI, or keep using manual mode.\n");
    printf("  (press Ctrl+C to stop)\n\n");
    fflush(stdout);

#ifdef _WIN32
    std::system(("start \"\" " + url).c_str());
#endif

    if (!svr.listen("127.0.0.1", port)) {
        fprintf(stderr, "ERROR: could not bind port %d (already in use?).\n", port);
        return 1;
    }
    return 0;
}
