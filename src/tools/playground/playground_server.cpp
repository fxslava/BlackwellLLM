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

using json = nlohmann::json;
using namespace agent::orch;
using playground::BlackwellLLMAdapter;

namespace {

constexpr size_t kMaxSeqLen = 8192;

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

    void do_generate(std::string transcript, BlackwellLLMAdapter::Params p) {
        std::string out, chatml, err;
        // adapter_ is stable here: it is only mutated by do_load, which cannot run
        // concurrently with this (single worker thread + state machine).
        try {
            out = adapter_->generate(transcript, p, &chatml);
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

extern const char* kIndexHtml;

}  // namespace

int main(int argc, char** argv) {
    int port = (argc > 1) ? std::atoi(argv[1]) : 8080;
    if (port <= 0 || port > 65535) port = 8080;

    ModelService model;
    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(kIndexHtml, "text/html; charset=utf-8");
    });

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

// ============================================================================
// Embedded single-page frontend. Self-contained (no external CDN/files). ASCII-
// only; the snippet caret sentinel is built with String.fromCharCode at runtime.
// ============================================================================
namespace {
const char* kIndexHtml = R"HTMLDOC(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BlackwellLLM &mdash; Agent Playground</title>
<style>
  :root{
    --bg:#0f1117; --panel:#171a23; --panel2:#1d212c; --border:#2a2f3c;
    --txt:#d7dbe6; --muted:#8b93a7; --accent:#5b8cff;
    --sys:#b98bff; --user:#5b8cff; --asst:#4fd08a; --obs:#f0a64a;
    --tag-known:#4fd08a; --tag-unknown:#ff5c6c; --attr:#8bd0ff; --val:#f0a64a;
    --ok:#4fd08a; --warn:#ff5c6c; --busy:#f0a64a;
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
    background:var(--bg);color:var(--txt)}
  code,pre,textarea,input,.mono{font-family:ui-monospace,"Cascadia Code",Consolas,monospace}
  header{display:flex;align-items:center;gap:12px;padding:10px 16px;
    background:var(--panel);border-bottom:1px solid var(--border)}
  header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
  header .pill{font-size:11px;color:var(--muted);border:1px solid var(--border);
    padding:2px 8px;border-radius:999px}
  header .spacer{flex:1}
  button{font:inherit;cursor:pointer;background:var(--panel2);color:var(--txt);
    border:1px solid var(--border);border-radius:7px;padding:5px 10px}
  button:hover{border-color:var(--accent)}
  button:active{transform:translateY(1px)}
  button:disabled{opacity:.4;cursor:not-allowed}
  input{background:#12141c;color:var(--txt);border:1px solid var(--border);
    border-radius:6px;padding:5px 8px;font-size:12.5px}
  .main{display:grid;grid-template-columns:1fr 400px;gap:14px;padding:14px;
    height:calc(100vh - 49px)}
  .col{display:flex;flex-direction:column;min-height:0;gap:12px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px}
  .card h2{font-size:11px;text-transform:uppercase;letter-spacing:.8px;
    color:var(--muted);margin:0;padding:9px 12px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;gap:8px}
  .card h2 .spacer{flex:1}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--muted);display:inline-block}
  .dot.ready{background:var(--ok)} .dot.error{background:var(--warn)}
  .dot.loading,.dot.generating{background:var(--busy);animation:pulse 1s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
  .spin{width:13px;height:13px;border:2px solid var(--border);border-top-color:var(--accent);
    border-radius:50%;display:inline-block;animation:spin .7s linear infinite;vertical-align:-2px}
  @keyframes spin{to{transform:rotate(360deg)}}
  #modelCard .body{padding:11px 12px;display:flex;flex-direction:column;gap:9px}
  .row{display:flex;gap:8px;align-items:center}
  .row .grow{flex:1}
  .field{display:flex;flex-direction:column;gap:3px}
  .field label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
  .field input{width:74px}
  .modelmsg{font-size:11.5px;color:var(--muted);word-break:break-word}
  details.chatml{font-size:11px}
  details.chatml summary{cursor:pointer;color:var(--muted);padding:2px 0}
  details.chatml pre{margin:6px 0 0;padding:8px;background:#12141c;border:1px solid var(--border);
    border-radius:6px;max-height:160px;overflow:auto;white-space:pre-wrap;word-break:break-word;
    font-size:11px;color:var(--attr)}
  #transcript{flex:1;overflow:auto;padding:10px;display:flex;flex-direction:column;gap:9px}
  .msg{border:1px solid var(--border);border-left-width:4px;border-radius:8px;
    background:var(--panel2);overflow:hidden}
  .msg .bar{display:flex;align-items:center;gap:8px;padding:5px 9px;font-size:11px;
    text-transform:uppercase;letter-spacing:.6px;font-weight:600}
  .msg .bar .spacer{flex:1}
  .msg .bar button{padding:1px 7px;font-size:11px;background:transparent}
  .msg pre{margin:0;padding:9px 11px;white-space:pre-wrap;word-break:break-word;
    font-size:12.5px;border-top:1px solid var(--border)}
  .role-System{border-left-color:var(--sys)} .role-System .bar{color:var(--sys)}
  .role-User{border-left-color:var(--user)} .role-User .bar{color:var(--user)}
  .role-Assistant{border-left-color:var(--asst)} .role-Assistant .bar{color:var(--asst)}
  .role-Observation{border-left-color:var(--obs)} .role-Observation .bar{color:var(--obs)}
  .badge{font-size:10px;padding:1px 7px;border-radius:999px;border:1px solid currentColor;
    text-transform:none;letter-spacing:0}
  .badge.ok{color:var(--ok)} .badge.warn{color:var(--warn)} .badge.muted{color:var(--muted)}
  .empty{color:var(--muted);text-align:center;padding:30px;font-style:italic}
  .composer{display:flex;flex-direction:column}
  .roles{display:flex;gap:6px;padding:9px 12px;flex-wrap:wrap;align-items:center}
  .roles .lbl{font-size:11px;color:var(--muted);margin-right:2px}
  .roles button{border-radius:999px}
  .roles button.active{color:#0f1117;font-weight:700;border-color:transparent}
  .roles button[data-r=System].active{background:var(--sys)}
  .roles button[data-r=User].active{background:var(--user)}
  .roles button[data-r=Assistant].active{background:var(--asst)}
  .roles button[data-r=Observation].active{background:var(--obs)}
  .toolbar{display:flex;flex-wrap:wrap;gap:6px;padding:8px 12px;
    border-top:1px solid var(--border);border-bottom:1px solid var(--border)}
  .toolbar .grp{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
  .toolbar .sep{width:1px;align-self:stretch;background:var(--border);margin:0 4px}
  .toolbar button{font-size:12px;padding:4px 9px}
  .toolbar .snip{color:var(--tag-known)}
  .toolbar .tool{color:var(--attr)}
  .toolbar .gl{font-size:10px;color:var(--muted);width:100%;margin-top:2px}
  .editor{position:relative;height:170px;margin:0}
  .editor pre,.editor textarea{position:absolute;inset:0;margin:0;padding:11px 13px;
    border:0;white-space:pre-wrap;word-break:break-word;overflow:auto;
    font-size:13px;line-height:1.55;letter-spacing:0;tab-size:2}
  .editor pre{pointer-events:none;color:var(--txt);background:transparent;z-index:1}
  .editor textarea{color:transparent;background:#12141c;caret-color:#fff;
    resize:none;z-index:2;outline:none}
  .editor textarea::selection{background:#33415f}
  .tag-known{color:var(--tag-known)}
  .tag-unknown{color:var(--tag-unknown);text-decoration:wavy underline}
  .attr{color:var(--attr)} .val{color:var(--val)}
  .composer .actions{display:flex;gap:8px;padding:10px 12px;align-items:center;flex-wrap:wrap}
  .composer .actions .spacer{flex:1}
  .composer .actions .primary{background:var(--accent);color:#fff;border-color:transparent;
    font-weight:600;padding:6px 14px}
  .composer .actions .gen{background:var(--asst);color:#0f1117;border-color:transparent;font-weight:700}
  .composer .actions .react{background:var(--sys);color:#0f1117;border-color:transparent;font-weight:700}
  .hint{font-size:11px;color:var(--muted)}
  #inspector{padding:11px;overflow:auto}
  .kv{display:flex;gap:8px;margin:3px 0;font-size:12.5px}
  .kv .k{color:var(--muted);min-width:64px}
  .kv .v{word-break:break-word}
  .insp-kind{font-size:18px;font-weight:700;margin-bottom:6px}
  .insp-none{color:var(--warn)} .insp-call{color:var(--asst)} .insp-finish{color:var(--user)}
  .argrow{display:flex;gap:8px;font-size:12px;padding:2px 0;border-top:1px dashed var(--border)}
  .argrow .ak{color:var(--attr);min-width:70px}
  .argrow .av{white-space:pre-wrap;word-break:break-word;color:var(--val)}
  #rendered{flex:1;min-height:0;display:flex;flex-direction:column}
  #renderedOut{margin:0;padding:11px;overflow:auto;height:100%;white-space:pre-wrap;
    word-break:break-word;font-size:12px;color:var(--muted)}
  .insp-note{font-size:11px;color:var(--muted);margin-top:8px;border-top:1px solid var(--border);
    padding-top:8px}
</style>
</head>
<body>
<header>
  <h1>Agent Playground</h1>
  <span class="pill">XML ReAct &middot; live engine</span>
  <span class="pill" id="enginePill"><span class="dot" id="engineDot"></span> <span id="engineState">no model</span></span>
  <span class="spacer"></span>
  <button id="seedBtn">Load demo</button>
  <button id="renderBtn">Render transcript</button>
  <button id="clearBtn">Clear</button>
</header>

<div class="main">
  <div class="col">
    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0">
      <h2>Transcript <span class="spacer"></span><span class="hint" id="genStatus"></span></h2>
      <div id="transcript"><div class="empty">No turns yet. Pick a role, type or click a snippet, then "Add turn".</div></div>
    </div>

    <div class="card composer">
      <div class="roles">
        <span class="lbl">Input role:</span>
        <button data-r="System">System</button>
        <button data-r="User">User</button>
        <button data-r="Assistant" class="active">Assistant</button>
        <button data-r="Observation">Observation</button>
      </div>

      <div class="toolbar" id="toolbar">
        <div class="grp">
          <button class="snip" data-snip="toolcall">&lt;tool_call&gt;</button>
          <button class="snip" data-snip="toolcall_sc">&lt;tool_call/&gt;</button>
          <button class="snip" data-snip="arg">&lt;arg&gt;</button>
          <button class="snip" data-snip="finish">&lt;finish&gt;</button>
          <button class="snip" data-snip="finish_sc">&lt;finish/&gt;</button>
        </div>
        <div class="sep"></div>
        <div class="grp" id="toolBtns"></div>
        <div class="gl">Green = structural tags. Blue = full tool_call for a registered tool. Caret lands where you'll type next.</div>
      </div>

      <div class="editor">
        <pre aria-hidden="true"><code id="hl"></code></pre>
        <textarea id="input" spellcheck="false" placeholder="Type the turn here. Unknown tags (e.g. <file_content>) are flagged in red -- that's how you catch hallucinations."></textarea>
      </div>

      <div class="actions">
        <span class="hint" id="liveHint"></span>
        <span class="spacer"></span>
        <button id="addBtn" class="primary">Add turn &#9166;</button>
        <button id="genBtn" class="gen" disabled title="Run the real model to produce the next assistant turn">Generate (model)</button>
        <button id="reactBtn" class="react" disabled title="Run the real ReAct loop with real tools">Run ReAct</button>
      </div>
    </div>
  </div>

  <div class="col">
    <div class="card" id="modelCard">
      <h2>Model <span class="spacer"></span><span id="modelStateText" class="hint">empty</span></h2>
      <div class="body">
        <div class="row">
          <input class="grow" id="modelPath" placeholder="checkpoint dir, e.g. F:/AI/qwen2.5-coder-7b">
          <button id="loadBtn">Load Model</button>
        </div>
        <div class="row">
          <div class="field"><label>temp</label><input id="temp" type="number" step="0.05" min="0" max="2" value="0.20"></div>
          <div class="field"><label>top_p</label><input id="topp" type="number" step="0.05" min="0" max="1" value="0.90"></div>
          <div class="field"><label>max tokens</label><input id="maxtok" type="number" step="32" min="1" value="512"></div>
          <div class="field"><label>react iters</label><input id="iters" type="number" step="1" min="1" value="8"></div>
        </div>
        <div class="modelmsg" id="modelMsg">No model loaded. The manual simulator works without one.</div>
        <details class="chatml" id="chatmlBox" style="display:none">
          <summary>Last ChatML prompt sent to the model</summary>
          <pre id="chatmlOut"></pre>
        </details>
      </div>
    </div>

    <div class="card" style="flex:0 0 auto">
      <h2>Parser inspector &mdash; ground truth (real ToolParser)</h2>
      <div id="inspector"><div class="hint">Start typing as the Assistant role to see what the orchestrator's parser extracts.</div></div>
    </div>
    <div class="card" id="rendered">
      <h2>Rendered transcript (what the model sees)</h2>
      <pre id="renderedOut" class="mono">Click "Render transcript".</pre>
    </div>
  </div>
</div>

<script>
)HTMLDOC"
        R"HTMLDOC("use strict";
const CARET = String.fromCharCode(0x2038);    // caret-placement sentinel for snippets
const KNOWN_TAGS = new Set(["tool_call","arg","finish"]);
const TOOL_ARGS = {
  read_file:["path"], write_file:["path","content"], patch_file:["path","find","replace"],
  list_dir:["path"], analyze_source:["source"], callers_of:["symbol"], callees_of:["symbol"]
};
let messages = [];
let currentRole = "Assistant";
let editIndex = -1;

const $ = s => document.querySelector(s);
const input = $("#input"), hl = $("#hl");

/* ---------- XML syntax highlighting (overlay) ---------- */
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function hlTag(tag){
  const m = tag.match(/^<\/?([\w-]+)/);
  const name = m ? m[1] : "";
  const cls = KNOWN_TAGS.has(name) ? "tag-known" : "tag-unknown";
  let inner = esc(tag).replace(/([\w-]+)=("[^"]*"|'[^']*')/g,
    '<span class="attr">$1</span>=<span class="val">$2</span>');
  return '<span class="'+cls+'">'+inner+'</span>';
}
function highlight(text){
  let out = "", last = 0, m; const re = /<\/?[\w-]+[^>]*>/g;
  while((m = re.exec(text))){ out += esc(text.slice(last,m.index)); out += hlTag(m[0]); last = m.index+m[0].length; }
  out += esc(text.slice(last));
  return out + "\n";
}
function syncHL(){ hl.innerHTML = highlight(input.value); hl.parentElement.scrollTop = input.scrollTop; }
input.addEventListener("input", () => { syncHL(); liveParse(); });
input.addEventListener("scroll", () => { hl.parentElement.scrollTop = input.scrollTop; hl.parentElement.scrollLeft = input.scrollLeft; });

/* ---------- snippet insertion ---------- */
function insertSnippet(text){
  const s = input.selectionStart, e = input.selectionEnd, v = input.value;
  let snip = text, caret = snip.indexOf(CARET);
  snip = snip.replace(CARET, "");
  if(caret < 0) caret = snip.length;
  input.value = v.slice(0,s) + snip + v.slice(e);
  const pos = s + caret;
  input.focus(); input.setSelectionRange(pos,pos);
  syncHL(); liveParse();
}
const SNIPPETS = {
  toolcall:    '<tool_call name="'+CARET+'">\n  \n</tool_call>',
  toolcall_sc: '<tool_call name="'+CARET+'"/>',
  arg:         '<arg name="'+CARET+'"></arg>',
  finish:      '<finish>'+CARET+'</finish>',
  finish_sc:   '<finish/>'
};
document.querySelectorAll('.toolbar [data-snip]').forEach(b =>
  b.addEventListener("click", () => insertSnippet(SNIPPETS[b.dataset.snip])));
function toolSnippet(name){
  const args = TOOL_ARGS[name] || [];
  if(!args.length) return '<tool_call name="'+name+'">'+CARET+'</tool_call>';
  const body = args.map((a,i) => '  <arg name="'+a+'">'+(i===0?CARET:"")+'</arg>').join("\n");
  return '<tool_call name="'+name+'">\n'+body+'\n</tool_call>';
}

/* ---------- role selector ---------- */
document.querySelectorAll('.roles button').forEach(b => b.addEventListener("click", () => {
  currentRole = b.dataset.r;
  document.querySelectorAll('.roles button').forEach(x => x.classList.toggle("active", x === b));
  liveParse();
}));

/* ---------- live parse (real ToolParser) ---------- */
let parseTimer = null;
function liveParse(){ clearTimeout(parseTimer); parseTimer = setTimeout(doParse, 120); }
async function doParse(){
  const insp = $("#inspector");
  if(currentRole !== "Assistant"){
    insp.innerHTML = '<div class="hint">Parser runs on <b style="color:var(--asst)">Assistant</b> turns -- the only role the orchestrator parses. Current role: '+currentRole+'.</div>';
    $("#liveHint").textContent = "";
    return;
  }
  let r;
  try{
    r = await (await fetch("/api/parse",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text:input.value})})).json();
  }catch(e){ insp.innerHTML = '<div class="insp-none">server error</div>'; return; }
  renderInspector(insp, r);
}
function renderInspector(insp, r){
  const lh = $("#liveHint");
  if(r.kind === "None"){
    insp.innerHTML = '<div class="insp-kind insp-none">None</div>'+
      '<div class="hint">No valid &lt;tool_call&gt; or &lt;finish&gt; found. If you expected an action, you probably typed a hallucinated/typo tag -- check the red underlines on the left.</div>';
    lh.innerHTML = '<span style="color:var(--warn)">&#9888; no actionable tag</span>';
  }else if(r.kind === "ToolCall"){
    const t = r.tool, known = t.known;
    let h = '<div class="insp-kind insp-call">ToolCall</div>';
    h += '<div class="kv"><span class="k">name</span><span class="v">'+esc(t.name)+
      (known ? ' <span class="badge ok">registered</span>' : ' <span class="badge warn">&#9888; not a registered tool</span>')+'</span></div>';
    const ks = Object.keys(t.args||{});
    h += '<div class="kv"><span class="k">args</span><span class="v">'+(ks.length?'':'<span class="hint">(none)</span>')+'</span></div>';
    ks.forEach(k => h += '<div class="argrow"><span class="ak">'+esc(k)+'</span><span class="av">'+esc(t.args[k])+'</span></div>');
    insp.innerHTML = h + '<div class="insp-note">This is the exact ToolInvocation the dispatcher would receive.</div>';
    lh.innerHTML = known ? '<span style="color:var(--ok)">&#10003; tool_call &rarr; '+esc(t.name)+'</span>'
                         : '<span style="color:var(--warn)">&#9888; unknown tool: '+esc(t.name)+'</span>';
  }else if(r.kind === "Finish"){
    insp.innerHTML = '<div class="insp-kind insp-finish">Finish</div>'+
      '<div class="kv"><span class="k">answer</span><span class="v">'+(esc(r.finish_text)||'<span class="hint">(empty)</span>')+'</span></div>'+
      '<div class="insp-note">The ReAct loop would terminate and return this text.</div>';
    lh.innerHTML = '<span style="color:var(--user)">&#10003; finish</span>';
  }
}

/* ---------- transcript model ---------- */
async function badgeFor(content){
  try{
    const r = await (await fetch("/api/parse",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text:content})})).json();
    if(r.kind === "ToolCall") return r.tool.known
      ? '<span class="badge ok">tool_call: '+esc(r.tool.name)+'</span>'
      : '<span class="badge warn">&#9888; unknown tool: '+esc(r.tool.name)+'</span>';
    if(r.kind === "Finish") return '<span class="badge muted">finish</span>';
    return '<span class="badge warn">&#9888; no action</span>';
  }catch(e){ return ''; }
}
async function renderTranscript(){
  const box = $("#transcript");
  if(!messages.length){ box.innerHTML = '<div class="empty">No turns yet. Pick a role, type or click a snippet, then "Add turn".</div>'; return; }
  box.innerHTML = "";
  for(let i=0;i<messages.length;i++){
    const m = messages[i];
    const el = document.createElement("div");
    el.className = "msg role-"+m.role;
    let badge = "";
    if(m.role === "Assistant") badge = await badgeFor(m.content);
    el.innerHTML =
      '<div class="bar"><span>'+m.role+'</span>'+badge+'<span class="spacer"></span>'+
      '<button data-edit="'+i+'">edit</button><button data-del="'+i+'">delete</button></div>'+
      '<pre>'+highlight(m.content)+'</pre>';
    box.appendChild(el);
  }
  box.scrollTop = box.scrollHeight;
  box.querySelectorAll('[data-del]').forEach(b => b.onclick = () => { messages.splice(+b.dataset.del,1); renderTranscript(); });
  box.querySelectorAll('[data-edit]').forEach(b => b.onclick = () => {
    const i = +b.dataset.edit; const m = messages[i];
    currentRole = m.role;
    document.querySelectorAll('.roles button').forEach(x => x.classList.toggle("active", x.dataset.r === m.role));
    input.value = m.content; editIndex = i;
    $("#addBtn").innerHTML = "Update turn #"+i;
    syncHL(); liveParse(); input.focus();
  });
}

/* ---------- manual actions ---------- */
$("#addBtn").addEventListener("click", () => {
  const content = input.value;
  if(!content.trim()) return;
  if(editIndex >= 0){ messages[editIndex].content = content; messages[editIndex].role = currentRole; editIndex = -1; $("#addBtn").innerHTML = "Add turn &#9166;"; }
  else messages.push({role:currentRole, content});
  input.value = ""; syncHL(); liveParse(); renderTranscript();
});
input.addEventListener("keydown", e => { if((e.ctrlKey||e.metaKey) && e.key === "Enter"){ e.preventDefault(); $("#addBtn").click(); } });
$("#clearBtn").addEventListener("click", () => { messages = []; renderTranscript(); $("#renderedOut").textContent = 'Click "Render transcript".'; });
$("#renderBtn").addEventListener("click", async () => {
  const r = await (await fetch("/api/render",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({messages})})).json();
  $("#renderedOut").textContent = r.transcript;
});

/* ---------- tool toolbar from manifest ---------- */
(async function initTools(){
  let tools = [];
  try{ tools = await (await fetch("/api/tools")).json(); }catch(e){ tools = Object.keys(TOOL_ARGS).map(n=>({name:n,description:""})); }
  const box = $("#toolBtns");
  tools.forEach(t => {
    const b = document.createElement("button");
    b.className = "tool"; b.textContent = t.name; b.title = t.description || "";
    b.addEventListener("click", () => insertSnippet(toolSnippet(t.name)));
    box.appendChild(b);
  });
})();

/* ---------- demo seed ---------- */
$("#seedBtn").addEventListener("click", () => {
  messages = [
    {role:"System", content:"You are a coding agent. Act using exactly one <tool_call name=\"...\"><arg name=\"...\">...</arg></tool_call> per turn, or <finish>...</finish> when done."},
    {role:"User", content:"Read main.cpp and tell me what it does."},
    {role:"Assistant", content:"I'll open the file.\n<tool_call name=\"read_file\">\n  <arg name=\"path\">main.cpp</arg>\n</tool_call>"},
    {role:"Observation", content:"int main(){ return 0; }"},
    {role:"Assistant", content:"<finish>main.cpp is an empty entry point that returns 0.</finish>"}
  ];
  renderTranscript();
});

/* ====================================================================== */
/* LIVE ENGINE: model loading + async generation via /api/status polling  */
/* ====================================================================== */
let engineState = "empty";
let lastSeq = 0;           // result_seq we have already consumed
let awaiting = null;       // "generate" | "react" | null
let seqInit = false;

function params(){
  return {
    temperature: parseFloat($("#temp").value) || 0,
    top_p: parseFloat($("#topp").value) || 0.9,
    max_new_tokens: parseInt($("#maxtok").value) || 512
  };
}
function setEngineUI(s){
  engineState = s.state;
  $("#engineDot").className = "dot " + s.state;
  $("#engineState").textContent = s.state==="ready" ? "ready" : s.state==="loading" ? "loading" :
    s.state==="generating" ? "working" : s.state==="error" ? "error" : "no model";
  $("#modelStateText").textContent = s.state;
  $("#modelMsg").textContent = s.message || "";
  const ready = s.state === "ready";
  const idle = ready || s.state === "empty" || s.state === "error";
  $("#genBtn").disabled = !ready || !!awaiting;
  $("#reactBtn").disabled = !ready || !!awaiting;
  $("#loadBtn").disabled = !idle;
  const gs = $("#genStatus");
  if(awaiting && s.state === "generating"){
    gs.innerHTML = '<span class="spin"></span> ' + (awaiting==="react" ? "running ReAct loop..." : "generating...");
  } else if(s.state === "loading"){
    gs.innerHTML = '<span class="spin"></span> loading model...';
  } else {
    gs.textContent = "";
  }
}
async function pollStatus(){
  let s;
  try{ s = await (await fetch("/api/status")).json(); }catch(e){ return; }
  if(!seqInit){ lastSeq = s.result_seq; seqInit = true; }   // ignore pre-existing result
  setEngineUI(s);
  if(s.result_seq > lastSeq){
    lastSeq = s.result_seq;
    const kind = s.last_kind, was = awaiting;
    awaiting = null;
    if(kind === "generate" && was === "generate"){
      if(s.last_output){ messages.push({role:"Assistant", content:s.last_output}); renderTranscript(); }
      if(s.last_chatml){ $("#chatmlBox").style.display=""; $("#chatmlOut").textContent = s.last_chatml; }
)HTMLDOC"
        R"HTMLDOC(    } else if(kind === "react" && was === "react"){
      if(Array.isArray(s.last_react) && s.last_react.length){
        messages = s.last_react.map(m => ({role:m.role, content:m.content}));
        renderTranscript();
      }
    }
    setEngineUI(s);  // re-enable buttons now that awaiting cleared
  }
}
setInterval(pollStatus, 800);
pollStatus();

$("#loadBtn").addEventListener("click", async () => {
  const path = $("#modelPath").value.trim();
  if(!path){ $("#modelMsg").textContent = "Enter a checkpoint directory first."; return; }
  $("#modelMsg").textContent = "Requesting load...";
  const r = await (await fetch("/api/model/load",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({path})})).json();
  if(!r.ok) $("#modelMsg").textContent = r.message;
  pollStatus();
});

$("#genBtn").addEventListener("click", async () => {
  if(engineState !== "ready") return;
  awaiting = "generate";
  $("#genBtn").disabled = true; $("#reactBtn").disabled = true;
  $("#genStatus").innerHTML = '<span class="spin"></span> generating...';
  const body = Object.assign({messages}, params());
  const r = await (await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)})).json();
  if(!r.ok){ awaiting = null; $("#genStatus").textContent = ""; $("#modelMsg").textContent = r.message; pollStatus(); }
});

$("#reactBtn").addEventListener("click", async () => {
  if(engineState !== "ready") return;
  const sys = (messages.find(m => m.role === "System") || {}).content || "";
  const lastUser = [...messages].reverse().find(m => m.role === "User");
  if(!lastUser){ $("#modelMsg").textContent = "Add a User turn to use as the ReAct goal."; return; }
  awaiting = "react";
  $("#genBtn").disabled = true; $("#reactBtn").disabled = true;
  $("#genStatus").innerHTML = '<span class="spin"></span> running ReAct loop...';
  const body = Object.assign({system:sys, goal:lastUser.content, max_iterations: parseInt($("#iters").value)||8}, params());
  const r = await (await fetch("/api/react",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)})).json();
  if(!r.ok){ awaiting = null; $("#genStatus").textContent = ""; $("#modelMsg").textContent = r.message; pollStatus(); }
});

syncHL(); liveParse();
</script>
</body>
</html>
)HTMLDOC";
}  // namespace
