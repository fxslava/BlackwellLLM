// agent_playground -- Agent Simulator / Prompt Playground.
//
// A local HTTP server that serves a single-page web UI for manually simulating
// the Orchestrator/Environment around the BlackwellLLM XML ReAct protocol. It
// lets you build a multi-turn transcript by hand, with a click-to-inject snippet
// toolbar and live XML syntax highlighting, so you can iterate on system prompts
// and spot model hallucinations (e.g. a stray <file_content> instead of a
// <tool_call>) at a glance.
//
// The /api/parse endpoint runs the GENUINE production parser
// (agent::orch::ToolParser::parse from agent_orchestrator/tool_parser.cpp), so
// the "what the orchestrator sees" panel is ground truth, not a JS guess.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>

#include "httplib.h"
#include "tool_parser.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace agent::orch;

namespace {

// Mirrors the tools wired up in agent_orchestrator/default_tools.cpp. Kept here
// (rather than linked) so the playground stays parser-only; the UI uses this to
// build snippet buttons and to flag tool names the registry would reject.
const std::vector<std::pair<std::string, std::string>>& default_tools() {
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
    for (const auto& t : default_tools())
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

// Turn a ParsedAction from the real parser into a JSON the UI can render.
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

// Re-implements AgentOrchestrator::render_transcript() byte-for-byte so the
// "Rendered transcript" view shows exactly the string the model is fed.
std::string render_transcript(const json& messages) {
    auto role_tag = [](const std::string& r) -> const char* {
        if (r == "System")      return "SYSTEM";
        if (r == "User")        return "USER";
        if (r == "Assistant")   return "ASSISTANT";
        if (r == "Observation") return "OBSERVATION";
        return "UNKNOWN";
    };
    std::string out;
    for (const auto& m : messages) {
        out += '[';
        out += role_tag(m.value("role", std::string{}));
        out += "]\n";
        out += m.value("content", std::string{});
        out += "\n\n";
    }
    out += "[ASSISTANT]\n";  // trailing cue, exactly as the orchestrator emits
    return out;
}

extern const char* kIndexHtml;

}  // namespace

int main(int argc, char** argv) {
    int port = (argc > 1) ? std::atoi(argv[1]) : 8080;
    if (port <= 0 || port > 65535) port = 8080;

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(kIndexHtml, "text/html; charset=utf-8");
    });

    // Tool manifest -> drives the quick-insert toolbar.
    svr.Get("/api/tools", [](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (const auto& t : default_tools())
            arr.push_back({{"name", t.first}, {"description", t.second}});
        res.set_content(arr.dump(), "application/json");
    });

    // Ground-truth parse: runs the actual production ToolParser.
    svr.Post("/api/parse", [](const httplib::Request& req, httplib::Response& res) {
        std::string text;
        try {
            text = json::parse(req.body).value("text", std::string{});
        } catch (...) { /* tolerate junk -> empty text */ }
        res.set_content(action_to_json(ToolParser::parse(text)).dump(),
                        "application/json");
    });

    // Exact transcript the orchestrator would hand to the model.
    svr.Post("/api/render", [](const httplib::Request& req, httplib::Response& res) {
        json messages = json::array();
        try {
            json body = json::parse(req.body);
            if (body.contains("messages")) messages = body["messages"];
        } catch (...) {}
        res.set_content(json{{"transcript", render_transcript(messages)}}.dump(),
                        "application/json");
    });

    const std::string url = "http://localhost:" + std::to_string(port);
    printf("\n  Agent Playground running at %s\n", url.c_str());
    printf("  (press Ctrl+C to stop)\n\n");
    fflush(stdout);

#ifdef _WIN32
    // Best-effort: pop the default browser. Harmless if it fails.
    std::system(("start \"\" " + url).c_str());
#endif

    if (!svr.listen("127.0.0.1", port)) {
        fprintf(stderr, "ERROR: could not bind port %d (already in use?).\n", port);
        return 1;
    }
    return 0;
}

// ============================================================================
// Embedded single-page frontend. Self-contained (no external CDN/files) so the
// executable is fully standalone. ASCII-only on purpose; the snippet caret
// sentinel is produced with a JS \u escape to stay safe across MSVC raw strings.
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
    --ok:#4fd08a; --warn:#ff5c6c;
  }
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
    background:var(--bg);color:var(--txt)}
  code,pre,textarea,.mono{font-family:ui-monospace,"Cascadia Code",Consolas,monospace}
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
  .main{display:grid;grid-template-columns:1fr 380px;gap:14px;padding:14px;
    height:calc(100vh - 49px)}
  .col{display:flex;flex-direction:column;min-height:0;gap:12px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px}
  .card h2{font-size:11px;text-transform:uppercase;letter-spacing:.8px;
    color:var(--muted);margin:0;padding:9px 12px;border-bottom:1px solid var(--border)}
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
  .editor{position:relative;height:190px;margin:0}
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
  .composer .actions{display:flex;gap:8px;padding:10px 12px;align-items:center}
  .composer .actions .spacer{flex:1}
  .composer .actions .primary{background:var(--accent);color:#fff;border-color:transparent;
    font-weight:600;padding:6px 16px}
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
  #rendered{flex:1;min-height:0}
  #renderedOut{margin:0;padding:11px;overflow:auto;height:100%;white-space:pre-wrap;
    word-break:break-word;font-size:12px;color:var(--muted)}
  .insp-note{font-size:11px;color:var(--muted);margin-top:8px;border-top:1px solid var(--border);
    padding-top:8px}
</style>
</head>
<body>
<header>
  <h1>Agent Playground</h1>
  <span class="pill">XML ReAct simulator</span>
  <span class="spacer"></span>
  <button id="seedBtn">Load demo</button>
  <button id="renderBtn">Render transcript</button>
  <button id="copyBtn">Copy transcript</button>
  <button id="clearBtn">Clear</button>
</header>

<div class="main">
  <div class="col">
    <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0">
      <h2>Transcript</h2>
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
        <div class="gl">Green snippets = structural tags. Blue = full tool_call for a registered tool. The caret lands where you'll type next.</div>
      </div>

      <div class="editor">
        <pre aria-hidden="true"><code id="hl"></code></pre>
        <textarea id="input" spellcheck="false" placeholder="Type the turn here. Unknown tags (e.g. <file_content>) are flagged in red -- that's how you catch hallucinations."></textarea>
      </div>

      <div class="actions">
        <span class="hint" id="liveHint"></span>
        <span class="spacer"></span>
        <button id="addBtn" class="primary">Add turn &#9166;</button>
      </div>
    </div>
  </div>

  <div class="col">
    <div class="card" style="flex:0 0 auto">
      <h2>Parser inspector &mdash; ground truth (real ToolParser)</h2>
      <div id="inspector"><div class="hint">Start typing as the Assistant role to see what the orchestrator's parser extracts.</div></div>
    </div>
    <div class="card" id="rendered" style="display:flex;flex-direction:column">
      <h2>Rendered transcript (what the model sees)</h2>
      <pre id="renderedOut" class="mono">Click "Render transcript".</pre>
    </div>
  </div>
</div>

<script>
"use strict";
const CARET = String.fromCharCode(0x2038);                  // caret-placement sentinel for snippets
const KNOWN_TAGS = new Set(["tool_call","arg","finish"]);
const TOOL_ARGS = {                           // arg templates, mirrors default_tools.cpp
  read_file:["path"], write_file:["path","content"], patch_file:["path","find","replace"],
  list_dir:["path"], analyze_source:["source"], callers_of:["symbol"], callees_of:["symbol"]
};
let messages = [];
let currentRole = "Assistant";
let editIndex = -1;                           // index being re-edited, or -1

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
  return out + "\n";                          // trailing NL so the last line keeps height
}
function syncHL(){ hl.innerHTML = highlight(input.value); hl.parentElement.scrollTop = input.scrollTop; }
input.addEventListener("input", () => { syncHL(); liveParse(); });
input.addEventListener("scroll", () => { hl.parentElement.scrollTop = input.scrollTop; hl.parentElement.scrollLeft = input.scrollLeft; });

/* ---------- snippet insertion (cursor-aware) ---------- */
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

/* ---------- live parse against the REAL parser ---------- */
let parseTimer = null;
)HTMLDOC"
        R"HTMLDOC(function liveParse(){ clearTimeout(parseTimer); parseTimer = setTimeout(doParse, 120); }
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

/* ---------- actions ---------- */
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
$("#copyBtn").addEventListener("click", async () => {
  const r = await (await fetch("/api/render",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({messages})})).json();
  try{ await navigator.clipboard.writeText(r.transcript); $("#copyBtn").textContent = "Copied!"; setTimeout(()=>$("#copyBtn").textContent="Copy transcript",1200); }
  catch(e){ $("#renderedOut").textContent = r.transcript; }
});

/* ---------- build tool toolbar from the live manifest ---------- */
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

syncHL(); liveParse();
</script>
</body>
</html>
)HTMLDOC";
}  // namespace
