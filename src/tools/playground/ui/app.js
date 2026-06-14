"use strict";
const CARET = String.fromCharCode(0x2038);    // caret-placement sentinel for snippets
const KNOWN_TAGS = new Set(["tool_call","arg","finish"]);
const TOOL_ARGS = {
  read_file:["path"], write_file:["path","content"], patch_file:["path","find","replace"],
  list_dir:["path"], analyze_source:["source"], callers_of:["symbol"], callees_of:["symbol"]
};
// Branch TREE keyed by engine seq_id (0 = "main"). Each branch stores metadata
// (id, name, parentId, forkIndex) plus its OWN turns -- those added after its
// fork point. A branch's full transcript is reconstructed by climbing parentId to
// the root and stitching: fullHistory(parent)[0..forkIndex] + own. `messages`
// holds the materialized transcript of the ACTIVE branch (rebuilt on switch);
// in-place mutations to it are persisted back into the branch's own via syncOwn().
let branches = { 0: { id: 0, name: "main", parentId: null, forkIndex: null, own: [] } };
let currentBranch = 0;
let nextSeqId = 1;            // monotonic engine seq_id assigned to each new branch
let messages = [];           // materialized transcript of the active branch
let enginePaged = false;     // engine loaded with Paged (CoW) attention?
let currentRole = "Assistant";
let editIndex = -1;

const $ = s => document.querySelector(s);
const input = $("#input"), hl = $("#hl");

/* ---------- branch tree (Paged-attention CoW) ---------- */
const isRoot    = id => branches[id].parentId === null;
const prefixLen = id => isRoot(id) ? 0 : branches[id].forkIndex + 1;  // # inherited turns

// Reconstruct a branch's full transcript by stitching its inherited prefix
// (recursively climbing to the root) with its own turns. Returns fresh copies.
function fullHistory(id){
  const b = branches[id];
  const own = b.own.map(m => ({role:m.role, content:m.content}));
  return isRoot(id) ? own : fullHistory(b.parentId).slice(0, b.forkIndex + 1).concat(own);
}
// Persist the materialized `messages` back into the active branch: everything
// from its fork point on is "own" (the inherited prefix is read-only, so the
// slice is lossless).
function syncOwn(){ branches[currentBranch].own = messages.slice(prefixLen(currentBranch)); }

function loadBranch(id){ currentBranch = id; messages = fullHistory(id); }
function switchBranch(id){
  if(!(id in branches)) return;
  loadBranch(id);
  renderBranches();
  renderTranscript();
}
// Replace MAIN's whole conversation -- used by the demo seed and the autonomous
// ReAct loop, which both produce a fresh standalone transcript that belongs at
// the root rather than as a delta on some child branch.
function replaceMain(msgs){
  branches[0].own = msgs.map(m => ({role:m.role, content:m.content}));
  loadBranch(0);
  renderBranches();
  renderTranscript();
}

/* ---------- recursive tree rendering (nested <ul>/<li> + connector lines) ---- */
const childrenOf = pid =>
  Object.keys(branches).map(Number).filter(id => branches[id].parentId === pid).sort((a,b) => a-b);

function renderBranchNode(id){
  const b = branches[id];
  const li = document.createElement("li");
  li.className = "branch-node";
  const row = document.createElement("div");
  row.className = "branch-item" + (id === currentBranch ? " active" : "");
  const tag = isRoot(id) ? "" : ' <span class="forkat">@'+b.forkIndex+'</span>';
  row.innerHTML = '<span class="bname">'+esc(b.name)+tag+'</span><span class="bseq">seq '+id+'</span>';
  row.onclick = () => switchBranch(id);
  li.appendChild(row);
  const kids = childrenOf(id);
  if(kids.length){
    const ul = document.createElement("ul");
    ul.className = "branch-children";
    kids.forEach(k => ul.appendChild(renderBranchNode(k)));
    li.appendChild(ul);
  }
  return li;
}
function renderBranches(){
  const box = $("#branchList");
  if(!box) return;
  box.innerHTML = "";
  const root = document.createElement("ul");
  root.className = "branch-tree";
  root.appendChild(renderBranchNode(0));   // main is always the tree root
  box.appendChild(root);
}

// Fork a new branch from turn `msgIndex` of the active branch: the child inherits
// turns 0..msgIndex (via stitch) and starts with no own turns. CoW-fork the engine
// KV behind it, then mount it instantly as a visual child of the active branch.
async function forkFrom(msgIndex){
  if(!enginePaged){ $("#modelMsg").textContent = "Enable Paged Attention (reload the model) to fork branches."; return; }
  const parentSeq = currentBranch;
  const childSeq = nextSeqId++;
  let r;
  try {
    r = await (await fetch("/api/branch/fork",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({parent_seq_id: parentSeq, new_seq_id: childSeq})})).json();
  } catch(e){ $("#modelMsg").textContent = "fork request failed"; return; }
  if(!r.ok){ $("#modelMsg").textContent = r.message || "fork rejected"; return; }
  branches[childSeq] = { id: childSeq, name: "branch "+childSeq, parentId: parentSeq,
                         forkIndex: msgIndex, own: [] };
  switchBranch(childSeq);
}

/* ---------- XML syntax highlighting ---------- */
function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function hlTag(tag){
  const m = tag.match(/^<\/?([\w-]+)/);
  const name = m ? m[1] : "";
  const cls = KNOWN_TAGS.has(name) ? "tag-known" : "tag-unknown";
  // esc() the whole tag first, then colour attribute name="value" pairs.
  let inner = esc(tag).replace(/([\w-]+)=("[^"]*"|'[^']*')/g,
    '<span class="attr">$1</span>=<span class="val">$2</span>');
  return '<span class="'+cls+'">'+inner+'</span>';
}
// Core: escape text and wrap any <tag ...> in a colored span. Returns safe HTML.
// Operates purely on a raw string and always emits balanced spans, so its output
// can never corrupt surrounding markup.
function highlightInline(text){
  let out = "", last = 0, m; const re = /<\/?[\w-]+[^>]*>/g;
  while((m = re.exec(text))){ out += esc(text.slice(last,m.index)); out += hlTag(m[0]); last = m.index+m[0].length; }
  out += esc(text.slice(last));
  return out;
}
// Editor overlay variant: trailing newline keeps the last line's height.
function highlight(text){ return highlightInline(text) + "\n"; }
function syncHL(){ hl.innerHTML = highlight(input.value); hl.parentElement.scrollTop = input.scrollTop; }
input.addEventListener("input", () => { syncHL(); liveParse(); });
input.addEventListener("scroll", () => { hl.parentElement.scrollTop = input.scrollTop; hl.parentElement.scrollLeft = input.scrollLeft; });

/* ---------- Markdown + dual highlighter pipeline (renderer-level) -------------
   Two highlighters, applied to DISJOINT content so they never conflict:
     * highlight.js  -> fenced code blocks in a real programming language
                        (cpp, c, python, ...). Colourises the source.
     * XML highlighter (highlightInline) -> prose / raw-HTML tokens, inline code,
                        and xml/text/unlabelled blocks. Colours <tool_call> etc.
   Each markdown token is routed to exactly ONE of them in the renderer, so the
   XML highlighter never runs over hljs output (and vice-versa), and marked's
   <pre>/<code> structure is always well-formed by construction. */
let markedReady = false;
function tokenText(t){
  if (t == null) return "";
  if (typeof t === "string") return t;
  return (t.text !== undefined && t.text !== null) ? t.text
       : (t.raw  !== undefined && t.raw  !== null) ? t.raw : "";
}
function firstWord(s){ return s ? String(s).match(/\S*/)[0] : ""; }
function hljsHas(lang){ return !!(window.hljs && lang && hljs.getLanguage(lang)); }

if (window.marked && typeof marked.use === "function") {
  marked.use({
    gfm: true,
    breaks: false,
    renderer: {
      // Fenced/indented code block. This marked build calls renderers
      // positionally -- code(codeText, infostring, escaped) -- so the language is
      // the 2nd argument; we also accept a token object for forward-compatibility.
      code(token, infostring){
        const text = tokenText(token);
        const rawLang = (token && typeof token === "object" && token.lang) ? token.lang : infostring;
        const lang = firstWord(rawLang).toLowerCase();
        // Real source language -> highlight.js. We deliberately do NOT also run the
        // XML highlighter here: inside a ```cpp/```python fence the text is source
        // code, and our protocol tags only live in prose / xml-or-text blocks.
        if (lang && lang !== "xml" && lang !== "text" && hljsHas(lang)) {
          try {
            const out = hljs.highlight(text, { language: lang, ignoreIllegals: true }).value;
            return '<pre><code class="hljs language-' + esc(lang) + '">' + out + '</code></pre>';
          } catch (e) { /* fall through to XML highlighter */ }
        }
        // xml / text / unlabelled / unknown language -> XML tag highlighter.
        const cls = lang ? ' language-' + esc(lang) : '';
        return '<pre><code class="xmlhl' + cls + '">' + highlightInline(text) + '</code></pre>';
      },
      // Inline `code` -> XML highlighter (tool names, tags typed inline).
      codespan(token){
        return '<code>' + highlightInline(tokenText(token)) + '</code>';
      },
      // Raw HTML the model emitted -- e.g. our XML tags. Escape + highlight so the
      // tags are VISIBLE and coloured rather than injected as (invisible) live DOM.
      // Multi-line blocks (a standalone <tool_call>...) become a monospaced card.
      html(token){
        const raw = tokenText(token);
        const inner = highlightInline(raw);
        return raw.indexOf("\n") >= 0 ? '<div class="xmlblock">' + inner + '</div>' : inner;
      }
    }
  });
  markedReady = true;
}
function renderMarkdown(content){
  if (markedReady) {
    try { return marked.parse(content); }
    catch (e) { /* fall through to plain rendering */ }
  }
  return '<pre class="plain">' + highlightInline(content) + '</pre>';
}

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
  if(!messages.length){ box.innerHTML = '<div class="empty-state">No turns yet. Pick a role, type or click a snippet, then "Add turn".</div>'; return; }
  box.innerHTML = "";
  const inheritedCount = prefixLen(currentBranch);   // turns inherited from ancestors
  for(let i=0;i<messages.length;i++){
    const m = messages[i];
    const inherited = i < inheritedCount;             // belongs to an ancestor branch
    const el = document.createElement("div");
    el.className = "msg role-"+m.role + (inherited ? " inherited" : "");
    let badge = "";
    if(m.role === "Assistant") badge = await badgeFor(m.content);
    // Inherited turns can be forked from but not edited/deleted (they live on the
    // parent branch); the branch's own turns get the full action set.
    const fork = '<button data-fork="'+i+'" title="Fork a new branch from this turn">fork</button>';
    const actions = inherited ? fork
      : fork + '<button data-edit="'+i+'">edit</button><button data-del="'+i+'">delete</button>';
    el.innerHTML =
      '<div class="avatar">'+m.role.charAt(0)+'</div>'+
      '<div class="bubble">'+
        '<div class="meta"><span class="who">'+m.role+'</span>'+badge+
          '<span class="actions">'+actions+'</span></div>'+
        '<div class="md">'+renderMarkdown(m.content)+'</div>'+
      '</div>';
    box.appendChild(el);
  }
  box.scrollTop = box.scrollHeight;
  box.querySelectorAll('[data-fork]').forEach(b => b.onclick = () => forkFrom(+b.dataset.fork));
  box.querySelectorAll('[data-del]').forEach(b => b.onclick = () => { messages.splice(+b.dataset.del,1); syncOwn(); renderTranscript(); });
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
  syncOwn();
  input.value = ""; syncHL(); liveParse(); renderTranscript();
});
input.addEventListener("keydown", e => { if((e.ctrlKey||e.metaKey) && e.key === "Enter"){ e.preventDefault(); $("#addBtn").click(); } });
$("#clearBtn").addEventListener("click", () => { branches[currentBranch].own = []; loadBranch(currentBranch); renderTranscript(); $("#renderedOut").textContent = 'Click "Render transcript".'; });
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

/* ---------- demo seed (markdown + cpp code + XML, to show the pipeline) ---------- */
$("#seedBtn").addEventListener("click", () => {
  replaceMain([
    {role:"System", content:"You are a coding agent. Act using exactly one <tool_call name=\"...\"><arg name=\"...\">...</arg></tool_call> per turn, or <finish>...</finish> when done."},
    {role:"User", content:"Read main.cpp and tell me what it does."},
    {role:"Assistant", content:"I'll open the file.\n<tool_call name=\"read_file\">\n  <arg name=\"path\">main.cpp</arg>\n</tool_call>"},
    {role:"Observation", content:"int main(){ return 0; }"},
    {role:"Assistant", content:"Here's the summary:\n\n```cpp\n#include <vector>\nint main() {\n  std::vector<int> v{1, 2, 3};  // <-- exit code below\n  return 0;\n}\n```\n\nAnd the Python equivalent:\n\n```python\ndef main() -> int:\n    return 0  # done\n```\n\n**main.cpp** is an empty entry point that returns `0`.\n<finish>main.cpp is an empty entry point that returns 0.</finish>"}
  ]);
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
    max_new_tokens: parseInt($("#maxtok").value) || 2048
  };
}
function setEngineUI(s){
  engineState = s.state;
  enginePaged = !!s.paged;
  const bh = $("#branchHint");
  if(bh) bh.textContent = enginePaged
    ? 'Click "fork" on a turn to branch from it.'
    : 'Load a model with Paged Attention to branch conversations (CoW).';
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
/* ---------- live streaming preview ----------
   A throwaway "Assistant" bubble appended to the transcript while the engine is
   decoding. It is NOT part of `messages`: renderTranscript() rebuilds #transcript
   from scratch and would wipe it, so we only ever poke it directly here and tear
   it down before the real (completed) turn is committed. */
function ensureStreamEl(){
  const box = $("#transcript");
  let el = $("#streamMsg");
  if(!el){
    const empty = box.querySelector(".empty-state");
    if(empty) empty.remove();
    el = document.createElement("div");
    el.id = "streamMsg";
    el.className = "msg role-Assistant streaming";
    el.innerHTML =
      '<div class="avatar">A</div>'+
      '<div class="bubble">'+
        '<div class="meta"><span class="who">Assistant</span>'+
          '<span class="badge muted"><span class="spin"></span> streaming&hellip;</span></div>'+
        '<div class="md"></div>'+
      '</div>';
    box.appendChild(el);
  }
  return el;
}
function updateStreamPreview(text){
  const el = ensureStreamEl();
  // renderMarkdown handles partial input: an unterminated ```cpp fence still
  // formats as a (growing) code block, so the user watches it type out.
  el.querySelector(".md").innerHTML = renderMarkdown(text || "");
  const box = $("#transcript");
  box.scrollTop = box.scrollHeight;
}
function clearStreamPreview(){
  const el = $("#streamMsg");
  if(el) el.remove();
}

let pollTimer = null;
function schedulePoll(busy){ clearTimeout(pollTimer); pollTimer = setTimeout(pollStatus, busy ? 150 : 800); }

async function pollStatus(){
  let s;
  try{ s = await (await fetch("/api/status")).json(); }catch(e){ schedulePoll(false); return; }
  if(!seqInit){ lastSeq = s.result_seq; seqInit = true; }   // ignore pre-existing result
  setEngineUI(s);

  // Live token stream: paint the partial output while a job we kicked off runs.
  if(awaiting && s.state === "generating") updateStreamPreview(s.stream);

  if(s.result_seq > lastSeq){
    lastSeq = s.result_seq;
    const kind = s.last_kind, was = awaiting;
    awaiting = null;
    clearStreamPreview();   // hand off from the live bubble to the committed turn
    if(kind === "generate" && was === "generate"){
      if(s.last_output){ messages.push({role:"Assistant", content:s.last_output}); syncOwn(); renderTranscript(); }
      if(s.last_chatml){ $("#chatmlBox").style.display=""; $("#chatmlOut").textContent = s.last_chatml; }
    } else if(kind === "react" && was === "react"){
      // ReAct produces a fresh standalone conversation -> it lands on main.
      if(Array.isArray(s.last_react) && s.last_react.length){
        replaceMain(s.last_react.map(m => ({role:m.role, content:m.content})));
      }
    }
    setEngineUI(s);  // re-enable buttons now that awaiting cleared
  }
  // Poll fast while the engine is busy (loading/generating) so the stream is
  // smooth; idle back off to keep the status endpoint cheap.
  schedulePoll(s.busy);
}
pollStatus();

$("#loadBtn").addEventListener("click", async () => {
  const path = $("#modelPath").value.trim();
  if(!path){ $("#modelMsg").textContent = "Enter a checkpoint directory first."; return; }
  const usePaged = $("#usePaged").checked;
  // A fresh load resets the engine's seq ids; collapse the tree to a single
  // 'main' branch (keeping the currently visible transcript as its history) so
  // UI seq ids match the engine again.
  const keep = messages.map(m => ({role:m.role, content:m.content}));
  branches = { 0: { id: 0, name: "main", parentId: null, forkIndex: null, own: keep } };
  currentBranch = 0; nextSeqId = 1; loadBranch(0);
  renderBranches(); renderTranscript();
  $("#modelMsg").textContent = "Requesting load...";
  const r = await (await fetch("/api/model/load",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({path, use_paged_attention: usePaged})})).json();
  if(!r.ok) $("#modelMsg").textContent = r.message;
  pollStatus();
});

$("#genBtn").addEventListener("click", async () => {
  if(engineState !== "ready") return;
  awaiting = "generate";
  $("#genBtn").disabled = true; $("#reactBtn").disabled = true;
  $("#genStatus").innerHTML = '<span class="spin"></span> generating...';
  const body = Object.assign({messages, seq_id: currentBranch}, params());
  const r = await (await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)})).json();
  if(!r.ok){ awaiting = null; $("#genStatus").textContent = ""; $("#modelMsg").textContent = r.message; }
  pollStatus();  // immediately reflect new state + start fast polling for the stream
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
  if(!r.ok){ awaiting = null; $("#genStatus").textContent = ""; $("#modelMsg").textContent = r.message; }
  pollStatus();  // immediately reflect new state + start fast polling for the stream
});

$("#forkMainBtn").addEventListener("click", () => {
  if(!messages.length){ $("#modelMsg").textContent = "Nothing to fork yet -- add a turn first."; return; }
  forkFrom(messages.length - 1);   // fork from the end of the current branch
});

renderBranches();
syncHL(); liveParse();
