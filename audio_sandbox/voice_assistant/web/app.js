"use strict";
/* ---------------------------------------------------------------------------
   voice_assistant — messenger front-end.

   PROTOCOL. One JSON message per direction, over the WebView2 host channel:
     app -> page   phase | local.delta | local.final | user.text | remote.start
                   | remote.delta | remote.final | transport | backend | stats
                   | settings | settings.saved | browsed
     page -> app   ready | send | mic | browse | settings.save | restart

   WHO IS WHO IN THE BUBBLES. The LOCAL model transcribes what you said and
   decides whether it was a finished thought; the REMOTE model answers it. So
   local output is the USER's bubble (your words, appearing as you speak) and
   remote output is the ASSISTANT's. A typed message skips the local stage and
   lands as a user bubble immediately.

   RENDERING is ported from agent_playground: marked + highlight.js with the
   same renderer overrides, and the same rule that model-emitted HTML is escaped
   and shown rather than injected. What is new here is the FALLBACK -- a
   self-contained markdown renderer and highlighter used whenever the CDN is
   unreachable, because a local voice assistant that only formats code when it
   has internet is not actually a local voice assistant.
--------------------------------------------------------------------------- */

const $ = s => document.querySelector(s);
const host = window.chrome && window.chrome.webview ? window.chrome.webview : null;

function send(msg) { if (host) host.postMessage(msg); }

/* ====================== markdown + syntax highlighting ==================== */

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// highlight.js has no cuda/cu grammar; CUDA is a C++ superset. Aliases also let
// the fallback highlighter and hljs agree on which grammar a fence means.
const LANG_ALIAS = {
  cu: "cpp", cuda: "cpp", "c++": "cpp", cc: "cpp", h: "cpp", hpp: "cpp", c: "cpp",
  py: "python", python3: "python",
  js: "javascript", jsx: "javascript", ts: "javascript", tsx: "javascript",
  sh: "bash", shell: "bash", zsh: "bash", console: "bash", bat: "bash",
  yml: "yaml"
};
const normLang = l => LANG_ALIAS[String(l || "").toLowerCase()] || String(l || "").toLowerCase();

/* ---- fallback highlighter ----------------------------------------------
   One pass, one alternation: comment | string | number | keyword. Anything
   unmatched is escaped literal text, so the output is always balanced HTML no
   matter what the model emitted. Deliberately small -- it exists so the app is
   never worse than "readable" offline, not to compete with highlight.js. */
const KEYWORDS = {
  cpp: "alignas alignof and auto bool break case catch char class concept const consteval constexpr constinit continue co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept nullptr operator private protected public register reinterpret_cast requires return short signed sizeof static static_assert static_cast struct switch template this throw true try typedef typeid typename union unsigned using virtual void volatile while __global__ __device__ __host__ __shared__ size_t uint32_t int64_t",
  python: "and as assert async await break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield self",
  javascript: "async await break case catch class const continue debugger default delete do else export extends false finally for function if import in instanceof let new null return static super switch this throw true try typeof var void while yield of",
  bash: "if then else elif fi for while until do done case esac function return export local source echo cd set unset trap read",
  json: "true false null",
  yaml: "true false null yes no on off",
  sql: "select from where insert update delete join left right inner outer on group by order having limit as and or not null create table index into values set"
};

const HL_CACHE = {};
function grammar(lang) {
  if (HL_CACHE[lang]) return HL_CACHE[lang];
  const words = KEYWORDS[lang] || KEYWORDS.cpp;
  const hash = (lang === "python" || lang === "bash" || lang === "yaml");
  const parts = [
    hash ? "(#[^\\n]*)" : "(\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/)",       // 1 comment
    "(\"(?:[^\"\\\\\\n]|\\\\.)*\"|'(?:[^'\\\\\\n]|\\\\.)*'|`(?:[^`\\\\]|\\\\.)*`)", // 2 string
    "\\b(0[xX][0-9a-fA-F]+|\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?[fFuUlL]*)\\b",        // 3 number
    "\\b(" + words.trim().split(/\s+/).join("|") + ")\\b",                          // 4 keyword
    "\\b([A-Za-z_]\\w*)(?=\\s*\\()"                                                 // 5 call
  ];
  return (HL_CACHE[lang] = new RegExp(parts.join("|"), "g"));
}

function miniHighlight(code, lang) {
  const re = grammar(normLang(lang) || "cpp");
  re.lastIndex = 0;
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const cls = m[1] ? "tok-com" : m[2] ? "tok-str" : m[3] ? "tok-num"
              : m[4] ? "tok-kw" : "tok-fn";
    out += '<span class="' + cls + '">' + esc(m[0]) + "</span>";
    last = m.index + m[0].length;
    if (m[0].length === 0) re.lastIndex++;   // never spin on an empty match
  }
  return out + esc(code.slice(last));
}

const hljsHas = l => !!(window.hljs && l && hljs.getLanguage(l));

// Colourise one fenced block, preferring highlight.js and degrading to the
// built-in tokenizer. Always returns a complete <pre><code> element.
function renderCode(code, lang) {
  const norm = normLang(lang);
  const label = lang ? ' language-' + esc(String(lang).toLowerCase()) : "";
  if (norm && hljsHas(norm)) {
    try {
      const v = hljs.highlight(code, { language: norm, ignoreIllegals: true }).value;
      return '<pre><code class="hljs' + label + '">' + v + "</code></pre>";
    } catch (e) { /* fall through to the built-in */ }
  }
  return '<pre><code class="' + (label || " plain").trim() + '">' +
         miniHighlight(code, norm) + "</code></pre>";
}

/* ---- fallback markdown --------------------------------------------------
   Block level: fences, ATX headings, hr, blockquote, ul/ol, paragraphs.
   Inline: code, bold, italic, strikethrough, links. Everything is escaped
   BEFORE any markup is inserted, so model-emitted HTML is shown, not run. */
function inlineMd(text) {
  let s = esc(text);
  s = s.replace(/`([^`]+)`/g, (m, c) => "<code>" + c + "</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/(^|[^_\w])_([^_\n]+)_/g, "$1<em>$2</em>");
  s = s.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s;
}

function miniMarkdown(src) {
  const lines = String(src == null ? "" : src).split("\n");
  let out = "", i = 0;

  const flushList = (tag, items) =>
    "<" + tag + ">" + items.map(t => "<li>" + inlineMd(t) + "</li>").join("") + "</" + tag + ">";

  while (i < lines.length) {
    const line = lines[i];

    // fenced code — an UNTERMINATED fence still renders (the model is mid-stream)
    const fence = line.match(/^\s*```(\S*)\s*$/);
    if (fence) {
      const lang = fence[1];
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
      i++;  // consume the closing fence if there was one
      out += renderCode(body.join("\n"), lang);
      continue;
    }
    if (/^\s*$/.test(line)) { i++; continue; }
    const h = line.match(/^\s*(#{1,4})\s+(.*)$/);
    if (h) {
      const lvl = Math.min(h[1].length + 1, 4);
      out += "<h" + lvl + ">" + inlineMd(h[2]) + "</h" + lvl + ">";
      i++; continue;
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) { out += "<hr>"; i++; continue; }
    if (/^\s*>\s?/.test(line)) {
      const body = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ""));
      out += "<blockquote>" + miniMarkdown(body.join("\n")) + "</blockquote>";
      continue;
    }
    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*+]\s+/, ""));
      out += flushList("ul", items);
      continue;
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*\d+[.)]\s+/, ""));
      out += flushList("ol", items);
      continue;
    }
    // paragraph: consume until a blank line or the start of another block
    const para = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) &&
           !/^\s*```/.test(lines[i]) && !/^\s*#{1,4}\s/.test(lines[i]) &&
           !/^\s*>/.test(lines[i]) && !/^\s*[-*+]\s/.test(lines[i]) &&
           !/^\s*\d+[.)]\s/.test(lines[i])) {
      para.push(lines[i++]);
    }
    out += "<p>" + inlineMd(para.join("\n")).replace(/\n/g, "<br>") + "</p>";
  }
  return out;
}

/* ---- marked wiring (identical routing to agent_playground) --------------- */
let markedReady = false;
function tokenText(t) {
  if (t == null) return "";
  if (typeof t === "string") return t;
  return t.text != null ? t.text : (t.raw != null ? t.raw : "");
}
const firstWord = s => (s ? String(s).match(/\S*/)[0] : "");

if (window.marked && typeof marked.use === "function") {
  marked.use({
    gfm: true,
    breaks: true,           // chat: a newline the user sees IS a newline
    renderer: {
      // This marked build calls renderers positionally -- code(text, info) --
      // so accept both shapes.
      code(token, infostring) {
        const raw = token && typeof token === "object" && token.lang ? token.lang : infostring;
        return renderCode(tokenText(token), firstWord(raw));
      },
      // DO NOT escape here. marked delivers codespan content ALREADY escaped --
      // as a bare string in this build, as token.text in newer ones -- unlike
      // `code` above, whose text is raw (that asymmetry is what marked's third
      // `escaped` argument is about). Escaping again renders `<algorithm>` as a
      // literal "&lt;algorithm&gt;" in the bubble.
      codespan(token) { return "<code>" + tokenText(token) + "</code>"; },
      // Raw HTML the model emitted: ESCAPE and show it. Never inject live DOM
      // from model output into a window that has a native bridge attached.
      html(token) { return esc(tokenText(token)); }
    }
  });
  markedReady = true;
}

function renderMarkdown(content) {
  if (markedReady) {
    try { return marked.parse(content); } catch (e) { /* fall through */ }
  }
  return miniMarkdown(content);
}

/* ---- <think> reasoning cards (ported from agent_playground) --------------
   Reasoning models stream a chain of thought before the answer. Lifting it out
   into a collapsible card keeps the conversation readable: a finished thought
   collapses, one still streaming stays open so it can be watched. */
function renderThink(inner, done) {
  const lead = done ? "" : '<span class="spin"></span> ';
  return '<details class="think"' + (done ? "" : " open") + ">" +
           "<summary>" + lead + (done ? "Thought process" : "Thinking…") + "</summary>" +
           '<div class="think-body">' + renderMarkdown(inner) + "</div>" +
         "</details>";
}

function renderContent(content) {
  content = content == null ? "" : String(content);
  let out = "", last = 0, m;
  const re = /<think>([\s\S]*?)<\/think>/gi;
  while ((m = re.exec(content))) {
    if (m.index > last) out += renderMarkdown(content.slice(last, m.index));
    out += renderThink(m[1], true);
    last = re.lastIndex;
  }
  const rest = content.slice(last);
  const open = rest.search(/<think>/i);
  if (open >= 0) {
    if (open > 0) out += renderMarkdown(rest.slice(0, open));
    out += renderThink(rest.slice(open + 7), false);   // 7 == "<think>".length
  } else if (rest.length || !out) {
    out += renderMarkdown(rest);
  }
  return out;
}

/* ============================ transcript ================================= */

const chat = $("#chat");
const emptyState = $("#emptyState");
let liveUser = null;        // the bubble filling with live transcription
let lastUser = null;        // the most recent user bubble, live or finished
let liveAssistant = null;   // the bubble filling with the remote reply

function atBottom() {
  return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 120;
}
function scrollDown(force) {
  if (force || atBottom()) chat.scrollTop = chat.scrollHeight;
}

function addBubble(role, text, streaming) {
  if (emptyState && emptyState.parentNode) emptyState.remove();
  const row = document.createElement("div");
  row.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (streaming ? " streaming" : "");
  const md = document.createElement("div");
  md.className = "md";
  bubble.appendChild(md);
  row.appendChild(bubble);
  chat.appendChild(row);
  const handle = { row, bubble, md, text: "" };
  setText(handle, text || "");
  if (role === "user") lastUser = handle;
  scrollDown(true);
  return handle;
}

function setText(handle, text) {
  handle.text = text;
  handle.md.innerHTML = renderContent(text);
}
function appendText(handle, delta) {
  setText(handle, handle.text + delta);
  scrollDown(false);
}
function freeze(handle) {
  if (handle) handle.bubble.classList.remove("streaming");
}
function addNote(handle, text, kind) {
  if (!handle) return;
  const n = document.createElement("span");
  n.className = "note " + (kind || "");
  n.textContent = text;
  handle.bubble.appendChild(n);
}
function showTyping(handle) {
  handle.md.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
}

/* ============================== status ==================================== */

const statusEl = $("#status"), statusText = $("#statusText"), micBtn = $("#micBtn");
const PHASE_LABEL = {
  idle: "Idle",
  listening: "Listening",
  generating: "Transcribing",
  interrupted: "Interrupted",
  committed: "Thinking",
  dispatching: "Thinking"
};

function setPhase(phase) {
  const p = PHASE_LABEL[phase] ? phase : "idle";
  statusEl.dataset.phase = p;
  statusText.textContent = PHASE_LABEL[p];
  micBtn.classList.toggle("hearing", p === "listening" && micBtn.dataset.on === "true");
}

/* ============================== composer ================================== */

const input = $("#input"), sendBtn = $("#sendBtn");

// Collapse to 0 before measuring, NOT to "auto": the textarea is a flex item, so
// `auto` resolves against the stretched box on the very first call and reports a
// scrollHeight several lines tall -- which is how the composer opened three rows
// high on a freshly loaded, empty page.
function autoGrow() {
  input.style.height = "0px";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}
function refreshSend() {
  sendBtn.disabled = input.value.trim().length === 0;
}
input.addEventListener("input", () => { autoGrow(); refreshSend(); });
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
});
sendBtn.addEventListener("click", submit);

function submit() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  autoGrow();
  refreshSend();
  // The bubble is NOT painted here. The app echoes the message back as
  // `user.text`, so the transcript has one source of truth for both input paths
  // -- and that echo is a PostMessage on a thread already inside the message
  // loop, so it lands on the next pump, not perceptibly later.
  send({ type: "send", text });
}

micBtn.addEventListener("click", () => {
  const on = micBtn.dataset.on !== "true";
  micBtn.dataset.on = String(on);
  if (!on) micBtn.classList.remove("hearing");
  send({ type: "mic", on });
});

/* ============================== settings ================================== */

const overlay = $("#settingsOverlay");
const restartBanner = $("#restartBanner");
const saveMsg = $("#saveMsg");
const saveBtn = $("#settingsSave");

// id suffix -> {get, set}. One table drives load, save and the restart check, so
// a field cannot be persisted but forgotten by the banner (or the reverse).
const FIELDS = {
  model_dir: "value", audio_head: "value", projector_path: "value", data_dir: "value",
  system_prompt: "value", device_id: "int", max_context: "int",
  simulated: "check", neural_vad: "check", loopback_capture: "check",
  vad_threshold: "float", context_mode: "value",
  history_budget_tokens: "int", live_streaming: "check"
};

let current = {};            // what the app is actually running
let seeded = false;          // has the form ever been filled from `current`?

function el(k) { return $("#s_" + k); }

function readForm() {
  const out = {};
  for (const k in FIELDS) {
    const e = el(k);
    if (!e) continue;
    const kind = FIELDS[k];
    out[k] = kind === "check" ? e.checked
           : kind === "int"   ? (parseInt(e.value, 10) || 0)
           : kind === "float" ? (parseFloat(e.value) || 0)
           : e.value;
  }
  return out;
}

function writeForm(s) {
  for (const k in FIELDS) {
    const e = el(k);
    if (!e || s[k] === undefined) continue;
    if (FIELDS[k] === "check") e.checked = !!s[k];
    else e.value = s[k];
  }
  $("#vadValue").textContent = Number(s.vad_threshold || 0).toFixed(2);
  // Only a REAL payload counts as seeded. Opening the modal before the app's
  // first push writes an empty form, and marking that as seeded would lock the
  // blank state in for the rest of the session.
  if (Object.keys(s).length) seeded = true;
  refreshRestartBanner();
}

// A restart-tier field is any input tagged data-restart in the markup -- the
// same set settings_store.hpp's requires_restart() compares. Keeping the tag in
// the HTML means the banner is declarative rather than a second list to forget.
function restartNeeded() {
  const form = readForm();
  for (const k in FIELDS) {
    const e = el(k);
    if (!e || !e.hasAttribute("data-restart")) continue;
    if (String(form[k]) !== String(current[k])) return true;
  }
  return false;
}
function refreshRestartBanner() {
  const need = restartNeeded();
  restartBanner.classList.toggle("hidden", !need);
  saveBtn.textContent = need ? "Save & Restart" : "Save";
}

overlay.addEventListener("input", refreshRestartBanner);
overlay.addEventListener("change", () => {
  $("#vadValue").textContent = Number(el("vad_threshold").value).toFixed(2);
  refreshRestartBanner();
});
el("vad_threshold").addEventListener("input", () => {
  $("#vadValue").textContent = Number(el("vad_threshold").value).toFixed(2);
});

function openSettings() {
  writeForm(current);
  saveMsg.textContent = "";
  overlay.classList.remove("hidden");
}
function closeSettings() { overlay.classList.add("hidden"); }

$("#settingsBtn").addEventListener("click", openSettings);
$("#settingsClose").addEventListener("click", closeSettings);
$("#settingsCancel").addEventListener("click", closeSettings);
overlay.addEventListener("mousedown", e => { if (e.target === overlay) closeSettings(); });
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !overlay.classList.contains("hidden")) closeSettings();
});

document.querySelectorAll("[data-browse]").forEach(b =>
  b.addEventListener("click", () => send({ type: "browse", target: b.dataset.browse })));

saveBtn.addEventListener("click", () => {
  const need = restartNeeded();
  saveMsg.textContent = "Saving…";
  send({ type: "settings.save", payload: readForm() });
  if (need) send({ type: "restart" });
});

/* ============================ diagnostics ================================= */

function setStats(s) {
  $("#d_committed").textContent = s.committed;
  $("#d_bargein").textContent = s.barge_in;
  $("#d_tokencap").textContent = s.token_cap;
  $("#d_queuefull").textContent = s.queue_full;
  const capped = Number(s.token_cap) > 0;
  $("#d_capRow").classList.toggle("active", capped);
  $("#d_capNote").classList.toggle("hidden", !capped);
}

/* =========================== app -> page ================================== */

function onMessage(msg) {
  switch (msg.type) {
    case "phase":
      setPhase(msg.phase);
      break;

    case "local.delta":
      // The live transcription of what is being said right now.
      if (msg.restart || !liveUser) {
        freeze(liveUser);
        liveUser = addBubble("user", "", true);
      }
      appendText(liveUser, msg.text);
      break;

    case "local.final": {
      // Attach the verdict to the bubble this turn belongs to: the live one on
      // the voice path, the last user bubble on the typed path (where there is
      // no live transcription to freeze).
      const target = liveUser || lastUser;
      freeze(liveUser);
      liveUser = null;
      if (target && !msg.dispatched) {
        // No cloud call, no error anywhere else in the system. Say so here or it
        // is silence the user cannot explain.
        target.bubble.classList.add("unsent");
        addNote(target,
                msg.reason === "BargeIn" ? "Interrupted — not sent"
                                         : "Cut off before it finished — not sent",
                "warn");
      }
      setPhase(msg.phase);
      break;
    }

    case "user.text":
      freeze(liveUser);
      liveUser = null;
      addBubble("user", msg.text, false);
      if (msg.phase) setPhase(msg.phase);
      break;

    case "remote.start":
      freeze(liveAssistant);
      liveAssistant = addBubble("assistant", "", true);
      showTyping(liveAssistant);
      break;

    case "remote.delta":
      if (!liveAssistant) liveAssistant = addBubble("assistant", "", true);
      appendText(liveAssistant, msg.text);
      break;

    case "remote.final":
      if (liveAssistant) {
        // A reply that produced no text at all would otherwise leave the typing
        // dots spinning forever -- clear them and say what happened.
        if (!liveAssistant.text) {
          setText(liveAssistant, msg.ok ? "_(no reply)_" : "");
        }
        freeze(liveAssistant);
        if (!msg.ok) addNote(liveAssistant, msg.detail || "The reply failed.", "err");
        liveAssistant = null;
      }
      break;

    case "transport": {
      const label = $("#transportLabel");
      label.textContent = msg.live ? (msg.name || "Remote") + " · billed"
                                   : "Offline · " + (msg.name || "simulated");
      label.classList.toggle("live", !!msg.live);
      $("#d_transport").textContent = (msg.name || "—") + (msg.live ? " (live)" : " (offline)");
      break;
    }

    case "backend":
      $("#d_backend").textContent = msg.detail ? msg.name + " — " + msg.detail : msg.name;
      $("#d_audio").textContent = msg.audio_ready ? "Speech input ready" : "Text only";
      break;

    case "settings":
      current = msg.payload || {};
      // Refill only when the modal is closed -- an update that arrives while the
      // user is typing must not wipe their edits. The `seeded` escape hatch
      // covers the race where the modal was opened before the app's first push:
      // an empty form the user could Save would silently blank every setting.
      if (overlay.classList.contains("hidden") || !seeded) writeForm(current);
      else refreshRestartBanner();
      break;

    case "settings.saved":
      saveMsg.textContent = msg.live_only ? "Saved — applied now."
                                          : "Saved — restarting…";
      if (msg.live_only) setTimeout(closeSettings, 550);
      break;

    case "browsed": {
      const e = el(msg.target);
      if (e) { e.value = msg.path; refreshRestartBanner(); }
      break;
    }

    case "stats":
      setStats(msg);
      break;
  }
}

if (host) {
  host.addEventListener("message", e => {
    // WebView2 delivers PostWebMessageAsJson already parsed on `data`.
    let msg = e.data;
    if (typeof msg === "string") { try { msg = JSON.parse(msg); } catch (err) { return; } }
    if (msg && msg.type) onMessage(msg);
  });
}

setPhase("idle");
refreshSend();
autoGrow();
send({ type: "ready" });
input.focus();
