"use strict";
/* ---------------------------------------------------------------------------
   voice_assistant — messenger front-end.

   PROTOCOL. One JSON message per direction, over the WebView2 host channel:
     app -> page   phase | local.delta | local.final | user.text | remote.start
                   | remote.delta | remote.final | transport | backend | stats
                   | settings | settings.saved | system_prompt.applied | mic
                   | browsed | audio.devices
     page -> app   ready | send | interrupt_generation | mic | browse
                   | settings.save | restart | audio.devices.request

   MIC STATE IS THE APP'S, not this page's: the talk hotkey is global and can
   flip it while the window is hidden, so the button RENDERS `mic` events and
   only ever sends one in response to a click. Settings are the same shape --
   the page never decides what is restart-tier, it renders the list the app
   pushes (see restartKeys). GENERATION STATE follows the identical rule (see
   `generating`): the page asks to stop and waits to be told the turn ended.

   WHO IS WHO IN THE BUBBLES. The LOCAL model transcribes what you said and
   decides whether it was a finished thought; the REMOTE model answers it. So
   local output is the USER's bubble (your words, appearing as you speak) and
   remote output is the ASSISTANT's. A typed message skips the local stage and
   lands as a user bubble immediately.

   ONE ANSWER, TWO AUDIENCES. The reply the model produces is tagged
   <voice>…</voice><ui>…</ui> and split BEFORE it reaches this page: the
   remote.delta stream carries the <ui> half only, and the <voice> half goes to
   the speaker. So nothing here parses tags, and a reply that arrives untagged
   (a model ignoring the contract) is displayed in full exactly as before --
   see reply_split.hpp for the contract and its fail-open behaviour.

   RENDERING is ported from agent_playground: marked + highlight.js with the
   same renderer overrides, and the same rule that model-emitted HTML is escaped
   and shown rather than injected. Both libraries are VENDORED into web/vendor/
   and loaded from there -- a local voice assistant that only formats code when
   it has internet is not actually a local voice assistant.

   THE FALLBACK BELOW STAYS ANYWAY. It was written for a missing CDN and now
   covers a missing or corrupt vendor file, which is a smaller risk but not a
   zero one -- and it is what makes the two <script> tags in index.html
   genuinely optional rather than load-bearing. Every `window.marked` /
   `window.hljs` test in this file is that contract.
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

/* THE TRANSCRIPT IS INSERTION-ORDERED, AND DELIBERATELY SO. Bubbles are DOM
   nodes appended in the order their messages arrive; nothing here sorts, and
   nothing here reads a clock. A wall-clock sort would be actively wrong: the
   messages are produced on four threads (engine, audio/VAD, dispatcher, UI) and
   stamping them would only record a race faithfully instead of preventing it.
   The app hands us one FIFO queue -- see AssistantWindow::post_event -- and the
   order it hands us IS the conversation. Do not add a sort.

   THE ONE ORDERING RULE THIS PAGE ENFORCES ITSELF: an answer may never be the
   first bubble of a turn. The app is supposed to publish the user's words before
   anything can announce a reply to them (WhisperCascadeMode::publish is where
   that is guaranteed for the voice path), but "supposed to" spans three threads,
   so the page also refuses to render an answer above its question -- see
   orphanAssistant below. */
const chat = $("#chat");
const emptyState = $("#emptyState");
let liveUser = null;        // the bubble filling with live transcription
let lastUser = null;        // the most recent user bubble, live or finished
let liveAssistant = null;   // the bubble filling with the remote reply

/* An assistant bubble that was created with no user bubble above it -- i.e. the
   reply reached the page before the question did. The next user bubble is
   inserted BEFORE it rather than appended, which repairs the order in place with
   no re-render and no flicker.

   Bounded to exactly one turn: it is set at remote.start (only when the answer
   genuinely has no question above it) and cleared the moment it is used or the
   reply ends. It therefore cannot reach forward and hijack the next turn's user
   bubble -- which is the failure mode a looser rule would have. The dispatcher
   serialises requests one at a time, so there is never a second turn in flight
   to confuse it with. */
let orphanAssistant = null;

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

  // The repair described at orphanAssistant: a user bubble that arrives after
  // the answer it prompted is placed above that answer, not below it.
  if (role === "user" && orphanAssistant && orphanAssistant.row.parentNode === chat) {
    chat.insertBefore(row, orphanAssistant.row);
    orphanAssistant = null;
  } else {
    chat.appendChild(row);
  }

  const handle = { row, bubble, md, text: "" };
  setText(handle, text || "");
  if (role === "user") {
    lastUser = handle;
  } else {
    // A turn always opens with the user -- typed messages arrive as `user.text`
    // and spoken ones as `local.delta`, and an utterance that produced no text
    // is never dispatched. So an assistant bubble whose predecessor is not a
    // user row is an answer that outran its question; remember it so the
    // question can be slotted in above when it lands.
    const prev = row.previousElementSibling;
    orphanAssistant = prev && prev.classList.contains("user") ? null : handle;
  }
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

/* ============================== session drawer ============================
   The app owns which conversation is active; this panel only asks. Nothing here
   mutates the transcript directly -- a click sends `session.select` and the
   REPAINT arrives back as `session.restore`, exactly like the composer paints
   nothing and waits for `user.text`. One source of truth for what is on screen,
   on every path.                                                            */

const drawer = $("#drawer");
const drawerScrim = $("#drawerScrim");
const sessionList = $("#sessionList");
const drawerEphemeral = $("#drawerEphemeral");
let activeSession = null;
let sessions = [];

function openDrawer() {
  drawer.classList.remove("hidden");
  drawerScrim.classList.remove("hidden");
  $("#drawerBtn").setAttribute("aria-expanded", "true");
  // Re-asked on every open, not cached: previews and ordering change with every
  // answered turn, and a drawer that shows yesterday's ordering is worse than
  // one that takes a frame to fill.
  send({ type: "session.list_request" });
}
function closeDrawer() {
  drawer.classList.add("hidden");
  drawerScrim.classList.add("hidden");
  $("#drawerBtn").setAttribute("aria-expanded", "false");
}
function toggleDrawer() {
  if (drawer.classList.contains("hidden")) openDrawer(); else closeDrawer();
}

$("#drawerBtn").addEventListener("click", toggleDrawer);
$("#drawerClose").addEventListener("click", closeDrawer);
drawerScrim.addEventListener("click", closeDrawer);
document.addEventListener("keydown", e => {
  // Escape closes the drawer, but never out from under the settings modal --
  // that dialog has its own handler and its own idea of what Escape means.
  if (e.key === "Escape" && !drawer.classList.contains("hidden") &&
      overlay.classList.contains("hidden")) {
    closeDrawer();
  }
});

$("#newChatBtn").addEventListener("click", () => {
  send({ type: "session.new" });
  closeDrawer();
});

// "14:32" today, "Mar 4" this year, "Mar 4, 2025" beyond it. Absolute rather
// than "3 hours ago": relative times need a ticking re-render to stay true, and
// this list is drawn once per open.
function sessionWhen(epochSeconds) {
  if (!epochSeconds) return "";
  const d = new Date(epochSeconds * 1000);
  if (isNaN(d.getTime())) return "";
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const opts = { month: "short", day: "numeric" };
  if (d.getFullYear() !== now.getFullYear()) opts.year = "numeric";
  return d.toLocaleDateString([], opts);
}

function sessionLabel(s) {
  // The preview is the name. Falling back to the id is deliberate and not a
  // placeholder: an id like "chat-1770384750-0" is ugly but it is UNIQUE, which
  // is the one property a row must have to be clickable at all.
  return (s.preview && s.preview.trim()) || s.id;
}

function renderSessions() {
  sessionList.textContent = "";

  // The active conversation always has a row, even before it has been persisted
  // -- a new chat is not written to disk until its first turn is answered, and a
  // sidebar that could not show where you ARE would be lying by omission.
  const rows = sessions.slice();
  if (activeSession && !rows.some(s => s.id === activeSession)) {
    rows.unshift({ id: activeSession, preview: "", turns: 0, updated_at: 0, pending: true });
  }

  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "session-empty";
    empty.textContent = "No saved conversations yet.";
    sessionList.appendChild(empty);
    return;
  }

  for (const s of rows) {
    const row = document.createElement("button");
    row.className = "session-row" + (s.id === activeSession ? " active" : "") +
                    (s.pending ? " pending" : "");
    row.type = "button";
    row.setAttribute("role", "listitem");

    const name = document.createElement("span");
    name.className = "session-name";
    name.textContent = s.pending ? "New chat" : sessionLabel(s);
    row.appendChild(name);

    const meta = document.createElement("span");
    meta.className = "session-meta";
    const when = sessionWhen(s.updated_at);
    const count = s.turns ? s.turns + (s.turns === 1 ? " message" : " messages") : "Empty";
    meta.textContent = when ? count + " · " + when : count;
    row.appendChild(meta);

    row.addEventListener("click", () => {
      if (s.id !== activeSession) send({ type: "session.select", id: s.id });
      closeDrawer();
    });

    // A pending session has nothing on disk to delete, so it gets no button --
    // the row would otherwise offer to remove something that does not exist.
    if (!s.pending) {
      const del = document.createElement("button");
      del.className = "session-del";
      del.type = "button";
      del.title = del.ariaLabel = "Delete this conversation";
      del.innerHTML =
        '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">' +
        '<path fill="currentColor" d="M9 3h6l1 2h4v2H4V5h4l1-2M6 9h12l-1 12H7L6 9Z"/></svg>';
      // stopPropagation, or the click also opens the conversation being deleted.
      del.addEventListener("click", e => {
        e.stopPropagation();
        send({ type: "session.delete", id: s.id });
      });
      row.appendChild(del);
    }

    sessionList.appendChild(row);
  }
}

/* Rebuilds the whole transcript from a session's stored turns.

   REPLACES rather than appends, and clears every live handle with it: the
   bubbles those handles pointed at are about to be removed from the DOM, so a
   delta arriving mid-switch must not find one of them still bound. */
function restoreTranscript(turns) {
  liveUser = null;
  lastUser = null;
  liveAssistant = null;
  orphanAssistant = null;
  chat.textContent = "";

  if (!turns || !turns.length) {
    // Put the welcome panel back. addBubble() detaches it rather than deleting
    // it, so the same node is reusable -- and an empty new chat that showed a
    // blank void instead would read as a broken window.
    if (emptyState) chat.appendChild(emptyState);
    return;
  }
  for (const t of turns) {
    if (t.user) freeze(addBubble("user", t.user, false));
    if (t.assistant) freeze(addBubble("assistant", t.assistant, false));
  }
  // The restored transcript is history, so it opens where a conversation is
  // resumed: at the end.
  scrollDown(true);
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

/* IS A REPLY BEING GENERATED RIGHT NOW?

   The composer's send button becomes a Stop button while this is true, which is
   the only interrupt a mouse user has -- the cancel gesture was a global hotkey
   and nothing else.

   THE APP OWNS THIS FLAG'S TRUTH, not the page. It is raised by the two events
   that mean a turn has started (`user.text` for a typed one, `remote.start` for
   the answer stream) and lowered by the one event that means it has finished
   (`remote.final`) -- no timers, no optimistic clearing when Stop is clicked.
   Clearing it on the click would be the tempting version and it is wrong: the
   stop may land after the last token, the transport may already be draining,
   and a button that says "sent" while audio is still playing is a lie the user
   can hear. `remote.final` arrives on every one of those paths, including the
   cancelled one.

   `local.final` with dispatched=false lowers it too: that is the gate refusing
   an utterance, so no answer stream is coming and nothing would ever clear it. */
let generating = false;

function setGenerating(on) {
  if (generating === on) return;
  generating = on;
  sendBtn.dataset.mode = on ? "stop" : "send";
  sendBtn.title = sendBtn.ariaLabel = on ? "Stop generating" : "Send";
  refreshSend();
}

// Collapse to 0 before measuring, NOT to "auto": the textarea is a flex item, so
// `auto` resolves against the stretched box on the very first call and reports a
// scrollHeight several lines tall -- which is how the composer opened three rows
// high on a freshly loaded, empty page.
function autoGrow() {
  input.style.height = "0px";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}
function refreshSend() {
  // Stop is ALWAYS enabled: an empty composer is the normal state while an
  // answer streams, and that is exactly when the button has to be clickable.
  sendBtn.disabled = !generating && input.value.trim().length === 0;
}
input.addEventListener("input", () => { autoGrow(); refreshSend(); });
input.addEventListener("keydown", e => {
  // Enter still SENDS while generating -- it does not stop. A key that means
  // "go" must not silently start meaning "abort" depending on timing; stopping
  // is a deliberate gesture and it has a deliberate target.
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
});
sendBtn.addEventListener("click", () => { if (generating) interrupt(); else submit(); });

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

/* Stop the decode, the network transfer and the speaker. Fire-and-forget: the
   button's state is NOT flipped back here -- see `generating` on why the app's
   `remote.final` is the only thing allowed to do that. */
function interrupt() {
  send({ type: "interrupt_generation" });
}

/* MIC MUTE. Routed through `mic`, NOT audio.hot_update, and that is deliberate:
   the same message is what the talk hotkey sends, so the button and the hotkey
   arrive at one handler and cannot disagree about the state. It is not a
   settings path either -- the app answers it with set_manual_mode, one atomic
   store, no engine work and nothing written to disk.

   aria-pressed is kept in step with data-on because the visual state is a CSS
   strike-through, which a screen reader cannot see. */
function paintMic(on) {
  micBtn.dataset.on = String(on);
  micBtn.setAttribute("aria-pressed", String(!on));
  micBtn.title = micBtn.ariaLabel = on ? "Mute the microphone" : "Unmute the microphone";
  if (!on) micBtn.classList.remove("hearing");
}

micBtn.addEventListener("click", () => {
  const on = micBtn.dataset.on !== "true";
  paintMic(on);
  send({ type: "mic", on });
});

/* ============================== settings ================================== */

const overlay = $("#settingsOverlay");
const restartBanner = $("#restartBanner");
const saveMsg = $("#saveMsg");
const saveBtn = $("#settingsSave");

// id suffix -> coercion. ONE table drives reading and writing the form, so a
// field cannot be persisted but forgotten on load (or the reverse). The types
// have to match settings_store.hpp's, because a value that arrives as a string
// where C++ wants an int is dropped by from_json and silently keeps its old
// value -- which looks exactly like a setting that will not stick.
const FIELDS = {
  // Tab 1
  model_dir: "value", audio_head: "value", projector_path: "value", data_dir: "value",
  device_id: "int", max_context: "int", simulated: "check",
  // Which speech-to-text pipeline runs, and the cascade's own knobs. All
  // restart-tier on the C++ side (they decide what gets ALLOCATED at bring-up),
  // so the banner picks them up from `restart_fields` with nothing to add here.
  // whisper_language stays "value": it is an ISO code the C++ side lowercases
  // and validates, and coercing it here would only be a second opinion.
  pipeline_mode: "value", whisper_model_path: "value", whisper_language: "value",
  whisper_threads: "int", whisper_max_utterance_ms: "int",
  temperature: "float", top_p: "float", max_new_tokens: "int",
  local_inference: "check",
  // The remote leg. All three are plain strings on the C++ side, so "value" --
  // the key in particular must NOT be coerced: it is opaque to us and any
  // parsing would be a way to corrupt it.
  remote_api_url: "value", remote_api_key: "value", remote_model: "value",
  // Tab 2
  neural_vad: "check", loopback_capture: "check", vad_threshold: "float",
  silence_hangover_ms: "int", pre_roll_ms: "int", warm_prefill_interval_ms: "int",
  context_mode: "value", history_budget_tokens: "int", live_streaming: "check",
  // Tab 2 — TTS. Types MUST match settings_store.hpp: a value arriving as a
  // string where C++ wants an int is dropped by from_json and silently keeps its
  // old value, which looks exactly like a setting that will not stick.
  tts_ckpt_dir: "value", tts_vocab_path: "value", tts_ref_audio: "value",
  tts_ref_text: "value", tts_nfe_step: "int",
  tts_split_on_commas: "check", tts_min_chunk_chars: "int",
  tts_max_chunk_chars: "int",
  // "float" (not "int") is load-bearing: tts_volume arriving as an integer would
  // be 0 or 1 and nothing between, i.e. a mute switch wearing a slider.
  tts_volume: "float", mic_gain: "float",
  // The four audio-endpoint settings (output/input x name/index) are NOT here.
  // They are driven by two <select>s whose options come from
  // GET /api/audio-devices, and one dropdown writes two settings, which the
  // one-id-one-field table above cannot express. See readDevices/writeDevices.
  aec_enabled: "check", aec_tail_ms: "int",
  // Tab 3
  hotkey_talk: "value", hotkey_cancel: "value", hotkey_show: "value",
  hotkey_push_to_talk: "check",
  // Tab 4 — the two prompts are separate fields, not one blob (see index.html).
  // speech_language is free text on purpose: it names a language to the model,
  // not an index into a list this page would have to keep in step with C++.
  system_prompt: "value", audio_task_prompt: "value", speech_language: "value",
  speech_task: "value"
};

// Sliders whose numeric value is echoed next to the label. Kept as a table for
// the same reason FIELDS is one: a slider without a readout is a slider nobody
// can set deliberately.
const READOUTS = {
  vad_threshold:            ["#vadValue",      v => Number(v).toFixed(2)],
  temperature:              ["#tempValue",     v => Number(v) === 0 ? "greedy" : Number(v).toFixed(2)],
  top_p:                    ["#toppValue",     v => Number(v) >= 1 ? "off" : Number(v).toFixed(2)],
  silence_hangover_ms:      ["#hangoverValue", v => Number(v) === 0 ? "off" : Number(v) + " ms"],
  pre_roll_ms:              ["#prerollValue",  v => Number(v) === 0 ? "off" : Number(v) + " ms"],
  warm_prefill_interval_ms: ["#warmValue",     v => Number(v) === 0 ? "off" : Number(v) + " ms"],
  // Chunk sizes read as characters, not an abstract scale -- the whole point of
  // the two knobs is "how much text before it starts talking".
  mic_gain:                 ["#micGainValue",  v => Math.round(Number(v) * 100) + "%"],
  tts_nfe_step:             ["#nfeValue",      v => Number(v) + " steps"],
  tts_min_chunk_chars:      ["#minChunkValue", v => Number(v) + " chars"],
  tts_max_chunk_chars:      ["#maxChunkValue", v => Number(v) + " chars"],
  // Milliseconds of room, not an abstract scale: it is a physical length the
  // canceller can model, and reading it as one is what makes it settable.
  aec_tail_ms:              ["#aecTailValue",  v => Number(v) + " ms"],
  // Percent, not the raw 0-1 gain: nobody sets loudness in linear amplitude.
  tts_volume:               ["#volumeValue",   v => Number(v) === 0
                                                    ? "muted"
                                                    : Math.round(Number(v) * 100) + "%"]
};

// Must match settings_store.hpp's defaults -- "Restore defaults" that restored
// something the app never shipped with would be worse than no button.
const DEFAULT_SYSTEM_PROMPT =
  "You are a concise voice assistant. Answer in one or two short sentences.";
const DEFAULT_AUDIO_TASK_PROMPT = "Transcribe the following speech exactly as spoken: ";

let current = {};            // what the app is actually running
let restartKeys = [];        // pushed by the app; see refreshRestartBanner
let seeded = false;          // has the form ever been filled from `current`?

function el(k) { return $("#s_" + k); }

/* ---------------------------- audio endpoints -----------------------------
   Two <select>s, populated from GET /api/audio-devices, replacing what used to
   be four hand-typed fields (a name and an index, per direction).

   ONE DROPDOWN WRITES TWO SETTINGS, which is why these are handled here instead
   of in the FIELDS table. What it writes, and why:

     "(system default)"  -> name "", index -1.
     a device            -> name = the exact enumerated string, index -1.
     a DUPLICATE name    -> name = the string, index = its real position.

   Preferring the NAME is the whole point of having picked from a list: the
   string is now exact, so the substring matching that made hand-typed names
   risky never comes into play, and a name keeps pointing at the right hardware
   after the OS reorders its endpoints. An index does not survive that.

   The duplicate case is the exception because it has to be. Two identical USB
   headsets enumerate under one name, and an exact-name match that hits both is
   treated as NO match by the resolver (deliberately -- see ma_device_select.h).
   Only the position can separate them, so there the index is pinned and the
   ordering fragility is accepted: a wrong-but-plausible device beats a silent
   fallback to the default. */
const DEVICE_SELECTS = {
  output: { el: "s_output_device", name: "output_device_name", index: "output_device_index" },
  input:  { el: "s_input_device",  name: "input_device_name",  index: "input_device_index"  }
};

// null until the fetch resolves. Kept so writeDevices() can re-select the right
// option when settings arrive after the list (or before it -- either order
// happens, and both have to end up showing the same thing).
let deviceLists = null;

function optionLabel(list, i) {
  // A duplicate name is disambiguated IN THE UI too, not just in what gets
  // saved: two identical rows would otherwise be an unexplained coin flip.
  if (i === 0) return list[0];
  return list.indexOf(list[i]) === i && list.lastIndexOf(list[i]) === i
    ? list[i]
    : list[i] + "  [" + (i - 1) + "]";
}

function fillSelect(sel, list) {
  sel.textContent = "";
  list.forEach((nameStr, i) => {
    const o = document.createElement("option");
    // value is the DEVICE index (row 0 -> -1, the system-default sentinel).
    o.value = String(i - 1);
    o.textContent = optionLabel(list, i);
    sel.appendChild(o);
  });
  sel.disabled = false;
}

// Captured before anything can overwrite it, so a failure that is later
// RECOVERED (the app answers a retry, a backend comes back) puts the real
// explanatory text back instead of leaving a stale error under a working list.
const OUTPUT_HINT_HTML = $("#outputDeviceHint") ? $("#outputDeviceHint").innerHTML : "";

function markDevicesUnavailable(why) {
  for (const k in DEVICE_SELECTS) {
    const sel = $("#" + DEVICE_SELECTS[k].el);
    if (!sel) continue;
    sel.textContent = "";
    const o = document.createElement("option");
    o.value = "-1";
    o.textContent = "(system default)";
    sel.appendChild(o);
    // Disabled rather than empty: an empty dropdown reads as "this machine has
    // no audio devices", which is a hardware diagnosis the page has not earned.
    sel.disabled = true;
  }
  const hint = $("#outputDeviceHint");
  if (hint) hint.innerHTML = "<b>Could not read the device list</b> (" + why +
    "). The assistant will use the system default. The console prints the same " +
    "list at launch.";
}

// Asks the app for the list. The reply arrives asynchronously as an
// "audio.devices" message (see onMessage) rather than as the return value here
// -- the app↔page transport is one-way postMessage, not request/response.
//
// NOT a fetch, though it started as one: a same-origin fetch to the virtual
// asset host is served straight off disk and never reaches the C++ side, so
// there is no way for the app to answer it. The full reasoning is in
// assistant_window.cpp above audio_devices_payload().
//
// The watchdog exists because a message that is never answered leaves no trace:
// the dropdown would sit empty forever with nothing on screen saying why. An app
// that is alive answers this in microseconds, so 4 s only ever fires when
// something is genuinely wrong.
let deviceRequestTimer = null;
function loadAudioDevices() {
  if (!host) {                       // opened outside the app shell
    markDevicesUnavailable("not running inside the assistant");
    return;
  }
  clearTimeout(deviceRequestTimer);
  deviceRequestTimer = setTimeout(() => {
    if (!deviceLists) markDevicesUnavailable("the app did not answer");
  }, 4000);
  send({ type: "audio.devices.request" });
}

// Applies a payload of the shape { outputs: [...], inputs: [...] }.
function applyAudioDevices(j) {
  clearTimeout(deviceRequestTimer);
  if (!j || !Array.isArray(j.outputs) || !Array.isArray(j.inputs)) {
    deviceLists = null;
    markDevicesUnavailable("the device list was malformed");
    return;
  }
  deviceLists = { output: j.outputs, input: j.inputs };
  fillSelect($("#" + DEVICE_SELECTS.output.el), j.outputs);
  fillSelect($("#" + DEVICE_SELECTS.input.el), j.inputs);
  const hint = $("#outputDeviceHint");
  if (hint) hint.innerHTML = OUTPUT_HINT_HTML;
  // Settings usually land first, but not always -- re-select from whatever the
  // app last told us, so either arrival order ends at the same selection.
  if (seeded) writeDevices(current);
  // The popover mirrors these options, so it has to be refilled whenever they
  // change -- otherwise it keeps showing the list from the previous open.
  syncPopoverFromForm();
}

function readDevices(out) {
  for (const k in DEVICE_SELECTS) {
    const spec = DEVICE_SELECTS[k];
    const sel = $("#" + spec.el);
    // Untouched when the list never loaded: writing "" + -1 from a disabled
    // dropdown would silently WIPE a working configured device on the next save.
    if (!sel || sel.disabled || !deviceLists) continue;
    const list = deviceLists[k];
    const idx = parseInt(sel.value, 10);
    if (Number.isNaN(idx) || idx < 0) {
      out[spec.name] = "";
      out[spec.index] = -1;
      continue;
    }
    const nameStr = list[idx + 1];
    const dup = list.indexOf(nameStr) !== list.lastIndexOf(nameStr);
    out[spec.name] = nameStr;
    out[spec.index] = dup ? idx : -1;
  }
  return out;
}

function writeDevices(s) {
  if (!deviceLists) return;
  for (const k in DEVICE_SELECTS) {
    const spec = DEVICE_SELECTS[k];
    const sel = $("#" + spec.el);
    if (!sel) continue;
    const list = deviceLists[k];
    const savedIdx = typeof s[spec.index] === "number" ? s[spec.index] : -1;
    const savedName = s[spec.name] || "";
    // Same precedence the C++ resolver applies (index beats name), so what the
    // dropdown shows is what the app will actually open -- including for a
    // settings.json edited by hand into a state the UI cannot produce.
    let pick = -1;
    if (savedIdx >= 0 && savedIdx < list.length - 1) {
      pick = savedIdx;
    } else if (savedName) {
      const at = list.indexOf(savedName);
      if (at > 0) pick = at - 1;
    }
    // A configured device that is not in the list (unplugged, or renamed) leaves
    // the dropdown on "(system default)" -- which is what the app will fall back
    // to anyway, so the UI agrees with the behaviour rather than showing a
    // selection that no longer exists.
    sel.value = String(pick);
  }
}

/* ------------------------- live audio application -------------------------
   The audio controls do NOT wait for Save. Every other setting in this panel is
   a considered change you might abandon with Cancel; an output device and a
   volume are things you adjust BY HEARING THE RESULT, and a slider you have to
   confirm cannot be adjusted that way at all.

   They are safe to apply immediately because of what they cost on the C++ side:
   the gains are an atomic each, and a device change closes and reopens two
   ma_device handles. Nothing here reloads a model, and the app does not restart
   -- which is exactly why these fields were moved off the restart tier. */

/* NEVER settings.save. That route carries the WHOLE form, and the app has to ask
   requires_restart() of it -- so a blank model path in a tab the user never
   opened turns a volume drag into an engine restart. That is not hypothetical;
   it is what this replaced.

   audio.hot_update carries ONLY the fields named below. The C++ side cannot
   compute a restart from it because it is not given anything a restart could
   depend on. */
function sendAudioHot(patch) {
  send(Object.assign({ type: "audio.hot_update" }, patch));
}

// Devices reopen a WASAPI handle, so they go immediately and unthrottled -- a
// select fires once, on commit.
function sendDeviceChange(which) {
  const spec = DEVICE_SELECTS[which];
  const sel = $("#" + spec.el);
  if (!sel || sel.disabled || !deviceLists) return;
  const list = deviceLists[which];
  const idx = parseInt(sel.value, 10);
  const patch = {};
  if (Number.isNaN(idx) || idx < 0) {
    patch[spec.name] = "";
    patch[spec.index] = -1;
  } else {
    const nameStr = list[idx + 1];
    const dup = list.indexOf(nameStr) !== list.lastIndexOf(nameStr);
    patch[spec.name] = nameStr;
    patch[spec.index] = dup ? idx : -1;
  }
  sendAudioHot(patch);
}

// Gains are atomic stores on the C++ side, so they can go at drag rate -- but
// each is still a JSON round trip, so 60 ms keeps the wire quiet while staying
// well below the point where the slider stops feeling attached to the sound.
//
// COALESCED, NOT DROPPED: the pending patch accumulates fields, so a drag that
// also flips the mute (the rail's unmute-on-drag) delivers both in one message
// and the app can never apply half of the pair.
let gainTimer = null;
let pendingGains = {};
function sendGainPatch(patch) {
  Object.assign(pendingGains, patch);
  clearTimeout(gainTimer);
  gainTimer = setTimeout(() => {
    sendAudioHot(pendingGains);
    pendingGains = {};
  }, 60);
}
function sendGainChange(field, value) {
  sendGainPatch({ [field]: value });
}

function wireAudioLiveControls() {
  for (const k in DEVICE_SELECTS) {
    const sel = $("#" + DEVICE_SELECTS[k].el);
    if (sel) sel.addEventListener("change", () => sendDeviceChange(k));
  }
  const gains = { s_tts_volume: "tts_volume", s_mic_gain: "mic_gain" };
  for (const id in gains) {
    const e = $("#" + id);
    if (!e) continue;
    e.addEventListener("input", () => {
      // Mirror into the readout immediately, before the round trip: the number
      // beside the slider must track the thumb, not the app's acknowledgement.
      refreshReadouts();
      // ...and into the dock rail, which shows the same value on the main
      // screen. Three controls, one number: whichever one moves, the other two
      // follow, because a user who drags the modal slider and then glances at
      // the dock must not see two different volumes.
      if (id === "s_tts_volume") syncDockVolumeFromForm();
      sendGainChange(gains[id], parseFloat(e.value) || 0);
    });
  }
  wireCheckSound("#checkSoundBtn", "#checkSoundMsg");
}

function wireCheckSound(btnSel, msgSel) {
  const btn = $(btnSel);
  const msg = $(msgSel);
  if (!btn) return;
  btn.addEventListener("click", () => {
    // Flush any pending gain BEFORE the tone, so the button tests the value on
    // screen rather than the one from before the drag. Still no settings.save.
    clearTimeout(gainTimer);
    if (Object.keys(pendingGains).length) {
      sendAudioHot(pendingGains);
      pendingGains = {};
    }
    send({ type: "audio.test_tone" });
    btn.disabled = true;
    if (msg) msg.textContent = "Playing…";
    setTimeout(() => {
      btn.disabled = false;
      if (msg) msg.textContent = "Heard nothing? Try another output device.";
    }, 700);
  });
}

/* ------------------------- the speaker + volume rail ------------------------
   MUTE IS NOT VOLUME 0, and the whole design of this control follows from that.
   Sending volume 0 on mute destroys the value being muted, so unmute has to
   guess -- and the only available guess promotes everyone who listens at 30% to
   full scale on their first tap. So the icon sends `tts_muted` and the slider
   sends `tts_volume`, they are independent fields of the same hot update, and
   the C++ side folds them (see AudioHotUpdate::tts_muted).

   The slider therefore keeps SHOWING the stored volume while muted, greyed. It
   is what you will get back, and hiding it would make unmute feel like a gamble.

   NOT PERSISTED. The app applies the mute and pointedly does not save it: an
   assistant that starts up silent because of a tap three days ago reads as
   broken, so the page does not seed this from settings either. */
let speakerOn = true;

function paintSpeaker() {
  const btn = $("#speakerBtn"), dock = $("#volumeDock");
  if (!btn) return;
  btn.dataset.on = String(speakerOn);
  btn.setAttribute("aria-pressed", String(!speakerOn));
  btn.title = btn.ariaLabel =
    speakerOn ? "Mute the assistant" : "Unmute the assistant";
  if (dock) dock.classList.toggle("muted", !speakerOn);
}

function wireSpeakerDock() {
  const btn = $("#speakerBtn"), dock = $("#volumeDock"), rail = $("#dock_tts_volume");
  if (!btn || !dock) return;

  btn.addEventListener("click", e => {
    // The rail opens on hover, but a click has to work on a touchpad and on
    // touch, where there is no hover at all -- .open latches what :hover would
    // otherwise be the only way to reach.
    e.stopPropagation();
    speakerOn = !speakerOn;
    paintSpeaker();
    dock.classList.add("open");
    sendAudioHot({ tts_muted: !speakerOn });
  });

  if (rail) {
    rail.addEventListener("input", () => {
      // Mirror into the OTHER view of the same number before the round trip: the
      // readout must track the thumb, not the app's acknowledgement.
      const f = $("#s_tts_volume");
      if (f) f.value = rail.value;
      refreshDockVolume();
      refreshReadouts();
      // Moving the slider while muted is an unmute: the user is reaching for a
      // volume, and leaving them dragging a control that produces no sound is
      // the kind of dead end that gets reported as "the slider does nothing".
      // Sent as ONE patch with the volume, so the app cannot briefly apply a new
      // volume to a still-muted speaker (or the reverse).
      const patch = { tts_volume: parseFloat(rail.value) || 0 };
      if (!speakerOn) {
        speakerOn = true;
        paintSpeaker();
        patch.tts_muted = false;
      }
      sendGainPatch(patch);
    });
  }

  // Click-away only unlatches the click-opened state; hover still governs the
  // rest, so this cannot fight the CSS.
  document.addEventListener("click", () => dock.classList.remove("open"));
  dock.addEventListener("click", e => e.stopPropagation());
  paintSpeaker();
  // Reconcile the readout with the thumb BEFORE the app's first settings push.
  // A range input with no value attribute lands at the midpoint, so the markup's
  // placeholder "100%" would sit next to a thumb at 50% for however long the
  // engine takes to come up -- which on a cold 8 GB load is seconds of the dock
  // stating a volume that is not the one in force.
  refreshDockVolume();
}

// The dock rail follows whatever the app last told us the volume is.
function refreshDockVolume() {
  const rail = $("#dock_tts_volume"), out = $("#dockVolValue");
  if (!rail) return;
  if (out) out.textContent = Math.round(Number(rail.value) * 100) + "%";
}

function syncDockVolumeFromForm() {
  const f = $("#s_tts_volume"), rail = $("#dock_tts_volume");
  if (f && rail) rail.value = f.value;
  refreshDockVolume();
}

/* --------------------------- the gear popover ------------------------------
   Quick settings: the two device dropdowns, the microphone gain with its level
   meter, and the tone button -- reachable without opening the settings modal,
   because changing output device is something you do while listening, and
   burying it three clicks deep in a modal that also holds checkpoint paths is
   what made it feel like a dangerous operation.

   SPEECH VOLUME IS NOT IN HERE any more: it lives on the dock rail next to the
   speaker icon, where it can be adjusted while the assistant is talking. A
   volume inside a popover is a volume you cannot hear yourself setting, because
   the popover is what your cursor is busy holding open.

   It mirrors rather than duplicates: the popover's controls write through the
   same sendDeviceChange/sendGainChange, and both views are re-synced from the
   app's own state, so they cannot disagree about what is selected. */
function syncPopoverFromForm() {
  for (const k in DEVICE_SELECTS) {
    const src = $("#" + DEVICE_SELECTS[k].el);
    const dst = $("#pop_" + k + "_device");
    if (!src || !dst) continue;
    dst.textContent = "";
    for (const o of src.options) {
      const c = document.createElement("option");
      c.value = o.value;
      c.textContent = o.textContent;
      dst.appendChild(c);
    }
    dst.value = src.value;
    dst.disabled = src.disabled;
  }
  const g = $("#s_mic_gain"), pg = $("#pop_mic_gain");
  if (g && pg) pg.value = g.value;
  refreshPopoverReadouts();
  syncDockVolumeFromForm();
}

function refreshPopoverReadouts() {
  const pg = $("#pop_mic_gain"), pgo = $("#popGainValue");
  if (pg && pgo) pgo.textContent = Math.round(Number(pg.value) * 100) + "%";
}

function wireAudioPopover() {
  const btn = $("#audioDockBtn");
  const pop = $("#audioPopover");
  if (!btn || !pop) return;

  const close = () => {
    pop.classList.add("hidden");
    btn.classList.remove("active");
    btn.setAttribute("aria-expanded", "false");
  };
  btn.addEventListener("click", e => {
    e.stopPropagation();
    const opening = pop.classList.contains("hidden");
    if (opening) {
      // Re-enumerate on open, exactly like the settings panel: a headset plugged
      // in since launch is the single most common reason to open this.
      loadAudioDevices();
      syncPopoverFromForm();
      pop.classList.remove("hidden");
      btn.classList.add("active");
      btn.setAttribute("aria-expanded", "true");
    } else {
      close();
    }
  });
  // Click-away and Escape, the two things every popover is expected to honour.
  pop.addEventListener("click", e => e.stopPropagation());
  document.addEventListener("click", () => { if (!pop.classList.contains("hidden")) close(); });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !pop.classList.contains("hidden")) close();
  });

  for (const k in DEVICE_SELECTS) {
    const dst = $("#pop_" + k + "_device");
    if (!dst) continue;
    dst.addEventListener("change", () => {
      // Write through the MODAL's control, then reuse its sender -- so the two
      // views cannot drift, and there is exactly one place that knows how a
      // selection becomes a name plus an index.
      const src = $("#" + DEVICE_SELECTS[k].el);
      if (src) src.value = dst.value;
      sendDeviceChange(k);
    });
  }
  const pg = $("#pop_mic_gain");
  if (pg) {
    pg.addEventListener("input", () => {
      const f = $("#s_mic_gain");
      if (f) f.value = pg.value;
      refreshPopoverReadouts();
      refreshReadouts();
      sendGainChange("mic_gain", parseFloat(pg.value) || 0);
    });
  }
  wireCheckSound("#popCheckSoundBtn", "#popCheckSoundMsg");
}

// Live microphone amplitude, pushed by the app several times a second.
function setMicLevel(level) {
  const v = Math.max(0, Math.min(1, Number(level) || 0));
  // sqrt, not the raw amplitude: speech peaks sit low in a linear scale and a
  // linear bar barely twitches on normal talking, which reads as a broken
  // meter. This is the same reason level meters are not linear in volts.
  const w = (Math.sqrt(v) * 100).toFixed(1) + "%";
  for (const id of ["#micMeterFill", "#popMicMeterFill"]) {
    const fill = $(id);
    if (fill) fill.style.width = w;
  }
}

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
  return readDevices(out);
}

function writeForm(s) {
  for (const k in FIELDS) {
    const e = el(k);
    if (!e || s[k] === undefined) continue;
    if (FIELDS[k] === "check") e.checked = !!s[k];
    else e.value = s[k];
  }
  writeDevices(s);
  refreshReadouts();
  // The dock rail is a third view onto tts_volume and is NOT inside the form, so
  // it does not get filled by the loop above. Without this it would sit at the
  // markup default until the user touched it -- showing 100% next to a speaker
  // playing at 40%.
  syncDockVolumeFromForm();
  refreshPromptCount();
  // Only a REAL payload counts as seeded. Opening the modal before the app's
  // first push writes an empty form, and marking that as seeded would lock the
  // blank state in for the rest of the session.
  if (Object.keys(s).length) seeded = true;
  refreshRestartBanner();
}

function refreshReadouts() {
  for (const k in READOUTS) {
    const e = el(k), out = $(READOUTS[k][0]);
    if (e && out) out.textContent = READOUTS[k][1](e.value);
  }
}

// A restart-tier field is one the APP named in `restart_fields`. That list comes
// straight from settings_store.hpp's tier table, so the banner cannot claim a
// restart the app does not perform, or stay quiet about one it does.
function restartNeeded() {
  const form = readForm();
  return restartKeys.some(k => k in FIELDS && String(form[k]) !== String(current[k]));
}
function refreshRestartBanner() {
  const need = restartNeeded();
  restartBanner.classList.toggle("hidden", !need);
  saveBtn.textContent = need ? "Save & Restart" : "Save";
}

overlay.addEventListener("input", () => { refreshReadouts(); refreshRestartBanner(); });
overlay.addEventListener("change", () => { refreshReadouts(); refreshRestartBanner(); });

/* ---- API key reveal -------------------------------------------------------
   A masked field you cannot read back is how a key with a stray character in it
   stays a mystery: the only symptom is a 401, which looks identical to a wrong
   key. This flips the input type only -- the value is never copied anywhere,
   and the field returns to masked when the modal closes. */
const revealKey = $("#revealKey");
const keyField = $("#s_remote_api_key");
function maskKey() {
  if (!keyField || !revealKey) return;
  keyField.type = "password";
  revealKey.textContent = "Show";
  revealKey.setAttribute("aria-pressed", "false");
}
if (revealKey && keyField) {
  revealKey.addEventListener("click", () => {
    const show = keyField.type === "password";
    keyField.type = show ? "text" : "password";
    revealKey.textContent = show ? "Hide" : "Show";
    revealKey.setAttribute("aria-pressed", String(show));
  });
}

/* ---- tabs ----------------------------------------------------------------
   Plain show/hide rather than separate documents: the form is ONE payload, and
   Save must send every field whether or not its tab was ever opened. */
const tabs = Array.from(document.querySelectorAll(".tab"));
const panels = Array.from(document.querySelectorAll(".panel"));

function selectTab(name) {
  tabs.forEach(t => {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", String(on));
  });
  panels.forEach(p => p.classList.toggle("active", p.dataset.panel === name));
}
tabs.forEach(t => t.addEventListener("click", () => selectTab(t.dataset.tab)));

/* ---- hotkey capture -------------------------------------------------------
   The field records a chord instead of accepting typed text: "Ctrl+Alt+Space"
   is a format people mistype, and a hotkey that silently failed to parse would
   just look like a hotkey that does not work. Modifier-only presses are ignored
   so the field does not settle on "Ctrl" while the user is still reaching for
   the second key. The strings produced here are exactly what parse_hotkey()
   accepts in assistant_window.cpp. */
const KEY_LABEL = {
  " ": "Space", Spacebar: "Space", Escape: "Escape", Enter: "Enter", Tab: "Tab",
  ArrowLeft: "Left", ArrowRight: "Right", ArrowUp: "Up", ArrowDown: "Down",
  PageUp: "PageUp", PageDown: "PageDown", Home: "Home", End: "End",
  Insert: "Insert", Delete: "Delete", Pause: "Pause"
};
const MODIFIER_KEYS = ["Control", "Alt", "Shift", "Meta", "OS", "AltGraph", "CapsLock"];

function keyName(e) {
  if (KEY_LABEL[e.key]) return KEY_LABEL[e.key];
  if (/^F([1-9]|1\d|2[0-4])$/.test(e.key)) return e.key;
  if (e.key.length === 1) {
    const c = e.key.toUpperCase();
    return /[A-Z0-9]/.test(c) ? c : "";     // punctuation varies by layout: refuse it
  }
  return "";
}

document.querySelectorAll("[data-hotkey]").forEach(input => {
  input.addEventListener("focus", () => {
    input.classList.add("recording");
    input.dataset.prev = input.value;
    input.value = "Press a combination…";
  });
  input.addEventListener("blur", () => {
    input.classList.remove("recording");
    // Nothing was captured -- put back what was there rather than leaving the
    // prompt text in a field that gets saved verbatim.
    if (input.value === "Press a combination…") input.value = input.dataset.prev || "";
    refreshRestartBanner();
  });
  input.addEventListener("keydown", e => {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === "Backspace" || e.key === "Delete") {   // clear the binding
      input.value = "";
      input.dataset.prev = "";
      refreshRestartBanner();
      return;
    }
    if (MODIFIER_KEYS.includes(e.key)) return;           // still reaching
    const base = keyName(e);
    if (!base) return;                                   // not a key we can register
    const parts = [];
    if (e.ctrlKey) parts.push("Ctrl");
    if (e.altKey) parts.push("Alt");
    if (e.shiftKey) parts.push("Shift");
    if (e.metaKey) parts.push("Win");
    parts.push(base);
    input.value = parts.join("+");
    input.dataset.prev = input.value;
    refreshRestartBanner();
  });
});

/* ---- system prompts tab ---------------------------------------------------
   Only the PERSONA carries a token count and a cache state, and that asymmetry
   is the point: it is the prompt whose length is prefilled once and paid for on
   every launch. The audio task instruction is re-prefilled every turn, so its
   size is noise and reporting it would imply a cost that is not there. */
const promptBox = $("#s_system_prompt");

function refreshPromptCount() {
  const n = promptBox.value.length;
  // A rough token estimate, labelled as rough. The real count comes back from
  // the engine after a precompute (prefixState below) -- this is only here so
  // the box is not silent while it is being edited.
  $("#promptCount").textContent =
    n + " characters · ~" + Math.ceil(n / 4) + " tokens";
}
promptBox.addEventListener("input", refreshPromptCount);

$("#promptReset").addEventListener("click", () => {
  promptBox.value = DEFAULT_SYSTEM_PROMPT;
  el("audio_task_prompt").value = DEFAULT_AUDIO_TASK_PROMPT;
  el("speech_language").value = "";      // Auto: let the model identify it
  el("speech_task").value = "transcribe";
  refreshPromptCount();
  refreshRestartBanner();
});

// Apply & precompute IS the Save button, with a progress label. Delegating
// rather than sending its own message is deliberate: the form is ONE payload, so
// this click persists every tab's fields either way, and giving it a second set
// of semantics would mean a restart-tier edit could be saved here and silently
// never applied -- the banner would clear itself on the echo while the old
// engine kept running.
$("#promptApply").addEventListener("click", () => {
  $("#prefixState").textContent = "Precomputing…";
  $("#prefixState").className = "prefix-state busy";
  saveBtn.click();
});

function openSettings() {
  writeForm(current);
  saveMsg.textContent = "";
  maskKey();   // a key left revealed from last time must not survive a reopen
  overlay.classList.remove("hidden");
  // RE-ENUMERATED ON EVERY OPEN, not just at startup. Plugging in a headset is
  // the single most common reason to open this panel at all, so a list cached
  // from launch would be stale exactly when it is being looked at. It resolves
  // asynchronously and re-selects the saved device when it lands (see
  // loadAudioDevices), so the panel is usable before the fetch returns.
  loadAudioDevices();
}
function closeSettings() {
  maskKey();
  overlay.classList.add("hidden");
}

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
        // The gate refused this utterance, so no answer stream is coming and no
        // `remote.final` will ever arrive to clear the button.
        setGenerating(false);
      }
      setPhase(msg.phase);
      break;
    }

    case "user.text":
      freeze(liveUser);
      liveUser = null;
      addBubble("user", msg.text, false);
      // The turn has begun: the message is in the gate and the dispatcher is
      // about to pick it up. Stopping is meaningful from this instant, which is
      // before there is any answer to stop.
      setGenerating(true);
      if (msg.phase) setPhase(msg.phase);
      break;

    case "remote.start":
      freeze(liveAssistant);
      liveAssistant = addBubble("assistant", "", true);
      showTyping(liveAssistant);
      // Also raised here, not only on `user.text`: a SPOKEN turn never sends
      // one, and it needs the Stop button just as much.
      setGenerating(true);
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
        // STOPPED IS NOT FAILED. The user cut this answer off themselves and a
        // red note explaining what went wrong would be describing their own
        // click back at them. The words already on screen are kept -- they were
        // generated, and in the cloud case they were paid for.
        if (msg.cancelled) addNote(liveAssistant, "Stopped", "muted");
        else if (!msg.ok) addNote(liveAssistant, msg.detail || "The reply failed.", "err");
        liveAssistant = null;
      }
      // The turn is over: a question that has not arrived by now is not coming,
      // and letting the anchor outlive its reply is what would let it capture
      // the NEXT turn's user bubble.
      orphanAssistant = null;
      // THE ONE PLACE the composer returns to Send. Reached on every ending --
      // success, failure, refusal, cancellation -- which is why nothing else
      // needs to guess at when a turn is over.
      setGenerating(false);
      break;

    case "session.list":
      sessions = Array.isArray(msg.sessions) ? msg.sessions : [];
      if (typeof msg.active === "string") activeSession = msg.active;
      drawerEphemeral.classList.toggle("hidden", !msg.ephemeral);
      renderSessions();
      break;

    case "session.restore":
      // The app has switched conversations (or just finished restoring one at
      // startup). Repaint from what it sent -- the page never reconstructs a
      // transcript from its own memory, so what is on screen is always what the
      // model was actually given.
      if (typeof msg.id === "string") activeSession = msg.id;
      restoreTranscript(msg.turns);
      // Switching conversations cancels the turn in flight (the app does this
      // before it repaints -- see switch_to_session), so the composer must not
      // be left offering to stop a turn that belongs to a conversation no longer
      // on screen.
      setGenerating(false);
      if (msg.phase) setPhase(msg.phase);
      renderSessions();   // the active row moved
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

    case "mic":
      // The app owns mic state: the talk hotkey flips it while this window may
      // not even be visible. Painted only -- echoing it back would ping-pong.
      paintMic(!!msg.on);
      break;

    case "settings":
      current = msg.payload || {};
      if (Array.isArray(msg.restart_fields)) restartKeys = msg.restart_fields;
      // Refill only when the modal is closed -- an update that arrives while the
      // user is typing must not wipe their edits. The `seeded` escape hatch
      // covers the race where the modal was opened before the app's first push:
      // an empty form the user could Save would silently blank every setting.
      if (overlay.classList.contains("hidden") || !seeded) writeForm(current);
      else refreshRestartBanner();
      break;

    case "audio.level":
      setMicLevel(msg.level);
      break;

    case "audio.devices":
      // Unconditional, unlike "settings" above: repopulating the dropdowns
      // cannot destroy typed input the way refilling the whole form can, and a
      // list arriving while the modal is open is exactly the headset-just-
      // plugged-in case this is for.
      applyAudioDevices(msg);
      break;

    case "settings.saved":
      // A prompt rebuild is engine work and answers separately; saying "applied
      // now" while the GPU is still prefilling would be a lie, and closing the
      // modal would hide the answer when it arrives.
      if (msg.prompt_rebuilding) {
        saveMsg.textContent = "Saved — rebuilding the prompt cache…";
      } else {
        // No rebuild means no system_prompt.applied is coming, so a
        // "Precomputing…" left by the Apply button would hang there forever.
        // Editing only the audio task instruction lands here, and it genuinely
        // needed no cache work -- say so rather than showing nothing.
        const state = $("#prefixState");
        if (state.textContent === "Precomputing…") {
          state.textContent = "Applied — no cache rebuild needed";
          state.className = "prefix-state ok";
        }
        saveMsg.textContent = msg.live_only ? "Saved — applied now."
                                            : "Saved — restarting…";
        if (msg.live_only) setTimeout(closeSettings, 550);
      }
      break;

    case "system_prompt.applied": {
      const state = $("#prefixState");
      if (msg.ok) {
        state.textContent = "Cached — " + msg.tokens + " tokens frozen";
        state.className = "prefix-state ok";
        $("#d_prefix").textContent = msg.tokens + " tokens cached";
        saveMsg.textContent = "Saved — applied now.";
        setTimeout(closeSettings, 700);
      } else {
        state.textContent = msg.detail || "Could not apply the prompt.";
        state.className = "prefix-state err";
        saveMsg.textContent = "Saved, but the prompt was not applied.";
      }
      break;
    }

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
// At load as well as on every settings open: the dropdowns must already hold
// real options the first time the panel is shown, or the initial writeForm()
// would have nothing to select the saved device from.
loadAudioDevices();
wireAudioLiveControls();
wireSpeakerDock();
wireAudioPopover();
input.focus();
