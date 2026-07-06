"use strict";
/* Blackwell Overlay — Settings UI logic.
   Pure front-end: it receives the Config as JSON from the C++ host on load and
   posts the edited Config back on Save. All host IPC goes through
   window.chrome.webview (postMessage / 'message' event). Every handler is
   wrapped so a single bad message can never blank the UI. */

const HK = { SHIFT: 1, CONTROL: 2, ALT: 4 };
const MOD_VK = [16, 17, 18, 91, 92]; // Shift / Ctrl / Alt / Win (left+right)
const CLEAR_VK = [8, 46];            // Backspace / Delete clear a hotkey field
const MAX_HOTKEY_PAIRS = 9;          // Alt+1 .. Alt+9

const bridge = (window.chrome && window.chrome.webview) ? window.chrome.webview : null;
const $ = (id) => document.getElementById(id);

/* [code, English name]. The name goes verbatim into the model's system prompt
   (Config.target); the code only shapes the "RU -> EN" label. */
const LANGUAGES = [
  ['EN', 'English'],    ['RU', 'Russian'],   ['ZH', 'Chinese'],    ['ES', 'Spanish'],
  ['FR', 'French'],     ['DE', 'German'],    ['JA', 'Japanese'],   ['KO', 'Korean'],
  ['PT', 'Portuguese'], ['IT', 'Italian'],   ['NL', 'Dutch'],      ['PL', 'Polish'],
  ['TR', 'Turkish'],    ['AR', 'Arabic'],    ['HE', 'Hebrew'],     ['HI', 'Hindi'],
  ['BN', 'Bengali'],    ['ID', 'Indonesian'],['VI', 'Vietnamese'], ['TH', 'Thai'],
  ['UK', 'Ukrainian'],  ['CS', 'Czech'],     ['SV', 'Swedish'],    ['DA', 'Danish'],
  ['FI', 'Finnish'],    ['NO', 'Norwegian'], ['EL', 'Greek'],      ['HU', 'Hungarian'],
  ['RO', 'Romanian'],   ['BG', 'Bulgarian'], ['FA', 'Persian'],    ['MS', 'Malay']
];
const AUTO_SOURCE = 'Auto'; // source is model-detected; "Auto" is the honest default

function codeOfTarget(name) {
  const hit = LANGUAGES.find(([, n]) => n.toLowerCase() === String(name).toLowerCase());
  return hit ? hit[0] : String(name).slice(0, 2).toUpperCase();
}

function pairLabel(sourceCode, targetName) {
  return sourceCode + ' -> ' + codeOfTarget(targetName);
}

/* Recover the source code from a stored label ("RU -> EN" -> "RU"); anything
   unparseable (hand-edited configs) falls back to Auto. */
function sourceOfLabel(label) {
  const m = /^\s*([A-Za-z]{2,5})\s*->/.exec(label || '');
  if (!m) return AUTO_SOURCE;
  const code = m[1].toUpperCase();
  if (code === AUTO_SOURCE.toUpperCase()) return AUTO_SOURCE;
  return LANGUAGES.some(([c]) => c === code) ? code : AUTO_SOURCE;
}

// ---- shared state -----------------------------------------------------------
const state = {
  activation: { modifiers: 0, vk: 0 },
  commit:     { modifiers: 0, vk: 0 },
  cycle:      { modifiers: 0, vk: 0 },
  pairs:      [],      // [{ label, target, source }] (source is UI-only)
  activeLanguage: 0,
  savedTargets: [],    // targets the C++ side currently serves (pre-cache gate)
  precacheState: {}    // row index -> 'compiling' | 'done' | 'error' | 'unsaved'
};

// ---- hotkey helpers ---------------------------------------------------------
function modsFromEvent(e) {
  return (e.ctrlKey ? HK.CONTROL : 0)
       | (e.shiftKey ? HK.SHIFT : 0)
       | (e.altKey ? HK.ALT : 0);
}

function vkName(vk) {
  const map = {
    8: 'Backspace', 9: 'Tab', 13: 'Enter', 27: 'Esc', 32: 'Space',
    33: 'PageUp', 34: 'PageDown', 35: 'End', 36: 'Home',
    37: 'Left', 38: 'Up', 39: 'Right', 40: 'Down', 45: 'Insert', 46: 'Delete',
    186: ';', 187: '=', 188: ',', 189: '-', 190: '.', 191: '/', 192: '`',
    219: '[', 220: '\\', 221: ']', 222: "'"
  };
  if (map[vk]) return map[vk];
  if (vk >= 65 && vk <= 90) return String.fromCharCode(vk);   // A-Z
  if (vk >= 48 && vk <= 57) return String.fromCharCode(vk);   // 0-9
  if (vk >= 96 && vk <= 105) return 'Num' + (vk - 96);        // numpad 0-9
  if (vk >= 112 && vk <= 123) return 'F' + (vk - 111);        // F1-F12
  return 'Key' + vk;
}

function hotkeyLabel(sc) {
  if (!sc || !sc.vk) return '';
  const parts = [];
  if (sc.modifiers & HK.CONTROL) parts.push('Ctrl');
  if (sc.modifiers & HK.SHIFT)   parts.push('Shift');
  if (sc.modifiers & HK.ALT)     parts.push('Alt');
  parts.push(vkName(sc.vk));
  return parts.join(' + ');
}

function bindHotkey(id) {
  const el = $(id);
  el.addEventListener('keydown', function (e) {
    e.preventDefault();
    e.stopPropagation();
    if (CLEAR_VK.includes(e.keyCode)) {   // Backspace / Delete unbinds
      state[id] = { modifiers: 0, vk: 0 };
      el.value = '';
      return;
    }
    if (MOD_VK.includes(e.keyCode)) return; // wait for a real (non-modifier) key
    state[id] = { modifiers: modsFromEvent(e), vk: e.keyCode };
    el.value = hotkeyLabel(state[id]);
  });
}

// ---- language pairs ---------------------------------------------------------
function makeSelect(options, value) {
  const sel = document.createElement('select');
  options.forEach(([val, text]) => {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = text;
    sel.appendChild(opt);
  });
  if (value && ![...sel.options].some(o => o.value === value)) {
    const opt = document.createElement('option'); // keep an exotic stored value alive
    opt.value = value;
    opt.textContent = value;
    sel.appendChild(opt);
  }
  sel.value = value;
  return sel;
}

/* Pre-cache button per row. State machine: idle -> compiling -> done, with
   transient 'error' / 'unsaved' ("Save first") states that auto-revert. */
function makePrecacheButton(index, target) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'secondary btn-precache';
  const st = state.precacheState[index];
  if (st === 'compiling') {
    btn.textContent = 'Compiling…';
    btn.disabled = true;
  } else if (st === 'done') {
    btn.textContent = 'Cached ✓';
    btn.disabled = true;
  } else if (st === 'error') {
    btn.textContent = 'Failed';
    btn.disabled = true;
  } else if (st === 'unsaved') {
    btn.textContent = 'Save first';
    btn.disabled = true;
  } else {
    btn.textContent = 'Pre-cache';
    btn.addEventListener('click', function () {
      // The C++ side compiles by target name against the SAVED pair set, so an
      // unsaved row can only fail -- say so instead of round-tripping.
      if (!state.savedTargets.includes(target)) {
        flashPrecacheState(index, 'unsaved');
        return;
      }
      state.precacheState[index] = 'compiling';
      renderPairs();
      try {
        if (bridge) bridge.postMessage({ type: 'precache', target: target, id: index });
      } catch (err) {
        console.error('settings: failed to post precache request', err);
        flashPrecacheState(index, 'error');
      }
    });
  }
  return btn;
}

function flashPrecacheState(index, transient) {
  state.precacheState[index] = transient;
  renderPairs();
  setTimeout(function () {
    if (state.precacheState[index] === transient) {
      delete state.precacheState[index];
      renderPairs();
    }
  }, 1800);
}

function renderPairs() {
  const list = $('pairsList');
  list.innerHTML = '';

  if (state.pairs.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'pairs-empty';
    empty.textContent = 'No language pairs — add at least one.';
    list.appendChild(empty);
    return;
  }
  if (state.activeLanguage >= state.pairs.length) state.activeLanguage = 0;

  state.pairs.forEach(function (pair, index) {
    const row = document.createElement('div');
    row.className = 'pair-row';

    const sourceSel = makeSelect(
      [[AUTO_SOURCE, AUTO_SOURCE]].concat(LANGUAGES.map(([c]) => [c, c])),
      pair.source || AUTO_SOURCE);
    const targetSel = makeSelect(
      LANGUAGES.map(([c, n]) => [n, c + ' — ' + n]),
      pair.target || 'English');

    const syncLabel = function () {
      pair.source = sourceSel.value;
      pair.target = targetSel.value;
      pair.label = pairLabel(pair.source, pair.target);
    };
    sourceSel.addEventListener('change', syncLabel);
    targetSel.addEventListener('change', function () {
      syncLabel();
      delete state.precacheState[index]; // a different target = a different branch
      renderPairs();
    });

    const hotkey = document.createElement('div');
    hotkey.className = 'pair-hotkey';
    hotkey.textContent = index < MAX_HOTKEY_PAIRS ? 'Alt+' + (index + 1) : '—';

    const activeWrap = document.createElement('div');
    activeWrap.className = 'pair-active';
    const active = document.createElement('input');
    active.type = 'radio';
    active.name = 'activePair';
    active.checked = index === state.activeLanguage;
    active.title = 'Default direction at startup';
    active.addEventListener('change', function () { state.activeLanguage = index; });
    activeWrap.appendChild(active);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'icon';
    remove.textContent = '×';
    remove.title = 'Remove this pair';
    remove.addEventListener('click', function () {
      state.pairs.splice(index, 1);
      delete state.precacheState[index];
      if (state.activeLanguage >= state.pairs.length) {
        state.activeLanguage = Math.max(0, state.pairs.length - 1);
      }
      renderPairs();
      reportSize();
    });

    row.append(sourceSel, targetSel, hotkey, activeWrap,
               makePrecacheButton(index, pair.target), remove);
    list.appendChild(row);
  });
}

$('addPair').addEventListener('click', function () {
  state.pairs.push({ source: AUTO_SOURCE, target: 'English',
                     label: pairLabel(AUTO_SOURCE, 'English') });
  renderPairs();
  reportSize();
});

// ---- tabs -------------------------------------------------------------------
function selectTab(name) {
  document.querySelectorAll('.tab').forEach(function (t) {
    t.classList.toggle('is-active', t.dataset.tab === name);
  });
  document.querySelectorAll('.panel').forEach(function (p) {
    p.classList.toggle('is-active', p.id === 'panel-' + name);
  });
  reportSize(); // each tab has a different height; refit the window
}
document.querySelectorAll('.tab').forEach(function (t) {
  t.addEventListener('click', function () { selectTab(t.dataset.tab); });
});

// ---- lifecycle timeouts (value + minutes/seconds unit; C++ stores seconds) ---
function setTimeInput(valId, unitId, seconds) {
  if (seconds >= 60 && seconds % 60 === 0) {
    $(valId).value = seconds / 60;
    $(unitId).value = '60';
  } else {
    $(valId).value = seconds;
    $(unitId).value = '1';
  }
}

function getTimeInput(valId, unitId, fallbackSec) {
  const value = parseInt($(valId).value, 10);
  if (!value || value <= 0) return fallbackSec;
  return value * parseInt($(unitId).value, 10);
}

// ---- config <-> form --------------------------------------------------------
function applyConfig(cfg) {
  state.activation = cfg.activation || { modifiers: 0, vk: 0 };
  state.commit     = cfg.commit     || { modifiers: 0, vk: 0 };
  state.cycle      = cfg.cycle      || { modifiers: 0, vk: 0 };
  $('activation').value = hotkeyLabel(state.activation);
  $('commit').value     = hotkeyLabel(state.commit);
  $('cycle').value      = hotkeyLabel(state.cycle);

  state.pairs = (Array.isArray(cfg.languagePairs) ? cfg.languagePairs : [])
    .map(function (p) {
      return { label: p.label || '', target: p.target || '',
               source: sourceOfLabel(p.label) };
    });
  state.activeLanguage = cfg.activeLanguage != null ? cfg.activeLanguage : 0;
  state.savedTargets = state.pairs.map(function (p) { return p.target; });
  state.precacheState = {};
  renderPairs();

  $('modelPath').value     = cfg.modelPath || '';
  $('spillFilePath').value = cfg.spillFilePath || '';
  setTimeInput('kvSpillVal', 'kvSpillUnit', cfg.kvSpillTimeoutSec || 600);
  setTimeInput('hibernateVal', 'hibernateUnit', cfg.hibernateTimeoutSec || 1800);

  $('contextSize').value  = cfg.contextSize != null ? cfg.contextSize : 4096;
  $('temperature').value  = cfg.temperature != null ? cfg.temperature : 0.7;
  $('topP').value         = cfg.topP != null ? cfg.topP : 0.95;
  $('maxTokens').value    = cfg.maxTokens != null ? cfg.maxTokens : 1024;
  $('granularity').value  = cfg.captureGranularity || 'sentence';
  $('idleTimerMs').value  = cfg.idleTimerMs != null ? cfg.idleTimerMs : 700;
  $('developerMode').checked = cfg.developerMode === true;
  $('activateOnStartup').checked = cfg.activateOnStartup === true;
  $('vramCacheBlocks').value = cfg.vramCacheBlocks != null ? cfg.vramCacheBlocks : 1024;
  $('ramTierBlocks').value   = cfg.ramTierBlocks != null ? cfg.ramTierBlocks : 2048;
  $('diskSpillEnabled').checked = cfg.diskSpillEnabled !== false;
  $('diskSpillBlocks').value = cfg.diskSpillBlocks != null ? cfg.diskSpillBlocks : 8192;
}

function buildPayload() {
  // Trim empty rows so a stray blank pair never reaches the config.
  const pairs = state.pairs
    .filter(function (p) { return (p.target || '').trim().length > 0; })
    .map(function (p) {
      return { label: pairLabel(p.source || AUTO_SOURCE, p.target), target: p.target };
    });
  let active = state.activeLanguage;
  if (active < 0 || active >= pairs.length) active = 0;

  return {
    type: 'save',
    activation: state.activation,
    commit: state.commit,
    cycle: state.cycle,
    languagePairs: pairs,
    activeLanguage: active,
    modelPath: $('modelPath').value,
    spillFilePath: $('spillFilePath').value,
    kvSpillTimeoutSec: getTimeInput('kvSpillVal', 'kvSpillUnit', 600),
    hibernateTimeoutSec: getTimeInput('hibernateVal', 'hibernateUnit', 1800),
    contextSize: parseInt($('contextSize').value, 10) || 4096,
    temperature: parseFloat($('temperature').value) || 0.0,
    topP: parseFloat($('topP').value) || 0.0,
    maxTokens: parseInt($('maxTokens').value, 10) || 1024,
    captureGranularity: $('granularity').value,
    idleTimerMs: parseInt($('idleTimerMs').value, 10) || 700,
    developerMode: $('developerMode').checked,
    activateOnStartup: $('activateOnStartup').checked,
    vramCacheBlocks: parseInt($('vramCacheBlocks').value, 10) || 0,
    ramTierBlocks: parseInt($('ramTierBlocks').value, 10) || 0,
    diskSpillEnabled: $('diskSpillEnabled').checked,
    diskSpillBlocks: parseInt($('diskSpillBlocks').value, 10) || 0
  };
}

function showToast() {
  const t = $('toast');
  t.classList.add('show');
  setTimeout(function () { t.classList.remove('show'); }, 1600);
}

// Ask the host to size its window to the content so there are no scrollbars.
// Measure body (content) height, not documentElement (clamped to the viewport).
function reportSize() {
  try {
    const h = Math.ceil(document.body.scrollHeight) + 2; // +2 guards DPI rounding
    if (bridge) bridge.postMessage({ type: 'resize', height: h });
  } catch (err) { /* ignore */ }
}

// ---- host IPC ---------------------------------------------------------------
if (bridge) {
  bridge.addEventListener('message', function (event) {
    try {
      const msg = event.data;
      if (!msg || typeof msg !== 'object') return;
      if (msg.type === 'load') {
        applyConfig(msg);
        reportSize();
      } else if (msg.type === 'browsed') {
        // Folder picker result. The spill path is a FILE inside the picked
        // directory; everything else takes the directory itself.
        const el = $(msg.target);
        if (el && msg.path) {
          el.value = msg.target === 'spillFilePath'
            ? msg.path.replace(/[\\/]+$/, '') + '\\spill.bkv'
            : msg.path;
        }
      } else if (msg.type === 'saved') {
        // The just-saved pairs are live now (hot-reload) -- pre-cache may
        // target them.
        state.savedTargets = state.pairs.map(function (p) { return p.target; });
        showToast();
      } else if (msg.type === 'precacheDone') {
        if (msg.ok) {
          state.precacheState[msg.id] = 'done';
          renderPairs();
        } else {
          flashPrecacheState(msg.id, 'error');
        }
      }
    } catch (err) {
      console.error('settings: failed to handle host message', err);
    }
  });
}

document.querySelectorAll('button[data-target]').forEach(function (btn) {
  btn.addEventListener('click', function () {
    try { if (bridge) bridge.postMessage({ type: 'browse', target: btn.dataset.target }); }
    catch (err) { console.error('settings: failed to request folder picker', err); }
  });
});

$('save').addEventListener('click', function () {
  try { if (bridge) bridge.postMessage(buildPayload()); }
  catch (err) { console.error('settings: failed to post save message', err); }
});

bindHotkey('activation');
bindHotkey('commit');
bindHotkey('cycle');
window.addEventListener('load', reportSize);
