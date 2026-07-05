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

// ---- shared state -----------------------------------------------------------
const state = {
  activation: { modifiers: 0, vk: 0 },
  commit:     { modifiers: 0, vk: 0 },
  cycle:      { modifiers: 0, vk: 0 },
  pairs:      [],      // [{ label, target }]
  activeLanguage: 0
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
    if (CLEAR_VK.indexOf(e.keyCode) !== -1) {   // Backspace / Delete unbinds
      state[id] = { modifiers: 0, vk: 0 };
      el.value = '';
      return;
    }
    if (MOD_VK.indexOf(e.keyCode) !== -1) return; // wait for a real (non-modifier) key
    state[id] = { modifiers: modsFromEvent(e), vk: e.keyCode };
    el.value = hotkeyLabel(state[id]);
  });
}

// ---- language pairs ---------------------------------------------------------
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

    const label = document.createElement('input');
    label.type = 'text';
    label.value = pair.label || '';
    label.placeholder = 'RU -> EN';
    label.addEventListener('input', function () { state.pairs[index].label = label.value; });

    const target = document.createElement('input');
    target.type = 'text';
    target.value = pair.target || '';
    target.placeholder = 'English';
    target.addEventListener('input', function () { state.pairs[index].target = target.value; });

    const hotkey = document.createElement('div');
    hotkey.className = 'pair-hotkey';
    hotkey.textContent = index < MAX_HOTKEY_PAIRS ? ('Alt+' + (index + 1)) : '—';

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
      if (state.activeLanguage >= state.pairs.length) {
        state.activeLanguage = Math.max(0, state.pairs.length - 1);
      }
      renderPairs();
      reportSize();
    });

    row.appendChild(label);
    row.appendChild(target);
    row.appendChild(hotkey);
    row.appendChild(activeWrap);
    row.appendChild(remove);
    list.appendChild(row);
  });
}

$('addPair').addEventListener('click', function () {
  state.pairs.push({ label: '', target: '' });
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

// ---- config <-> form --------------------------------------------------------
function applyConfig(cfg) {
  state.activation = cfg.activation || { modifiers: 0, vk: 0 };
  state.commit     = cfg.commit     || { modifiers: 0, vk: 0 };
  state.cycle      = cfg.cycle      || { modifiers: 0, vk: 0 };
  $('activation').value = hotkeyLabel(state.activation);
  $('commit').value     = hotkeyLabel(state.commit);
  $('cycle').value      = hotkeyLabel(state.cycle);

  state.pairs = Array.isArray(cfg.languagePairs)
    ? cfg.languagePairs.map(function (p) { return { label: p.label || '', target: p.target || '' }; })
    : [];
  state.activeLanguage = cfg.activeLanguage != null ? cfg.activeLanguage : 0;
  renderPairs();

  $('modelPath').value    = cfg.modelPath || '';
  $('contextSize').value  = cfg.contextSize != null ? cfg.contextSize : 4096;
  $('temperature').value  = cfg.temperature != null ? cfg.temperature : 0.7;
  $('topP').value         = cfg.topP != null ? cfg.topP : 0.95;
  $('maxTokens').value    = cfg.maxTokens != null ? cfg.maxTokens : 1024;
  $('granularity').value  = cfg.captureGranularity || 'sentence';
  $('idleTimerMs').value  = cfg.idleTimerMs != null ? cfg.idleTimerMs : 700;
  $('activateOnStartup').checked = cfg.activateOnStartup === true;
  $('kvSpillTimeoutMin').value   = cfg.kvSpillTimeoutMin != null ? cfg.kvSpillTimeoutMin : 10;
  $('hibernateTimeoutMin').value = cfg.hibernateTimeoutMin != null ? cfg.hibernateTimeoutMin : 30;
  $('vramCacheBlocks').value = cfg.vramCacheBlocks != null ? cfg.vramCacheBlocks : 1024;
  $('ramTierBlocks').value   = cfg.ramTierBlocks != null ? cfg.ramTierBlocks : 2048;
  $('diskSpillEnabled').checked = cfg.diskSpillEnabled !== false;
  $('diskSpillBlocks').value = cfg.diskSpillBlocks != null ? cfg.diskSpillBlocks : 8192;
  $('spillFilePath').value   = cfg.spillFilePath || '';
}

function buildPayload() {
  // Trim empty rows so a stray blank pair never reaches the config.
  const pairs = state.pairs
    .map(function (p) { return { label: (p.label || '').trim(), target: (p.target || '').trim() }; })
    .filter(function (p) { return p.target.length > 0; });
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
    contextSize: parseInt($('contextSize').value, 10) || 4096,
    temperature: parseFloat($('temperature').value) || 0.0,
    topP: parseFloat($('topP').value) || 0.0,
    maxTokens: parseInt($('maxTokens').value, 10) || 1024,
    captureGranularity: $('granularity').value,
    idleTimerMs: parseInt($('idleTimerMs').value, 10) || 700,
    activateOnStartup: $('activateOnStartup').checked,
    kvSpillTimeoutMin: parseInt($('kvSpillTimeoutMin').value, 10) || 10,
    hibernateTimeoutMin: parseInt($('hibernateTimeoutMin').value, 10) || 30,
    vramCacheBlocks: parseInt($('vramCacheBlocks').value, 10) || 0,
    ramTierBlocks: parseInt($('ramTierBlocks').value, 10) || 0,
    diskSpillEnabled: $('diskSpillEnabled').checked,
    diskSpillBlocks: parseInt($('diskSpillBlocks').value, 10) || 0,
    spillFilePath: $('spillFilePath').value
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
      } else if (msg.type === 'modelPath') {
        if (msg.path) $('modelPath').value = msg.path;
      } else if (msg.type === 'saved') {
        showToast();
      }
    } catch (err) {
      console.error('settings: failed to handle host message', err);
    }
  });
}

$('browse').addEventListener('click', function () {
  try { if (bridge) bridge.postMessage({ type: 'browse' }); }
  catch (err) { console.error('settings: failed to request folder picker', err); }
});

$('save').addEventListener('click', function () {
  try { if (bridge) bridge.postMessage(buildPayload()); }
  catch (err) { console.error('settings: failed to post save message', err); }
});

bindHotkey('activation');
bindHotkey('commit');
bindHotkey('cycle');
window.addEventListener('load', reportSize);
