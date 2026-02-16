// Shakar Playground - Web Worker-based execution with syntax highlighting

const EXAMPLES = {
    hello: `print("Hello, World!")`,
    fib: `fib := fn(n):
    if n <= 1:
        n
    else:
        fib(n - 1) + fib(n - 2)

print(fib(10))`,
    loop: `# Sum numbers 1-5
sum := 0
for x in [1, 2, 3, 4, 5]:
    sum = sum + x
print(sum)`,
    tetris: null,  // Special case - loaded from file
    exhaustive: null  // Special case - loaded from file
};

let worker = null;
let shakarLoaded = false;
let wasmHighlightActive = false;
let highlightTimeout = null;
let tetrisCode = null;
let exhaustiveCode = null;
let tetrisRunning = false;
let replMode = false;
let replContinuation = false;
let replContinuationIndent = '';
let replContinuationAutoIndent = false;
let replLines = [];
let replHistory = [];
let replHistoryIndex = -1;
let replHistoryDraft = '';
let replHistoryValue = '';
let replHistoryDisplay = '';
let replPending = false;
let replInitialized = false;
let replSuggestionMatches = [];
let replSuggestionIndex = -1;
let replLineEls = [];
let replHighlightSeq = 0;
let replHighlightRequests = new Map();
let replLastLiveHighlightId = 0;
let replLastBlockHighlightId = 0;
let replHighlightTimer = 0;
let replIndentProbeSeq = 0;
let replIndentProbeId = 0;
let replIndentProbeIndent = '';
let replIndentProbeBaseIndent = '';
let replIndentProbeWasBlock = false;
let editorOutdentLineStart = -1;
let editorIndentAdjusting = false;
let editorLastInputAt = 0;
let editorHighlightSeq = 0;
let editorHighlightRequestId = 0;
let editorHighlightRequestCode = '';
let editorHighlightRequestAt = 0;
let editorHighlightRequestStart = 0;
let editorHighlightRequestEnd = 0;
let editorHighlightRequestLines = null;
let editorLastHighlightCode = '';
let editorLastLineCount = 1;
let editorErrorRevealTimer = 0;
let lastStatusError = '';

const EDITOR_LARGE_DOC_CHARS = 12000;
const EDITOR_HIGHLIGHT_DELAY_SMALL = 30;
const EDITOR_HIGHLIGHT_DELAY_LARGE = 60;
const EDITOR_ERROR_HIGHLIGHT_DELAY = 350;
const REPL_HISTORY_LIMIT = 200;

const REPL_COMMANDS = [
    {cmd: '/clear', desc: 'Clear screen'},
    {
        cmd: '/py-traceback',
        display: '/py-traceback [on|off]',
        desc: 'Toggle python traceback',
        insert: '/py-traceback '
    },
    {cmd: '/editor', desc: 'Switch to editor'}
];

// SharedArrayBuffer for sending keys to worker while Python blocks.
// Layout: Int32Array[0]=write_idx, [1..32]=key codes (ring buffer, 32 slots)
// Key encoding: left=1, right=2, down=3, up=4, space=5, q=6
const KEY_BUF_SLOTS = 32;
const supportsSharedArrayBuffer = (
    typeof SharedArrayBuffer === 'function' &&
    typeof Atomics === 'object' &&
    self.crossOriginIsolated === true
);
const keyBuffer = supportsSharedArrayBuffer
    ? new SharedArrayBuffer((1 + KEY_BUF_SLOTS) * 4)
    : null;
const keyBufView = keyBuffer ? new Int32Array(keyBuffer) : null;

const KEY_CODES = {
    'left': 1, 'right': 2, 'down': 3, 'up': 4, ' ': 5, 'q': 6
};

function pushKey(key) {
    if (!keyBufView) return;
    const code = KEY_CODES[key];
    if (!code) return;
    const idx = Atomics.load(keyBufView, 0);
    // Coerce to uint32 so ring math stays correct after Int32 wraparound.
    const slot = ((idx >>> 0) % KEY_BUF_SLOTS) + 1;
    Atomics.store(keyBufView, slot, code);
    Atomics.store(keyBufView, 0, (idx + 1) | 0);
    Atomics.notify(keyBufView, 0);
}

const statusEl = document.getElementById('status');
const statusHintEl = document.getElementById('status-hint');
const splashEl = document.getElementById('splash');
const splashStatusEl = document.getElementById('splash-status');
const splashHintEl = document.getElementById('splash-hint');
const codeEl = document.getElementById('code');
const highlightEl = document.getElementById('highlight-layer');
const outputEl = document.getElementById('output');
const tracebackEl = document.getElementById('traceback');
const tracebackTextEl = document.getElementById('traceback-text');
const runBtn = document.getElementById('run-btn');
const clearBtn = document.getElementById('clear-btn');
const examplesEl = document.getElementById('examples');
const replToggleBtn = document.getElementById('repl-toggle-btn');
const replPane = document.querySelector('.repl-pane');
const replBody = document.querySelector('.repl-body');
const replTranscript = document.getElementById('repl-transcript');
const replLiveLine = document.getElementById('repl-live-line');
const replInputWrap = document.querySelector('.repl-input-wrap');
const replInputHl = document.getElementById('repl-input-hl');
const replInput = document.getElementById('repl-input');
const replPrompt = document.getElementById('repl-prompt');
const replResetBtn = document.getElementById('repl-reset-btn');
const replSuggestionsEl = document.getElementById('repl-suggestions');
const editorPane = document.querySelector('.editor-pane');
const outputPane = document.querySelector('.output-pane');
const mainEl = document.querySelector('main');

function pruneWhitespaceTextNodes(root) {
    if (!root) return;
    const nodes = Array.from(root.childNodes);
    for (const node of nodes) {
        if (node.nodeType === Node.TEXT_NODE && !node.nodeValue.trim()) {
            root.removeChild(node);
        }
    }
}

/* Prevent template indentation nodes from becoming selectable blank lines in REPL UI. */
pruneWhitespaceTextNodes(replLiveLine);
pruneWhitespaceTextNodes(replInputWrap);

const DEBUG_IO = localStorage.getItem('shakar_debug_io') === '1';
let DEBUG_PY_TRACE = localStorage.getItem('shakar_debug_py_trace') === '1';

const pyTraceBtn = document.getElementById('py-trace-btn');
function _syncPyTraceBtn() {
    pyTraceBtn.classList.toggle('active', DEBUG_PY_TRACE);
}
_syncPyTraceBtn();
function setPyTraceEnabled(enabled) {
    DEBUG_PY_TRACE = enabled;
    if (DEBUG_PY_TRACE) {
        localStorage.setItem('shakar_debug_py_trace', '1');
    } else {
        localStorage.removeItem('shakar_debug_py_trace');
    }
    _syncPyTraceBtn();
}
pyTraceBtn.addEventListener('click', () => {
    setPyTraceEnabled(!DEBUG_PY_TRACE);
});
const SHAKAR_VERSION = window.SHAKAR_VERSION || 'dev';

// Batch io_write updates to one per animation frame to prevent flicker.
let pendingWrite = null;
let pendingClear = false;
let writeRafId = 0;

function flushWrite() {
    if (pendingClear) {
        if (pendingWrite !== null) {
            outputEl.textContent = pendingWrite;
            pendingWrite = null;
        } else {
            outputEl.textContent = '';
        }
        pendingClear = false;
    } else if (pendingWrite !== null) {
        outputEl.textContent += pendingWrite;
        pendingWrite = null;
    }
    writeRafId = 0;
}

function scheduleWrite(text) {
    if (pendingWrite === null) {
        pendingWrite = text;
    } else {
        pendingWrite += text;
    }
    if (!writeRafId) {
        writeRafId = requestAnimationFrame(flushWrite);
    }
}

function scheduleClear() {
    pendingClear = true;
    pendingWrite = null;
    if (!writeRafId) {
        writeRafId = requestAnimationFrame(flushWrite);
    }
}

function scheduleOverwrite(text) {
    pendingClear = true;
    pendingWrite = text;
    if (!writeRafId) {
        writeRafId = requestAnimationFrame(flushWrite);
    }
}

function setStatus(text, state = '') {
    statusEl.textContent = text;
    statusEl.className = 'status ' + state;
    if (statusHintEl) {
        statusHintEl.classList.toggle('hidden', state !== 'error');
    }
    if (splashStatusEl) {
        splashStatusEl.textContent = text;
    }
    if (splashHintEl && state === 'error') {
        splashHintEl.textContent = 'Open your web console for details.';
    }
    if (splashEl && state === 'error') {
        splashEl.classList.remove('splash-hidden');
        splashEl.classList.add('error');
        splashEl.setAttribute('aria-busy', 'false');
    } else if (splashEl) {
        splashEl.classList.remove('error');
    }
    if (state === 'error' && text && text !== lastStatusError) {
        lastStatusError = text;
        console.error(text);
    }
}

function setReplMode(enabled) {
    replMode = enabled;
    mainEl.classList.toggle('repl-mode', enabled);
    editorPane.classList.toggle('hidden', enabled);
    outputPane.classList.toggle('hidden', enabled);
    replPane.classList.toggle('hidden', !enabled);
    replToggleBtn.classList.toggle('active', enabled);
    replToggleBtn.textContent = enabled ? 'Editor' : 'REPL';
    replToggleBtn.setAttribute('aria-pressed', enabled ? 'true' : 'false');

    if (enabled) {
        if (!replInitialized) {
            initReplTranscript();
            replInitialized = true;
        }
        stopTetris();
        replInput.focus();
        scheduleReplLiveHighlight();
        syncReplInputHeight();
    } else {
        hideReplSuggestions();
    }
}

function setTraceback(text) {
    if (!tracebackEl || !tracebackTextEl) return;
    if (!text) {
        tracebackTextEl.textContent = '';
        tracebackEl.open = false;
        tracebackEl.classList.add('hidden');
        return;
    }
    tracebackTextEl.textContent = text;
    tracebackEl.open = false;
    tracebackEl.classList.remove('hidden');
}

function setOutput(text, isError = false, traceback = '', errorRange = null) {
    outputEl.textContent = text;
    outputEl.className = isError ? 'error' : '';

    if (
        isError &&
        errorRange &&
        Number.isFinite(errorRange.line) &&
        Number.isFinite(errorRange.col_start) &&
        Number.isFinite(errorRange.col_end)
    ) {
        highlightErrorRange(
            errorRange.line,
            errorRange.col_start,
            errorRange.col_end
        );
    } else if (currentError) {
        clearError();
        updateHighlights();
    }

    if (isError && traceback) {
        setTraceback(traceback);
    } else {
        setTraceback('');
    }
}

function scrollReplToBottom() {
    if (!replTranscript) return;
    replTranscript.scrollTop = replTranscript.scrollHeight;
}

function initReplTranscript() {
    if (!replTranscript || !replLiveLine) return;
    replTranscript.textContent = '';
    replTranscript.appendChild(replLiveLine);
    if (replSuggestionsEl) {
        replTranscript.appendChild(replSuggestionsEl);
    }
    replHighlightRequests.clear();
    replLastLiveHighlightId = 0;
    hideReplSuggestions();
    setReplInputHighlight('');
    replInput.value = '';
    syncReplInputHeight();
}

function appendReplLine(text, className) {
    if (!replTranscript) return;
    const lineEl = document.createElement('div');
    lineEl.className = className ? `repl-line ${className}` : 'repl-line';
    lineEl.textContent = text;
    replTranscript.insertBefore(lineEl, replLiveLine);
    scrollReplToBottom();
}

function appendReplInputLine(prompt, text) {
    if (!replTranscript) return;
    const lineEl = document.createElement('div');
    lineEl.className = 'repl-line repl-line-input';
    const promptEl = document.createElement('span');
    promptEl.className = 'repl-prompt';
    promptEl.textContent = prompt;
    const textEl = document.createElement('span');
    textEl.textContent = text;
    lineEl.appendChild(promptEl);
    lineEl.appendChild(textEl);
    replTranscript.insertBefore(lineEl, replLiveLine);
    scrollReplToBottom();
    return textEl;
}

function appendReplInputBlock(lines) {
    if (!lines.length) return;
    const textEls = [];
    textEls.push(appendReplInputLine('>>> ', lines[0]));
    for (let i = 1; i < lines.length; i++) {
        textEls.push(appendReplInputLine('... ', lines[i]));
    }
    return textEls;
}

function appendReplOutput(text, isError) {
    if (!text) return;
    const lines = text.split('\n');
    const className = isError ? 'repl-line-error' : 'repl-line-output';
    for (const line of lines) {
        appendReplLine(line, className);
    }
}

function appendReplMetaOutput(text, isError) {
    if (!text) return;
    const lines = text.split('\n');
    const baseClass = isError ? 'repl-line-error' : 'repl-line-output';
    const className = `${baseClass} repl-line-meta-output`;
    for (const line of lines) {
        appendReplLine(line, className);
    }
}

function normalizeReplInputText(text) {
    return text.replace(/\r/g, '').replace(/\u200b/g, '').replace(/\u00a0/g, ' ');
}

function getReplInputText() {
    return normalizeReplInputText(replInput.value || '');
}

function shouldAllowReplManualMultiline() {
    const text = getReplInputText();
    if (text.includes('\n')) {
        return true;
    }

    if (replHistoryIndex >= 0 && replHistoryValue.includes('\n')) {
        return true;
    }

    return false;
}

function getLineIndent(text) {
    const match = text.match(/^[ \t]*/);
    return match ? match[0] : '';
}

function isBlockHeaderLine(text) {
    let depth = 0;
    let sawQuestion = false;
    let colonCount = 0;
    let lastSig = '';
    let inSingle = false;
    let inDouble = false;
    let escaped = false;

    for (let i = 0; i < text.length; i++) {
        const ch = text[i];

        if (inSingle) {
            if (escaped) {
                escaped = false;
            } else if (ch === '\\') {
                escaped = true;
            } else if (ch === '\'') {
                inSingle = false;
            }
            continue;
        }

        if (inDouble) {
            if (escaped) {
                escaped = false;
            } else if (ch === '\\') {
                escaped = true;
            } else if (ch === '"') {
                inDouble = false;
            }
            continue;
        }

        if (ch === '#') {
            break;
        }

        if (ch === '\'') {
            inSingle = true;
            continue;
        }
        if (ch === '"') {
            inDouble = true;
            continue;
        }

        if (ch === '(' || ch === '[' || ch === '{') {
            depth += 1;
        } else if (ch === ')' || ch === ']' || ch === '}') {
            depth = Math.max(0, depth - 1);
        }

        if (depth === 0) {
            if (ch === '?') {
                sawQuestion = true;
            } else if (ch === ':') {
                colonCount += 1;
            }

            if (ch.trim()) {
                lastSig = ch;
            }
        }
    }

    return depth === 0 && lastSig === ':' && !(sawQuestion && colonCount === 1);
}

function applyReplContinuationIndent(line) {
    if (!replContinuationAutoIndent || line === '') {
        return line;
    }
    if (/^[ \t]/.test(line)) {
        return line;
    }
    return replContinuationIndent + line;
}

function isLargeEditorDoc(code) {
    return code.length >= EDITOR_LARGE_DOC_CHARS;
}

function shouldSuppressEditorErrors() {
    return performance.now() - editorLastInputAt < EDITOR_ERROR_HIGHLIGHT_DELAY;
}

function scheduleEditorErrorReveal() {
    if (editorErrorRevealTimer) return;
    const elapsed = performance.now() - editorLastInputAt;
    const wait = Math.max(0, EDITOR_ERROR_HIGHLIGHT_DELAY - elapsed);
    editorErrorRevealTimer = setTimeout(() => {
        editorErrorRevealTimer = 0;
        scheduleHighlight();
    }, wait);
}

function countLines(value) {
    let count = 1;
    for (let i = 0; i < value.length; i++) {
        if (value[i] === '\n') {
            count += 1;
        }
    }
    return count;
}

function getLineInfo(value, pos) {
    const lineStart = value.lastIndexOf('\n', pos - 1) + 1;
    const lineEndIdx = value.indexOf('\n', pos);
    const lineEnd = lineEndIdx === -1 ? value.length : lineEndIdx;
    const lineText = value.slice(lineStart, lineEnd);
    const lineIndex = countLines(value.slice(0, lineStart)) - 1;
    return {lineStart, lineEnd, lineText, lineIndex};
}

function updateEditorHighlightLine(lineIdx, text) {
    if (!highlightEl) return;
    const lineEl = highlightEl.children[lineIdx];
    if (!lineEl) return;
    lineEl.className = 'hl-line';
    lineEl.innerHTML = escapeHtml(text) || ' ';
}

function adjustEditorHighlightLineCount(value, pos) {
    if (!highlightEl) return;
    const newCount = countLines(value);
    if (newCount === editorLastLineCount) return;

    const lineIndex = countLines(value.slice(0, pos)) - 1;

    if (newCount === editorLastLineCount + 1) {
        const lineEl = document.createElement('div');
        lineEl.className = 'hl-line';
        lineEl.innerHTML = ' ';
        const ref = highlightEl.children[lineIndex];
        if (ref) {
            highlightEl.insertBefore(lineEl, ref);
        } else {
            highlightEl.appendChild(lineEl);
        }
    } else if (newCount === editorLastLineCount - 1) {
        const removeIndex = Math.min(
            lineIndex + 1,
            highlightEl.children.length - 1
        );
        const toRemove = highlightEl.children[removeIndex];
        if (toRemove) {
            highlightEl.removeChild(toRemove);
        }
    } else {
        highlightEl.innerHTML = value
            .split('\n')
            .map((line) => `<div class="hl-line">${escapeHtml(line) || ' '}</div>`)
            .join('');
    }

    editorLastLineCount = newCount;
}

function applyEditorLightUpdate(value, pos) {
    if (!isLargeEditorDoc(value)) return;
    const info = getLineInfo(value, pos);
    adjustEditorHighlightLineCount(value, pos);
    updateEditorHighlightLine(info.lineIndex, info.lineText);
    syncScroll();
}

function getEditorLineHeight() {
    return parseFloat(getComputedStyle(codeEl).lineHeight) || 20;
}

function getEditorVisibleRange(totalLines) {
    const lineHeight = getEditorLineHeight();
    const buffer = 10;
    const start = Math.max(0, Math.floor(codeEl.scrollTop / lineHeight) - buffer);
    const end = Math.min(
        Math.max(0, totalLines - 1),
        Math.ceil((codeEl.scrollTop + codeEl.clientHeight) / lineHeight) + buffer
    );
    return {start, end};
}

function applyHighlightsRange(
    lines,
    highlights,
    startLine,
    endLine,
    byLine = null,
    errorLines = null,
    suppressErrors = false
) {
    if (!highlightEl) return;
    if ((!highlights || highlights.length === 0) && !byLine) {
        for (let lineIdx = startLine; lineIdx <= endLine; lineIdx++) {
            const lineEl = highlightEl.children[lineIdx];
            if (!lineEl) continue;
            lineEl.className = 'hl-line';
            lineEl.innerHTML = escapeHtml(lines[lineIdx] || '') || ' ';
        }
        return;
    }
    const hlByLine = byLine || {};
    const errorLineSet = errorLines || new Set();
    let sawError = false;
    if (!byLine) {
        for (const hl of highlights || []) {
            if (suppressErrors && hl.group === 'error') {
                sawError = true;
                continue;
            }
            const line = hl.line;
            if (line < startLine || line > endLine) continue;
            if (!hlByLine[line]) hlByLine[line] = [];
            hlByLine[line].push(hl);
            if (hl.group === 'error') {
                errorLineSet.add(line);
            }
        }
    }

    for (let lineIdx = startLine; lineIdx <= endLine; lineIdx++) {
        const line = lines[lineIdx] || '';
        const lineHls = hlByLine[lineIdx] || [];
        const hasError = errorLineSet.has(lineIdx);
        const html = renderHighlightedLine(line, lineHls);

        const lineEl = highlightEl.children[lineIdx];
        if (!lineEl) continue;
        lineEl.className = hasError ? 'hl-line hl-line-error' : 'hl-line';
        lineEl.innerHTML = html || ' ';
    }

    if (suppressErrors && sawError) {
        scheduleEditorErrorReveal();
    }
}

function refreshVisibleEditorHighlights() {
    if (!isLargeEditorDoc(codeEl.value)) return;
    scheduleHighlight();
}

function requestReplIndentProbe(line, indentStr, baseIndent, wasBlock) {
    if (!worker || !shakarLoaded) return;
    const requestId = ++replIndentProbeSeq;
    replIndentProbeId = requestId;
    replIndentProbeIndent = indentStr;
    replIndentProbeBaseIndent = baseIndent;
    replIndentProbeWasBlock = wasBlock;
    worker.postMessage({type: 'lex_probe', line: line, requestId: requestId});
}

function handleReplIndentProbeResult(msg) {
    if (!replContinuation) return;
    if (msg.requestId !== replIndentProbeId) return;
    const wantsBlock = msg.isBlockHeader === true;
    if (wantsBlock === replIndentProbeWasBlock) return;
    const current = getReplInputText();
    const safeToModify = !current.trim() || current === replIndentProbeIndent;

    if (wantsBlock) {
        replContinuationAutoIndent = true;
        replContinuationIndent = replIndentProbeIndent;
        if (safeToModify) {
            setReplInputText(replIndentProbeIndent);
        }
        return;
    }

    if (replContinuationAutoIndent && replLines.length === 1 && safeToModify) {
        replContinuation = false;
        replContinuationAutoIndent = false;
        replContinuationIndent = '';
        replPrompt.textContent = '>>> ';
        setReplInputText('');
        submitReplBlock(replLines);
        return;
    }

    replContinuationAutoIndent = false;
    replContinuationIndent = replIndentProbeBaseIndent;
    if (safeToModify) {
        setReplInputText(replIndentProbeBaseIndent);
    }
}

function setReplInputHighlight(html) {
    if (!replInputHl) return;
    replInputHl.innerHTML = html || '';
}

function syncReplInputHeight() {
    if (!replInput || !replInputWrap) return;
    /* Measure from zero height to avoid browser default textarea row inflation. */
    replInput.style.height = '0px';
    const computed = getComputedStyle(replInput);
    const lineHeight = parseFloat(computed.lineHeight) || 20;
    const hasText = !!replInput.value;
    const measuredHeight = replInput.scrollHeight;
    const height = hasText ? Math.max(measuredHeight, lineHeight) : lineHeight;
    replInput.style.height = `${height}px`;
    replInputWrap.style.height = `${height}px`;
    if (replInputHl) {
        replInputHl.style.height = `${height}px`;
    }
}

function setReplInputText(value) {
    replInput.value = value;
    replInput.scrollTop = 0;
    setReplInputHighlight(escapeHtml(value));
    scheduleReplLiveHighlight();
    replInput.selectionStart = replInput.value.length;
    replInput.selectionEnd = replInput.value.length;
    syncReplInputHeight();
}

function setReplInputEnabled(enabled) {
    replInput.disabled = !enabled;
    replInput.classList.toggle('disabled', !enabled);
}

function replSuggestionsVisible() {
    return (
        replSuggestionsEl &&
        !replSuggestionsEl.classList.contains('hidden') &&
        replSuggestionMatches.length > 0
    );
}

function hideReplSuggestions() {
    replSuggestionMatches = [];
    replSuggestionIndex = -1;
    if (!replSuggestionsEl) return;
    replSuggestionsEl.textContent = '';
    replSuggestionsEl.classList.add('hidden');
}

function setReplSuggestionIndex(index) {
    if (!replSuggestionsEl || !replSuggestionMatches.length) return;
    const items = replSuggestionsEl.querySelectorAll('.repl-suggestion');
    if (!items.length) return;
    let next = index;
    if (next < 0) next = items.length - 1;
    if (next >= items.length) next = 0;
    replSuggestionIndex = next;
    items.forEach((item, idx) => {
        const selected = idx === replSuggestionIndex;
        item.classList.toggle('selected', selected);
        item.setAttribute('aria-selected', selected ? 'true' : 'false');
    });
}

function applyReplSuggestion() {
    if (!replSuggestionMatches.length) return false;
    const index = replSuggestionIndex >= 0 ? replSuggestionIndex : 0;
    const suggestion = replSuggestionMatches[index];
    if (!suggestion) return false;
    setReplInputText(suggestion.insert || suggestion.cmd);
    hideReplSuggestions();
    return true;
}

function updateReplSuggestions() {
    if (!replSuggestionsEl) return;
    if (replContinuation || replPending) {
        hideReplSuggestions();
        return;
    }
    let line = getReplInputText();
    const trimmed = line.trim();
    if (!trimmed.startsWith('/')) {
        hideReplSuggestions();
        return;
    }
    const matches = REPL_COMMANDS.filter(
        (entry) => entry.cmd.startsWith(trimmed) || trimmed.startsWith(entry.cmd)
    );
    if (!matches.length) {
        hideReplSuggestions();
        return;
    }
    replSuggestionMatches = matches;
    replSuggestionIndex = 0;
    replSuggestionsEl.textContent = '';
    for (let i = 0; i < matches.length; i++) {
        const entry = matches[i];
        const item = document.createElement('div');
        item.className = 'repl-suggestion';
        item.setAttribute('role', 'option');
        item.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
        if (i === 0) {
            item.classList.add('selected');
        }

        const cmdEl = document.createElement('span');
        cmdEl.className = 'repl-suggest-cmd';
        cmdEl.textContent = entry.display || entry.cmd;

        const descEl = document.createElement('span');
        descEl.className = 'repl-suggest-desc';
        descEl.textContent = entry.desc;

        item.appendChild(cmdEl);
        item.appendChild(descEl);
        item.addEventListener('click', (e) => {
            e.preventDefault();
            setReplInputText(entry.insert || entry.cmd);
            hideReplSuggestions();
            replInput.focus();
        });

        replSuggestionsEl.appendChild(item);
    }
    replSuggestionsEl.classList.remove('hidden');
}

function handleReplInterrupt() {
    if (replPending) return;
    let line = getReplInputText();
    const promptText = replPrompt.textContent;
    hideReplSuggestions();
    appendReplInputLine(promptText, line);
    replLines = [];
    replLineEls = [];
    replContinuation = false;
    replContinuationIndent = '';
    replContinuationAutoIndent = false;
    replIndentProbeId = 0;
    replPrompt.textContent = '>>> ';
    setReplInputText('');
    appendReplLine('KeyboardInterrupt', 'repl-line-output');
}

function navigateReplHistory(delta) {
    if (replContinuation || replPending) return;
    if (!replHistory.length) return;

    if (replHistoryIndex === -1) {
        if (delta > 0) return;
        replHistoryDraft = getReplInputText();
        replHistoryIndex = replHistory.length - 1;
    } else {
        replHistoryIndex += delta;
    }

    if (replHistoryIndex < 0) {
        replHistoryIndex = 0;
    } else if (replHistoryIndex >= replHistory.length) {
        replHistoryIndex = -1;
        replHistoryValue = '';
        replHistoryDisplay = '';
        setReplInputText(replHistoryDraft);
        return;
    }

    replHistoryValue = replHistory[replHistoryIndex];
    replHistoryDisplay = replHistoryValue;
    setReplInputText(replHistoryDisplay);
}

function submitReplBlock(lines) {
    const src = lines.join('\n');
    replLines = [];
    const lineEls = replLineEls;
    replLineEls = [];

    if (!src.trim()) return;

    replHistory.push(src);
    if (replHistory.length > REPL_HISTORY_LIMIT) {
        replHistory.splice(0, replHistory.length - REPL_HISTORY_LIMIT);
    }
    replHistoryIndex = -1;
    replHistoryDraft = '';
    replHistoryValue = '';
    replHistoryDisplay = '';

    if (!worker || !shakarLoaded) {
        appendReplLine('Error: REPL not ready', 'repl-line-error');
        return;
    }

    replLastBlockHighlightId = 0;
    if (lineEls.length) {
        requestReplHighlight(src, {kind: 'block', lineEls: lineEls, code: src});
    }

    replPending = true;
    setReplInputEnabled(false);
    worker.postMessage({type: 'repl_eval', code: src, debugPyTrace: DEBUG_PY_TRACE});
}

function handleReplEnter() {
    if (replPending) return;

    let line = getReplInputText();
    const promptText = replPrompt.textContent;
    hideReplSuggestions();

    if (!replContinuation && line.trim() === '/clear') {
        initReplTranscript();
        setReplInputText('');
        replInput.focus();
        replLineEls = [];
        return;
    }

    if (!replContinuation && line.trim().startsWith('/py-traceback')) {
        const parts = line.trim().split(/\s+/);
        replLineEls.push(appendReplInputLine(promptText, line));
        if (parts.length === 1) {
            setPyTraceEnabled(!DEBUG_PY_TRACE);
            appendReplMetaOutput(
                `Py traceback: ${DEBUG_PY_TRACE ? 'on' : 'off'}`,
                false
            );
        } else if (parts[1] === 'on' || parts[1] === 'off') {
            setPyTraceEnabled(parts[1] === 'on');
            appendReplMetaOutput(`Py traceback: ${parts[1]}`, false);
        } else {
            appendReplMetaOutput('Usage: /py-traceback on|off', true);
        }
        setReplInputText('');
        replLineEls = [];
        return;
    }

    if (!replContinuation && line.trim() === '/editor') {
        replLineEls.push(appendReplInputLine(promptText, line));
        appendReplMetaOutput('Switching to editor...', false);
        setReplInputText('');
        replLineEls = [];
        setReplMode(false);
        return;
    }

    if (!replContinuation && line.trim().startsWith('/')) {
        replLineEls.push(appendReplInputLine(promptText, line));
        appendReplMetaOutput(`Unknown command: ${line.trim()}`, true);
        setReplInputText('');
        replLineEls = [];
        return;
    }

    if (replContinuation) {
        if (!line.trim()) {
            const lines = replLines.slice();
            replLines = [];
            replContinuation = false;
            replContinuationIndent = '';
            replContinuationAutoIndent = false;
            replPrompt.textContent = '>>> ';
            setReplInputText('');
            submitReplBlock(lines);
            return;
        }
        line = applyReplContinuationIndent(line);
        replLineEls.push(appendReplInputLine(promptText, line));
        replLines.push(line);
        const lastLine = replLines[replLines.length - 1] || '';
        if (isBlockHeaderLine(lastLine)) {
            const baseIndent = getLineIndent(lastLine);
            replContinuationAutoIndent = true;
            replContinuationIndent = baseIndent + '    ';
            requestReplIndentProbe(lastLine, replContinuationIndent, baseIndent, true);
        } else if (replContinuationAutoIndent) {
            replContinuationIndent = getLineIndent(lastLine);
        }
        const blockCode = replLines.join('\n');
        const blockHighlightId = requestReplHighlight(blockCode, {
            kind: 'block_partial',
            lineEls: replLineEls,
            code: blockCode,
            suppressErrors: isBlockHeaderLine(lastLine)
        });
        if (blockHighlightId) {
            replLastBlockHighlightId = blockHighlightId;
        }
        setReplInputText(replContinuationAutoIndent ? replContinuationIndent : '');
        return;
    }

    if (replHistoryIndex >= 0 && getReplInputText() === replHistoryDisplay) {
        const historyLines = replHistoryValue.split('\n');
        setReplInputText('');
        replHistoryIndex = -1;
        replHistoryValue = '';
        replHistoryDisplay = '';
        replLineEls = appendReplInputBlock(historyLines);
        submitReplBlock(historyLines);
        return;
    }

    if (!line.trim()) {
        setReplInputText('');
        return;
    }

    /* Edited recalled multiline input should keep block-style transcript/rendering. */
    if (line.includes('\n')) {
        const lines = line.split('\n');
        setReplInputText('');
        replLineEls = appendReplInputBlock(lines);
        submitReplBlock(lines);
        return;
    }

    replLines = [line];
    replLineEls.push(appendReplInputLine(promptText, line));

    if (isBlockHeaderLine(line)) {
        replContinuation = true;
        replContinuationAutoIndent = isBlockHeaderLine(line);
        const baseIndent = getLineIndent(line);
        replContinuationIndent = replContinuationAutoIndent
            ? baseIndent + '    '
            : '';
        replPrompt.textContent = '... ';
        setReplInputText(replContinuationAutoIndent ? replContinuationIndent : '');
        if (replContinuationAutoIndent) {
            requestReplIndentProbe(line, replContinuationIndent, baseIndent, true);
        }
        const blockCode = replLines.join('\n');
        const blockHighlightId = requestReplHighlight(blockCode, {
            kind: 'block_partial',
            lineEls: replLineEls,
            code: blockCode,
            suppressErrors: isBlockHeaderLine(line)
        });
        if (blockHighlightId) {
            replLastBlockHighlightId = blockHighlightId;
        }
        return;
    }

    setReplInputText('');

    submitReplBlock(replLines);
}

function resetRepl() {
    initReplTranscript();
    replLines = [];
    replLineEls = [];
    replContinuation = false;
    replContinuationIndent = '';
    replContinuationAutoIndent = false;
    replPrompt.textContent = '>>> ';
    setReplInputText('');
    setReplInputEnabled(true);
    replPending = false;
    if (worker && shakarLoaded) {
        worker.postMessage({type: 'repl_reset'});
    }
}

let currentError = null;

function highlightErrorRange(line, colStart, colEnd) {
    const code = codeEl.value;
    const lines = code.split('\n');

    if (line < 0 || line >= lines.length) return;

    const lineText = lines[line];
    const safeStart = Math.max(0, Math.min(colStart, lineText.length));
    const safeEnd = Math.max(safeStart + 1, Math.min(colEnd, lineText.length));

    currentError = {
        line: line,
        col_start: safeStart,
        col_end: safeEnd,
        group: 'error'
    };

    updateHighlights();

    let pos = 0;
    for (let i = 0; i < line; i++) {
        pos += lines[i].length + 1;
    }
    pos += safeStart;

    codeEl.focus();
    codeEl.setSelectionRange(pos, pos);

    const lineHeight = parseFloat(getComputedStyle(codeEl).lineHeight) || 20;
    const targetScroll = Math.max(0, (line - 3) * lineHeight);
    codeEl.scrollTop = targetScroll;
    syncScroll();
}

function clearError() {
    currentError = null;
}

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function renderHighlightedLine(line, lineHls) {
    if (!lineHls || lineHls.length === 0) {
        return escapeHtml(line);
    }

    const byStart = [...lineHls].sort(
        (a, b) => a.col_start - b.col_start || b.col_end - a.col_end
    );
    const errors = byStart.filter((h) => h.group === 'error');
    const nonErrors = byStart.filter((h) => h.group !== 'error');
    const sorted = [...errors, ...nonErrors].sort(
        (a, b) => a.col_start - b.col_start
    );

    const filtered = [];
    let lastEnd = 0;
    for (const hl of sorted) {
        if (hl.col_start >= lastEnd) {
            filtered.push(hl);
            lastEnd = hl.col_end;
        }
    }

    let html = '';
    let pos = 0;

    for (const hl of filtered) {
        const start = Math.max(0, Math.min(hl.col_start, line.length));
        const end = Math.max(start, Math.min(hl.col_end, line.length));

        if (start > pos) {
            html += escapeHtml(line.slice(pos, start));
        }

        if (end > start) {
            const spanText = escapeHtml(line.slice(start, end));
            html += `<span class="hl-${hl.group}">${spanText}</span>`;
        }

        pos = end;
    }

    if (pos < line.length) {
        html += escapeHtml(line.slice(pos));
    }

    return html;
}

function renderHighlightedLines(code, highlights) {
    const lines = code.split('\n');
    if (!highlights || highlights.length === 0) {
        return lines.map((line) => escapeHtml(line));
    }

    const hlByLine = {};
    for (const hl of highlights) {
        const line = hl.line;
        if (!hlByLine[line]) hlByLine[line] = [];
        hlByLine[line].push(hl);
    }

    const result = [];

    for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
        const line = lines[lineIdx];
        const lineHls = hlByLine[lineIdx] || [];
        result.push(renderHighlightedLine(line, lineHls));
    }

    return result;
}

function applyHighlights(code, highlights) {
    if (shouldSuppressEditorErrors() && highlights && highlights.length) {
        const hadError = highlights.some((hl) => hl.group === 'error');
        highlights = highlights.filter((hl) => hl.group !== 'error');
        if (hadError) {
            scheduleEditorErrorReveal();
        }
    }

    if (!highlights || highlights.length === 0) {
        highlightEl.innerHTML = code.split('\n').map(line =>
            `<div class="hl-line">${escapeHtml(line) || ' '}</div>`
        ).join('');
        editorLastLineCount = code.split('\n').length;
        syncScroll();
        return;
    }

    const lines = code.split('\n');
    const result = [];

    const hlByLine = {};
    for (const hl of highlights) {
        const line = hl.line;
        if (!hlByLine[line]) hlByLine[line] = [];
        hlByLine[line].push(hl);
    }

    const errorLines = new Set();
    for (const hl of highlights) {
        if (hl.group === 'error') {
            errorLines.add(hl.line);
        }
    }

    for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
        const line = lines[lineIdx];
        const lineHls = hlByLine[lineIdx] || [];
        const hasError = errorLines.has(lineIdx);
        const html = renderHighlightedLine(line, lineHls);

        const lineClass = hasError ? 'hl-line hl-line-error' : 'hl-line';
        result.push(`<div class="${lineClass}">${html || ' '}</div>`);
    }

    highlightEl.innerHTML = result.join('');
    editorLastLineCount = lines.length;
    syncScroll();
}

function requestReplHighlight(code, meta) {
    if (!worker || !shakarLoaded) return;
    if (meta.kind === 'live' && replLastLiveHighlightId) {
        replHighlightRequests.delete(replLastLiveHighlightId);
    } else if (meta.kind === 'block_partial' && replLastBlockHighlightId) {
        replHighlightRequests.delete(replLastBlockHighlightId);
    }
    const requestId = ++replHighlightSeq;
    replHighlightRequests.set(requestId, meta);
    worker.postMessage({type: 'highlight', code: code, target: 'repl', requestId: requestId});
    return requestId;
}

function handleReplHighlightResult(msg) {
    const meta = replHighlightRequests.get(msg.requestId);
    if (!meta) return;
    replHighlightRequests.delete(msg.requestId);

    if (meta.kind === 'live') {
        if (msg.requestId !== replLastLiveHighlightId) return;
        const htmlLines = renderHighlightedLines(meta.code, msg.highlights || []);
        setReplInputHighlight(htmlLines.join('<br>'));
        return;
    }

    if (meta.kind === 'block_partial') {
        if (msg.requestId !== replLastBlockHighlightId) return;
        const highlights = meta.suppressErrors
            ? (msg.highlights || []).filter((hl) => hl.group !== 'error')
            : (msg.highlights || []);
        const htmlLines = renderHighlightedLines(meta.code, highlights);
        for (let i = 0; i < meta.lineEls.length; i++) {
            const lineEl = meta.lineEls[i];
            if (!lineEl) continue;
            lineEl.innerHTML = htmlLines[i] || '';
        }
        return;
    }

    if (meta.kind === 'block') {
        const htmlLines = renderHighlightedLines(meta.code, msg.highlights || []);
        for (let i = 0; i < meta.lineEls.length; i++) {
            const lineEl = meta.lineEls[i];
            if (!lineEl) continue;
            lineEl.innerHTML = htmlLines[i] || '';
        }
    }
}

function scheduleReplLiveHighlight() {
    if (replHighlightTimer) {
        clearTimeout(replHighlightTimer);
    }
    replHighlightTimer = setTimeout(() => {
        replHighlightTimer = 0;
        if (!replMode) return;
        const code = getReplInputText();
        if (!code.trim()) {
            setReplInputHighlight('');
            return;
        }
        setReplInputHighlight(escapeHtml(code));
        if (code.trim().startsWith('/')) {
            replLastLiveHighlightId = 0;
            return;
        }
        const requestId = requestReplHighlight(code, {kind: 'live', code: code});
        if (requestId) {
            replLastLiveHighlightId = requestId;
        }
    }, 30);
}

function updateHighlights() {
    if (!shakarLoaded) {
        highlightEl.textContent = codeEl.value;
        return;
    }

    const code = codeEl.value;
    if (!code.trim()) {
        highlightEl.textContent = '';
        return;
    }
    if (!wasmHighlightActive && isLargeEditorDoc(code)) {
        const lines = code.split('\n');
        if (highlightEl.children.length !== lines.length) {
            highlightEl.innerHTML = lines
                .map((line) => `<div class="hl-line">${escapeHtml(line) || ' '}</div>`)
                .join('');
            editorLastLineCount = lines.length;
        }
        const range = getEditorVisibleRange(lines.length);
        const slice = lines.slice(range.start, range.end + 1).join('\n');
        const requestId = ++editorHighlightSeq;
        editorHighlightRequestId = requestId;
        editorHighlightRequestCode = code;
        editorHighlightRequestAt = performance.now();
        editorHighlightRequestStart = range.start;
        editorHighlightRequestEnd = range.end;
        editorHighlightRequestLines = lines;
        worker.postMessage({
            type: 'highlight_range',
            code: slice,
            startLine: range.start,
            requestId: requestId
        });
        return;
    }

    const requestId = ++editorHighlightSeq;
    editorHighlightRequestId = requestId;
    editorHighlightRequestCode = code;
    editorHighlightRequestAt = performance.now();
    worker.postMessage({type: 'highlight', code: code, requestId: requestId});
}

function scheduleHighlight() {
    clearError();

    if (highlightTimeout) {
        clearTimeout(highlightTimeout);
    }
    const delay = (!wasmHighlightActive && isLargeEditorDoc(codeEl.value))
        ? EDITOR_HIGHLIGHT_DELAY_LARGE
        : EDITOR_HIGHLIGHT_DELAY_SMALL;
    highlightTimeout = setTimeout(updateHighlights, delay);
}

function syncScroll() {
    highlightEl.scrollTop = codeEl.scrollTop;
    highlightEl.scrollLeft = codeEl.scrollLeft;
}

// --- Tetris ---

async function loadTetrisCode() {
    if (tetrisCode) return tetrisCode;
    try {
        const response = await fetch('tetris.sk?v=' + SHAKAR_VERSION);
        if (!response.ok) throw new Error('Failed to load tetris.sk');
        tetrisCode = await response.text();
        return tetrisCode;
    } catch (err) {
        console.error('Failed to load tetris:', err);
        return null;
    }
}

async function loadExhaustiveCode() {
    if (exhaustiveCode) return exhaustiveCode;
    try {
        const response = await fetch('exhaustive.sk?v=' + SHAKAR_VERSION);
        if (!response.ok) throw new Error('Failed to load exhaustive.sk');
        exhaustiveCode = await response.text();
        return exhaustiveCode;
    } catch (err) {
        console.error('Failed to load exhaustive:', err);
        return null;
    }
}

function isTetrisExample() {
    return examplesEl.value === 'tetris';
}

async function runCode() {
    if (!shakarLoaded) return;

    if (isTetrisExample()) {
        if (tetrisRunning) {
            stopTetris();
        } else {
            await startTetris();
        }
        return;
    }

    const code = codeEl.value;
    if (!code.trim()) {
        setOutput('');
        return;
    }

    runBtn.disabled = true;
    runBtn.classList.add('loading');
    setOutput('Running...');

    worker.postMessage({type: 'run', code: code, isTetris: false, debugPyTrace: DEBUG_PY_TRACE});
}

async function startTetris() {
    if (tetrisRunning) return;

    if (!supportsSharedArrayBuffer) {
        setOutput('Tetris requires cross-origin isolation (SharedArrayBuffer). Reload in a COOP/COEP-enabled context or use a browser that supports it.', true);
        return;
    }

    const code = await loadTetrisCode();
    if (!code) {
        setOutput('Failed to load tetris game', true);
        return;
    }

    tetrisRunning = true;
    runBtn.textContent = 'Stop';
    codeEl.blur();
    setOutput('Starting Tetris...');

    worker.postMessage({type: 'run', code: code, isTetris: true, debugPyTrace: DEBUG_PY_TRACE});
}

function stopTetris() {
    const wasRunning = tetrisRunning;
    tetrisRunning = false;

    if (wasRunning && shakarLoaded) {
        pushKey('q');
    }

    runBtn.textContent = 'Run';
}

// --- Worker setup ---

function initWorker() {
    setStatus('Loading Pyodide...');

    if (splashEl) {
        splashEl.classList.remove('splash-hidden');
    }

    const cacheV = window.SHAKAR_VERSION && window.SHAKAR_VERSION !== 'dev' ? window.SHAKAR_VERSION : Date.now();
    worker = new Worker('shakar_worker.js?v=' + cacheV);

    worker.onmessage = (e) => {
        const msg = e.data;
        if (DEBUG_IO && typeof msg?.type === 'string' && msg.type.startsWith('io_')) {
            const textLen = typeof msg.text === 'string' ? msg.text.length : 0;
            console.debug('[io]', msg.type, 'len=', textLen);
        }

        switch (msg.type) {
            case 'ready':
                shakarLoaded = true;
                wasmHighlightActive = msg.wasmHighlight === true;
                runBtn.disabled = false;
                setStatus('Ready', 'ready');
                if (splashEl) {
                    splashEl.classList.add('splash-hidden');
                    splashEl.setAttribute('aria-busy', 'false');
                }
                updateHighlights();
                break;

            case 'io_write':
                scheduleWrite(msg.text);
                break;

            case 'io_clear':
                scheduleClear();
                break;

            case 'io_overwrite':
                scheduleOverwrite(msg.text);
                break;

            case 'output':
                setOutput(
                    msg.text,
                    msg.isError,
                    msg.traceback || '',
                    msg.errorRange || null
                );
                if (msg.isError && !shakarLoaded) {
                    setStatus('Load error: ' + msg.text, 'error');
                    if (splashEl) {
                        splashEl.classList.remove('splash-hidden');
                        splashEl.setAttribute('aria-busy', 'false');
                    }
                }
                runBtn.disabled = false;
                runBtn.classList.remove('loading');
                break;

            case 'repl_output':
                replPending = false;
                setReplInputEnabled(true);
                replInput.focus();
                if (msg.text) {
                    appendReplOutput(msg.text, msg.isError);
                }
                if (msg.traceback) {
                    appendReplOutput(msg.traceback, true);
                }
                break;

            case 'running':
                if (!msg.value) {
                    // Execution finished
                    if (tetrisRunning) {
                        tetrisRunning = false;
                        runBtn.textContent = 'Run';
                    }
                    runBtn.disabled = false;
                    runBtn.classList.remove('loading');
                }
                break;

            case 'highlight_result': {
                if (msg.target === 'repl') {
                    handleReplHighlightResult(msg);
                    break;
                }
                if (msg.requestId !== editorHighlightRequestId) {
                    break;
                }
                if (editorHighlightRequestCode !== codeEl.value) {
                    scheduleHighlight();
                    break;
                }
                if (msg.startLine !== undefined) {
                    if (!editorHighlightRequestLines) {
                        break;
                    }
                    const suppressErrors = shouldSuppressEditorErrors();
                    applyHighlightsRange(
                        editorHighlightRequestLines,
                        msg.highlights || [],
                        editorHighlightRequestStart,
                        editorHighlightRequestEnd,
                        null,
                        null,
                        suppressErrors
                    );
                    syncScroll();
                    break;
                }
                let highlights = msg.highlights;
                if (currentError) {
                    highlights = highlights.concat([currentError]);
                }
                applyHighlights(codeEl.value, highlights);
                break;
            }
            case 'lex_probe_result':
                handleReplIndentProbeResult(msg);
                break;
        }
    };

    worker.onerror = (err) => {
        setStatus('Worker error: ' + err.message, 'error');
        if (splashEl) {
            splashEl.classList.remove('splash-hidden');
            splashEl.setAttribute('aria-busy', 'false');
        }
        console.error('Worker error:', err);
    };

    worker.postMessage({
        type: 'init',
        keyBuffer: keyBuffer,
        debugPyTrace: DEBUG_PY_TRACE,
        version: cacheV
    });
}

// Event listeners
runBtn.addEventListener('click', runCode);

clearBtn.addEventListener('click', () => {
    setOutput('');
});

replToggleBtn.addEventListener('click', () => {
    setReplMode(!replMode);
});

replResetBtn.addEventListener('click', () => {
    resetRepl();
});

replTranscript.addEventListener('click', () => {
    const selection = window.getSelection();
    if (selection && selection.type === 'Range' && selection.toString()) {
        return;
    }
    replInput.focus();
});

replBody.addEventListener('click', (e) => {
    if (e.target === replBody) {
        replInput.focus();
    }
});

replPane.addEventListener('click', (e) => {
    if (e.target === replPane) {
        replInput.focus();
    }
});

replInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === 'c') {
        e.preventDefault();
        handleReplInterrupt();
        return;
    }

    if (e.key === 'Tab') {
        if (replSuggestionsVisible()) {
            e.preventDefault();
            applyReplSuggestion();
            return;
        }
        e.preventDefault();
        const start = replInput.selectionStart;
        const end = replInput.selectionEnd;
        const value = replInput.value;
        replInput.value = value.slice(0, start) + '    ' + value.slice(end);
        replInput.selectionStart = replInput.selectionEnd = start + 4;
        updateReplSuggestions();
        setReplInputHighlight(escapeHtml(getReplInputText()));
        scheduleReplLiveHighlight();
        syncReplInputHeight();
        return;
    }

    if (e.key === 'ArrowUp') {
        if (replSuggestionsVisible()) {
            e.preventDefault();
            setReplSuggestionIndex(replSuggestionIndex - 1);
            return;
        }
        if (replHistory.length && !replContinuation && !replPending) {
            e.preventDefault();
            navigateReplHistory(-1);
            return;
        }
    }

    if (e.key === 'ArrowDown') {
        if (replSuggestionsVisible()) {
            e.preventDefault();
            setReplSuggestionIndex(replSuggestionIndex + 1);
            return;
        }
        if (replHistory.length && !replContinuation && !replPending) {
            e.preventDefault();
            navigateReplHistory(1);
            return;
        }
    }

    if (e.key === 'Escape') {
        if (replSuggestionsVisible()) {
            e.preventDefault();
            hideReplSuggestions();
            return;
        }
    }

    if (e.key === 'Enter' && e.shiftKey && shouldAllowReplManualMultiline()) {
        e.preventDefault();
        const start = replInput.selectionStart;
        const end = replInput.selectionEnd;
        const value = replInput.value;
        replInput.value = value.slice(0, start) + '\n' + value.slice(end);
        replInput.selectionStart = replInput.selectionEnd = start + 1;
        updateReplSuggestions();
        setReplInputHighlight(escapeHtml(getReplInputText()));
        scheduleReplLiveHighlight();
        syncReplInputHeight();
        return;
    }

    if (e.key === 'Enter') {
        if (replSuggestionsVisible()) {
            const trimmed = getReplInputText().trim();
            const selected = replSuggestionMatches[replSuggestionIndex] || replSuggestionMatches[0];
            if (selected && selected.cmd.startsWith(trimmed)) {
                e.preventDefault();
                applyReplSuggestion();
                return;
            }
        }
        e.preventDefault();
        handleReplEnter();
        return;
    }
});

replInput.addEventListener('input', () => {
    _armUnloadConfirm();
    const normalized = getReplInputText();
    if (normalized !== replInput.value) {
        setReplInputText(normalized);
        return;
    }
    replInput.scrollTop = 0;
    if (replHistoryIndex !== -1 && getReplInputText() !== replHistoryDisplay) {
        replHistoryIndex = -1;
        replHistoryValue = '';
        replHistoryDisplay = '';
    }
    updateReplSuggestions();
    setReplInputHighlight(escapeHtml(getReplInputText()));
    scheduleReplLiveHighlight();
    syncReplInputHeight();
});

examplesEl.addEventListener('change', async (e) => {
    stopTetris();

    if (e.target.value === 'tetris') {
        const code = await loadTetrisCode();
        if (code) {
            codeEl.value = code;
            _armUnloadConfirm();
            scheduleHighlight();
        }
    } else if (e.target.value === 'exhaustive') {
        const code = await loadExhaustiveCode();
        if (code) {
            codeEl.value = code;
            _armUnloadConfirm();
            scheduleHighlight();
        }
    } else {
        const example = EXAMPLES[e.target.value];
        if (example) {
            codeEl.value = example;
            _armUnloadConfirm();
            scheduleHighlight();
        }
    }
});

codeEl.addEventListener('input', () => {
    _armUnloadConfirm();
    editorLastInputAt = performance.now();
    if (editorErrorRevealTimer) {
        clearTimeout(editorErrorRevealTimer);
        editorErrorRevealTimer = 0;
    }
    editorLastHighlightCode = '';
    editorHighlightRequestLines = null;
    if (editorIndentAdjusting) {
        scheduleHighlight();
        return;
    }

    let value = codeEl.value;
    let pos = codeEl.selectionStart;
    const info = getLineInfo(value, pos);
    const lineStart = info.lineStart;

    if (lineStart !== editorOutdentLineStart) {
        const beforeCursor = value.slice(lineStart, pos);
        const match = beforeCursor.match(/^([ \t]+)(else|elif)(?:\b|:)\s*$/);
        if (match) {
            const indent = match[1];
            if (indent.length >= 4) {
                const lineEndIdx = value.indexOf('\n', lineStart);
                const lineEnd = lineEndIdx === -1 ? value.length : lineEndIdx;
                const line = value.slice(lineStart, lineEnd);
                const newIndent = indent.slice(0, indent.length - 4);
                const newLine = newIndent + line.slice(indent.length);
                const newValue =
                    value.slice(0, lineStart) + newLine + value.slice(lineEnd);
                const delta = indent.length - newIndent.length;
                editorIndentAdjusting = true;
                codeEl.value = newValue;
                codeEl.selectionStart = Math.max(lineStart + newIndent.length, pos - delta);
                codeEl.selectionEnd = codeEl.selectionStart;
                editorIndentAdjusting = false;
                editorOutdentLineStart = lineStart;
            }
        }
    }

    value = codeEl.value;
    pos = codeEl.selectionStart;
    if (!wasmHighlightActive && isLargeEditorDoc(value)) {
        applyEditorLightUpdate(value, pos);
    }
    scheduleHighlight();
});
codeEl.addEventListener('scroll', () => {
    syncScroll();
    if (!wasmHighlightActive) {
        refreshVisibleEditorHighlights();
    }
});

codeEl.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runCode();
        return;
    }

    if (e.key === 'Enter' && !e.shiftKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        const value = codeEl.value;
        const start = codeEl.selectionStart;
        const end = codeEl.selectionEnd;
        const lineStart = value.lastIndexOf('\n', start - 1) + 1;
        const lineEndIdx = value.indexOf('\n', start);
        const lineEnd = lineEndIdx === -1 ? value.length : lineEndIdx;
        const lineText = value.slice(lineStart, lineEnd);
        const baseIndent = getLineIndent(lineText);
        const needsBlockIndent = isBlockHeaderLine(lineText);
        const insert = '\n' + baseIndent + (needsBlockIndent ? '    ' : '');
        editorIndentAdjusting = true;
        codeEl.value = value.slice(0, start) + insert + value.slice(end);
        const newPos = start + insert.length;
        codeEl.selectionStart = newPos;
        codeEl.selectionEnd = newPos;
        editorIndentAdjusting = false;
        editorOutdentLineStart = -1;
        editorLastInputAt = performance.now();
        if (editorErrorRevealTimer) {
            clearTimeout(editorErrorRevealTimer);
            editorErrorRevealTimer = 0;
        }
        editorLastHighlightCode = '';
        editorHighlightRequestLines = null;
        if (!wasmHighlightActive) applyEditorLightUpdate(codeEl.value, newPos);
        scheduleHighlight();
        return;
    }

    if (e.key === 'Tab') {
        e.preventDefault();
        const start = codeEl.selectionStart;
        const end = codeEl.selectionEnd;
        codeEl.value = codeEl.value.substring(0, start) + '    ' + codeEl.value.substring(end);
        codeEl.selectionStart = codeEl.selectionEnd = start + 4;
        editorLastInputAt = performance.now();
        if (editorErrorRevealTimer) {
            clearTimeout(editorErrorRevealTimer);
            editorErrorRevealTimer = 0;
        }
        editorLastHighlightCode = '';
        editorHighlightRequestLines = null;
        if (!wasmHighlightActive) applyEditorLightUpdate(codeEl.value, codeEl.selectionStart);
        scheduleHighlight();
        return;
    }
});

// Global keyboard handler for game - pushes keys to worker
document.addEventListener('keydown', (e) => {
    if (!tetrisRunning) return;

    const keyMap = {
        'ArrowLeft': 'left',
        'ArrowRight': 'right',
        'ArrowDown': 'down',
        'ArrowUp': 'up',
        ' ': ' ',
        'q': 'q',
        'Q': 'q'
    };

    if (keyMap[e.key]) {
        e.preventDefault();
        pushKey(keyMap[e.key]);
    }
});

highlightEl.textContent = codeEl.value;

initWorker();

let _unloadArmed = false;
function _beforeUnloadConfirm(e) {
    e.preventDefault();
    e.returnValue = ' ';
    return ' ';
}

function _armUnloadConfirm() {
    if (_unloadArmed) return;
    _unloadArmed = true;
    window.onbeforeunload = _beforeUnloadConfirm;
    window.addEventListener('beforeunload', _beforeUnloadConfirm, {capture: true});
}
