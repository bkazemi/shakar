// Shakar Playground - Web Worker-based execution with syntax highlighting

const EXAMPLES = {
    hello: `print("Hello, Shakar!")`,
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
    tetris: null  // Special case - loaded from file
};

let worker = null;
let shakarLoaded = false;
let highlightTimeout = null;
let tetrisCode = null;
let tetrisRunning = false;

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
const codeEl = document.getElementById('code');
const highlightEl = document.getElementById('highlight-layer');
const outputEl = document.getElementById('output');
const runBtn = document.getElementById('run-btn');
const clearBtn = document.getElementById('clear-btn');
const examplesEl = document.getElementById('examples');

const DEBUG_IO = localStorage.getItem('shakar_debug_io') === '1';
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
}

function setOutput(text, isError = false) {
    outputEl.textContent = text;
    outputEl.className = isError ? 'error' : '';

    if (isError && text) {
        const errInfo = parseErrorLocation(text);
        if (errInfo) {
            highlightError(errInfo.line, errInfo.col);
        }
    }
}

function parseErrorLocation(errorText) {
    const match = errorText.match(/(?:at\s+)?line\s+(\d+),?\s*(?:col(?:umn)?\s*)?(\d+)?/i);
    if (match) {
        return {
            line: parseInt(match[1], 10) - 1,
            col: match[2] ? parseInt(match[2], 10) - 1 : 0
        };
    }
    return null;
}

let currentError = null;

function highlightError(line, col) {
    const code = codeEl.value;
    const lines = code.split('\n');

    if (line < 0 || line >= lines.length) return;

    const lineText = lines[line];
    const colEnd = Math.min(col + 10, lineText.length);
    currentError = {
        line: line,
        col_start: Math.max(0, col),
        col_end: Math.max(col + 1, colEnd),
        group: 'error'
    };

    updateHighlights();

    let pos = 0;
    for (let i = 0; i < line; i++) {
        pos += lines[i].length + 1;
    }
    pos += Math.min(col, lineText.length);

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

function applyHighlights(code, highlights) {
    if (!highlights || highlights.length === 0) {
        highlightEl.innerHTML = code.split('\n').map(line =>
            `<div class="hl-line">${escapeHtml(line) || ' '}</div>`
        ).join('');
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

        lineHls.sort((a, b) => a.col_start - b.col_start || b.col_end - a.col_end);

        const errors = lineHls.filter(h => h.group === 'error');
        const nonErrors = lineHls.filter(h => h.group !== 'error');
        const sorted = [...errors, ...nonErrors].sort((a, b) => a.col_start - b.col_start);

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

        const lineClass = hasError ? 'hl-line hl-line-error' : 'hl-line';
        result.push(`<div class="${lineClass}">${html || ' '}</div>`);
    }

    highlightEl.innerHTML = result.join('');
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

    worker.postMessage({type: 'highlight', code: code});
}

function scheduleHighlight() {
    clearError();

    if (highlightTimeout) {
        clearTimeout(highlightTimeout);
    }
    highlightTimeout = setTimeout(updateHighlights, 30);
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

    worker.postMessage({type: 'run', code: code, isTetris: false});
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

    worker.postMessage({type: 'run', code: code, isTetris: true});
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

    worker = new Worker('shakar_worker.js');

    worker.onmessage = (e) => {
        const msg = e.data;
        if (DEBUG_IO && typeof msg?.type === 'string' && msg.type.startsWith('io_')) {
            const textLen = typeof msg.text === 'string' ? msg.text.length : 0;
            console.debug('[io]', msg.type, 'len=', textLen);
        }

        switch (msg.type) {
            case 'ready':
                shakarLoaded = true;
                runBtn.disabled = false;
                setStatus('Ready', 'ready');
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
                setOutput(msg.text, msg.isError);
                if (!msg.isError) {
                    runBtn.disabled = false;
                    runBtn.classList.remove('loading');
                }
                if (msg.isError) {
                    runBtn.disabled = false;
                    runBtn.classList.remove('loading');
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
                let highlights = msg.highlights;
                if (currentError) {
                    highlights = highlights.concat([currentError]);
                }
                applyHighlights(codeEl.value, highlights);
                break;
            }
        }
    };

    worker.onerror = (err) => {
        setStatus('Worker error: ' + err.message, 'error');
        console.error('Worker error:', err);
    };

    worker.postMessage({type: 'init', keyBuffer: keyBuffer});
}

// Event listeners
runBtn.addEventListener('click', runCode);

clearBtn.addEventListener('click', () => {
    setOutput('');
});

examplesEl.addEventListener('change', async (e) => {
    stopTetris();

    if (e.target.value === 'tetris') {
        const code = await loadTetrisCode();
        if (code) {
            codeEl.value = code;
            scheduleHighlight();
        }
    } else {
        const example = EXAMPLES[e.target.value];
        if (example) {
            codeEl.value = example;
            scheduleHighlight();
        }
    }
});

codeEl.addEventListener('input', scheduleHighlight);
codeEl.addEventListener('scroll', syncScroll);

codeEl.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runCode();
    }

    if (e.key === 'Tab') {
        e.preventDefault();
        const start = codeEl.selectionStart;
        const end = codeEl.selectionEnd;
        codeEl.value = codeEl.value.substring(0, start) + '    ' + codeEl.value.substring(end);
        codeEl.selectionStart = codeEl.selectionEnd = start + 4;
        scheduleHighlight();
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
