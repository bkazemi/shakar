// Shakar Playground - Pyodide-based execution with syntax highlighting

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

let pyodide = null;
let shakarLoaded = false;
let highlightTimeout = null;
let tetrisCode = null;
let tetrisRunning = false;

const statusEl = document.getElementById('status');
const codeEl = document.getElementById('code');
const highlightEl = document.getElementById('highlight-layer');
const outputEl = document.getElementById('output');
const runBtn = document.getElementById('run-btn');
const clearBtn = document.getElementById('clear-btn');
const examplesEl = document.getElementById('examples');

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

async function updateHighlights() {
    if (!shakarLoaded) {
        highlightEl.textContent = codeEl.value;
        return;
    }

    const code = codeEl.value;
    if (!code.trim()) {
        highlightEl.textContent = '';
        return;
    }

    try {
        const escaped = code.replace(/\\/g, '\\\\').replace(/'''/g, "\\'\\'\\'");
        const resultJson = await pyodide.runPythonAsync(`
import json
json.dumps(shk_highlight('''${escaped}'''))
`);
        const result = JSON.parse(resultJson);
        let highlights = result.highlights || [];

        if (currentError) {
            highlights = highlights.concat([currentError]);
        }

        applyHighlights(code, highlights);
    } catch (err) {
        highlightEl.textContent = code;
        console.error('Highlight error:', err);
    }
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

// --- io module JavaScript interface ---

// DOM output for io.write()
window.shk_io_write = (text) => {
    outputEl.textContent = text;
};

// DOM clear for io.clear()
window.shk_io_clear = () => {
    outputEl.textContent = '';
};

// Push key to Python io key queue (direct access avoids runPython blocking)
let ioKeyQueue = null;

function pushKeyToQueue(key) {
    if (!shakarLoaded) return;
    try {
        if (!ioKeyQueue) {
            const stdlib = pyodide.pyimport('shakar_ref.stdlib');
            ioKeyQueue = stdlib._io_key_queue;
        }
        ioKeyQueue.append(key);
    } catch (e) {
        console.error('pushKeyToQueue error:', e);
    }
}

// --- Tetris ---

async function loadTetrisCode() {
    if (tetrisCode) return tetrisCode;
    try {
        const response = await fetch('tetris.sk');
        if (!response.ok) throw new Error('Failed to load tetris.sk');
        tetrisCode = await response.text();
        return tetrisCode;
    } catch (err) {
        console.error('Failed to load tetris:', err);
        return null;
    }
}

async function startTetris() {
    if (tetrisRunning) return;

    const code = await loadTetrisCode();
    if (!code) {
        setOutput('Failed to load tetris game', true);
        return;
    }

    tetrisRunning = true;
    runBtn.textContent = 'Stop';

    // Blur textarea so arrow keys don't get captured by it
    codeEl.blur();

    setOutput('Starting Tetris...');

    // Clear any stale keys from previous run
    await pyodide.runPythonAsync(`
from shakar_ref.stdlib import _io_key_queue
_io_key_queue.clear()
`);

    try {
        const escaped = code.replace(/\\/g, '\\\\').replace(/"""/g, '\\"\\"\\"');
        await pyodide.runPythonAsync(`
import sys
from io import StringIO

_shk_stdout = StringIO()
sys.stdout = _shk_stdout

try:
    shk_run("""${escaped}""")
except Exception as e:
    print(f"Error: {e}")
finally:
    sys.stdout = sys.__stdout__
`);
    } catch (err) {
        setOutput('Tetris error: ' + err.message, true);
    } finally {
        stopTetris();
    }
}

function stopTetris() {
    const wasRunning = tetrisRunning;
    tetrisRunning = false;

    // Push 'q' to exit game loop if it was actually running
    if (wasRunning && shakarLoaded) {
        pushKeyToQueue('q');
    }

    runBtn.textContent = 'Run';
}

function isTetrisExample() {
    return examplesEl.value === 'tetris';
}

async function initPyodide() {
    try {
        setStatus('Loading Pyodide...');
        pyodide = await loadPyodide();

        setStatus('Loading dependencies...');
        await pyodide.loadPackage('typing-extensions');

        setStatus('Loading Shakar...');
        await loadShakar();

        shakarLoaded = true;
        runBtn.disabled = false;
        setStatus('Ready', 'ready');

        updateHighlights();
    } catch (err) {
        setStatus('Failed to load: ' + err.message, 'error');
        console.error(err);
    }
}

async function loadShakar() {
    const response = await fetch('shakar_bundle.py');
    if (!response.ok) {
        throw new Error('Failed to load shakar_bundle.py');
    }
    const bundleCode = await response.text();
    await pyodide.runPythonAsync(bundleCode);

    // Cache the key queue reference for keyboard events
    await pyodide.runPythonAsync(`import shakar_ref.stdlib`);
    const stdlib = pyodide.pyimport('shakar_ref.stdlib');
    ioKeyQueue = stdlib._io_key_queue;
}

async function runCode() {
    if (!shakarLoaded) return;

    // Handle tetris specially - it needs interactive keyboard input
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

    try {
        await pyodide.runPythonAsync(`
import sys
from io import StringIO

_shk_stdout = StringIO()
_shk_stderr = StringIO()
sys.stdout = _shk_stdout
sys.stderr = _shk_stderr
`);

        const escaped = code.replace(/\\/g, '\\\\').replace(/"""/g, '\\"\\"\\"');
        await pyodide.runPythonAsync(`
try:
    _shk_result = shk_run("""${escaped}""")
    if _shk_result is not None:
        print(_shk_result)
    _shk_error = None
except Exception as e:
    _shk_error = str(e)
`);

        const stdout = await pyodide.runPythonAsync('_shk_stdout.getvalue()');
        const stderr = await pyodide.runPythonAsync('_shk_stderr.getvalue()');
        const error = await pyodide.runPythonAsync('_shk_error');

        await pyodide.runPythonAsync(`
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
`);

        if (error) {
            setOutput(error, true);
        } else {
            const output = (stdout + stderr).trim();
            setOutput(output || '(no output)');
        }
    } catch (err) {
        setOutput('Error: ' + err.message, true);
    } finally {
        runBtn.disabled = false;
        runBtn.classList.remove('loading');
    }
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

// Global keyboard handler for game - pushes to io key queue
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
        pushKeyToQueue(keyMap[e.key]);
    }
});

highlightEl.textContent = codeEl.value;

initPyodide();
