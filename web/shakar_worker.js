// Shakar Web Worker — runs Pyodide off the main thread
// Uses runPython (sync) since blocking the worker is fine.
// Keys arrive via SharedArrayBuffer (postMessage can't reach us while runPython blocks).

importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');

let pyodide = null;

// SharedArrayBuffer for key input: Int32Array[0]=write_idx, [1..32]=key ring buffer
// Allocated by main thread, passed to us via 'init' message.
let keyBuf = null;

// Functions called from Python's stdlib (js.self.shk_io_write etc.)
self.shk_io_write = (text) => {
    self.postMessage({type: 'io_write', text: text});
};

self.shk_io_clear = () => {
    self.postMessage({type: 'io_clear'});
};

self.shk_io_overwrite = (text) => {
    self.postMessage({type: 'io_overwrite', text: text});
};

// Expose keyBuf on self so Python can access it via js.self.shk_key_buf
self.shk_key_buf = null;

self.onmessage = async (e) => {
    const msg = e.data;

    switch (msg.type) {
        case 'init':
            keyBuf = msg.keyBuffer ? new Int32Array(msg.keyBuffer) : null;
            self.shk_key_buf = keyBuf;
            await handleInit();
            break;
        case 'run':
            handleRun(msg.code, msg.isTetris);
            break;
        case 'highlight':
            handleHighlight(msg.code);
            break;
    }
};

async function handleInit() {
    try {
        pyodide = await loadPyodide({enableRunUntilComplete: false});
        await pyodide.loadPackage('typing-extensions');

        const response = await fetch('shakar_bundle.py?v=' + Date.now());
        if (!response.ok) throw new Error('Failed to load shakar_bundle.py');
        const bundleCode = await response.text();
        pyodide.runPython(bundleCode);

        self.postMessage({type: 'ready'});
    } catch (err) {
        self.postMessage({type: 'output', text: 'Failed to load: ' + err.message, isError: true});
    }
}

function handleRun(code, isTetris) {
    if (!pyodide) return;

    self.postMessage({type: 'running', value: true});

    if (isTetris) {
        if (!keyBuf) {
            self.postMessage({type: 'output', text: 'Tetris requires SharedArrayBuffer support.', isError: true});
            self.postMessage({type: 'running', value: false});
            return;
        }
        // Reset read index to match write index (clear stale keys)
        pyodide.runPython(`
from shakar_ref.stdlib import _io_key_read_idx
import js
_io_key_read_idx[0] = int(js.Atomics.load(js.self.shk_key_buf, 0)) & 0xFFFFFFFF
`);
    }

    try {
        const escaped = code.replace(/\\/g, '\\\\').replace(/"""/g, '\\"\\"\\"');
        pyodide.runPython(`
import sys
from io import StringIO

_shk_stdout = StringIO()
_shk_stderr = StringIO()
sys.stdout = _shk_stdout
sys.stderr = _shk_stderr

try:
    _shk_result = shk_run("""${escaped}""")
    if _shk_result is not None:
        print(_shk_result)
    _shk_error = None
except Exception as e:
    _shk_error = str(e)
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
`);

        const stdout = pyodide.runPython('_shk_stdout.getvalue()');
        const stderr = pyodide.runPython('_shk_stderr.getvalue()');
        const error = pyodide.runPython('_shk_error');

        if (error) {
            self.postMessage({type: 'output', text: error, isError: true});
        } else {
            const output = (stdout + stderr).trim();
            self.postMessage({type: 'output', text: output || '(no output)', isError: false});
        }
    } catch (err) {
        self.postMessage({type: 'output', text: 'Error: ' + err.message, isError: true});
    } finally {
        self.postMessage({type: 'running', value: false});
    }
}

function handleHighlight(code) {
    if (!pyodide) return;

    try {
        const escaped = code.replace(/\\/g, '\\\\').replace(/'''/g, "\\'\\'\\'");
        const resultJson = pyodide.runPython(`
import json
json.dumps(shk_highlight('''${escaped}'''))
`);
        const result = JSON.parse(resultJson);
        self.postMessage({type: 'highlight_result', highlights: result.highlights || []});
    } catch (err) {
        self.postMessage({type: 'highlight_result', highlights: []});
        console.error('Highlight error:', err);
    }
}
