// Shakar Web Worker — runs Pyodide off the main thread
// Uses runPython (sync) since blocking the worker is fine.
// Keys arrive via SharedArrayBuffer (postMessage can't reach us while runPython blocks).

importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');

let pyodide = null;
let debugPyTrace = false;

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
            debugPyTrace = msg.debugPyTrace === true;
            self.shk_debug_py_trace = debugPyTrace;
            await handleInit();
            break;
        case 'run':
            if (msg.debugPyTrace !== undefined) {
                self.shk_debug_py_trace = msg.debugPyTrace === true;
            }
            handleRun(msg.code, msg.isTetris);
            break;
        case 'repl_eval':
            if (msg.debugPyTrace !== undefined) {
                self.shk_debug_py_trace = msg.debugPyTrace === true;
            }
            handleReplEval(msg.code);
            break;
        case 'repl_reset':
            handleReplReset();
            break;
        case 'highlight':
            handleHighlight(msg.code, msg.target, msg.requestId);
            break;
        case 'lex_probe':
            handleLexProbe(msg.line, msg.requestId);
            break;
        case 'highlight_range':
            handleHighlightRange(msg.code, msg.startLine, msg.requestId);
            break;
    }
};

function runPythonWithGlobals(globals, source) {
    if (!pyodide) return null;
    const keys = Object.keys(globals);
    for (const key of keys) {
        pyodide.globals.set(key, globals[key]);
    }
    try {
        return pyodide.runPython(source);
    } finally {
        for (const key of keys) {
            pyodide.globals.delete(key);
        }
    }
}

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
        runPythonWithGlobals({_shk_src: code}, `
import sys
import traceback
import os
import js
from io import StringIO

_shk_stdout = StringIO()
_shk_stderr = StringIO()
_shk_traceback = None
sys.stdout = _shk_stdout
sys.stderr = _shk_stderr

if bool(getattr(js.self, "shk_debug_py_trace", False)):
    os.environ["SHAKAR_DEBUG_PY_TRACE"] = "1"
else:
    os.environ.pop("SHAKAR_DEBUG_PY_TRACE", None)

try:
    from shakar_ref.types import ShkNil as _ShkNil
    from shakar_ref.runner import _parse_and_lower, _last_is_stmt
    _shk_ast = _parse_and_lower(_shk_src)
    _shk_is_stmt = _last_is_stmt(_shk_ast)
    _shk_result = shk_run(_shk_src)
    if not _shk_is_stmt and not isinstance(_shk_result, _ShkNil):
        print(_shk_result)
    _shk_error = None
except Exception as e:
    _shk_error = str(e)
    if os.environ.get("SHAKAR_DEBUG_PY_TRACE"):
        _shk_traceback = traceback.format_exc()
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
`);

        const stdout = pyodide.runPython('_shk_stdout.getvalue()');
        const stderr = pyodide.runPython('_shk_stderr.getvalue()');
        const error = pyodide.runPython('_shk_error');
        const tracebackText = pyodide.runPython('_shk_traceback');
        const detailText = tracebackText || '';

        if (error) {
            self.postMessage({type: 'output', text: error, isError: true, traceback: detailText});
        } else {
            const output = (stdout + stderr).trim();
            self.postMessage({type: 'output', text: output || '(no output)', isError: false});
        }
    } catch (err) {
        self.postMessage({
            type: 'output',
            text: 'Error: ' + err.message,
            isError: true,
            traceback: err?.stack || String(err) || ''
        });
    } finally {
        self.postMessage({type: 'running', value: false});
    }
}

function handleReplEval(code) {
    if (!pyodide) return;

    try {
        runPythonWithGlobals({_shk_src: code}, `
import sys
import traceback
import os
import js
from io import StringIO
from shakar_ref.runtime import Frame, init_stdlib
from shakar_ref.types import ShkNil, ShakarRuntimeError
from shakar_ref.lexer_rd import LexError
from shakar_ref.parser_rd import ParseError
from shakar_ref.runner import repl_eval

_shk_stdout = StringIO()
_shk_stderr = StringIO()
_shk_repl_error = None
_shk_traceback = None
sys.stdout = _shk_stdout
sys.stderr = _shk_stderr

if bool(getattr(js.self, "shk_debug_py_trace", False)):
    os.environ["SHAKAR_DEBUG_PY_TRACE"] = "1"
else:
    os.environ.pop("SHAKAR_DEBUG_PY_TRACE", None)

if "_shk_repl_frame" not in globals() or _shk_repl_frame is None:
    init_stdlib()
    _shk_repl_frame = Frame(source="", source_path="<web-repl>")

try:
    _shk_result, _shk_is_stmt = repl_eval(_shk_src, _shk_repl_frame)
    if not _shk_is_stmt and not isinstance(_shk_result, ShkNil):
        print(_shk_result)
except (ParseError, LexError, ShakarRuntimeError) as exc:
    _shk_repl_error = f"Error: {exc}"
    if os.environ.get("SHAKAR_DEBUG_PY_TRACE"):
        _shk_traceback = traceback.format_exc()
except Exception as exc:
    _shk_repl_error = f"Error: {exc}"
    if os.environ.get("SHAKAR_DEBUG_PY_TRACE"):
        _shk_traceback = traceback.format_exc()
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
`);

        const stdout = pyodide.runPython('_shk_stdout.getvalue()');
        const stderr = pyodide.runPython('_shk_stderr.getvalue()');
        const error = pyodide.runPython('_shk_repl_error');
        const tracebackText = pyodide.runPython('_shk_traceback');
        let text = (stdout + stderr).replace(/\n+$/, '');

        if (error) {
            if (text) {
                text = text + '\n' + error;
            } else {
                text = error;
            }
            self.postMessage({type: 'repl_output', text: text, isError: true, traceback: tracebackText || ''});
        } else {
            self.postMessage({type: 'repl_output', text: text || '', isError: false});
        }
    } catch (err) {
        self.postMessage({
            type: 'repl_output',
            text: 'Error: ' + err.message,
            isError: true,
            traceback: err?.stack || String(err) || ''
        });
    }
}

function handleReplReset() {
    if (!pyodide) return;

    try {
        pyodide.runPython(`
from shakar_ref.runtime import Frame, init_stdlib
init_stdlib()
_shk_repl_frame = Frame(source="", source_path="<web-repl>")
`);
    } catch (err) {
        self.postMessage({
            type: 'repl_output',
            text: 'Error: ' + err.message,
            isError: true,
            traceback: err?.stack || String(err) || ''
        });
    }
}

function handleHighlight(code, target, requestId) {
    if (!pyodide) return;

    try {
        const resultJson = runPythonWithGlobals({_shk_src: code}, `
import json
json.dumps(shk_highlight(_shk_src))
`);
        const result = JSON.parse(resultJson);
        self.postMessage({
            type: 'highlight_result',
            highlights: result.highlights || [],
            target: target,
            requestId: requestId
        });
    } catch (err) {
        self.postMessage({
            type: 'highlight_result',
            highlights: [],
            target: target,
            requestId: requestId
        });
        console.error('Highlight error:', err);
    }
}

function handleLexProbe(line, requestId) {
    if (!pyodide) return;

    try {
        const resultJson = runPythonWithGlobals({_shk_line: line}, `
import json
from shakar_ref import lexer_rd
from shakar_ref.token_types import TT

def _is_block_header(line: str) -> bool:
    try:
        tokens = lexer_rd.tokenize(line, track_indentation=False)
    except Exception:
        return False

    depth = 0
    last_sig = None
    colon_count = 0
    saw_qmark = False

    for tok in tokens:
        t = tok.type
        if t in (TT.NEWLINE, TT.SEMI, TT.EOF):
            continue
        if t in (TT.LPAR, TT.LSQB, TT.LBRACE):
            depth += 1
        elif t in (TT.RPAR, TT.RSQB, TT.RBRACE):
            depth = max(0, depth - 1)
        if depth == 0:
            if t == TT.QMARK:
                saw_qmark = True
            if t == TT.COLON:
                colon_count += 1
            last_sig = t

    if depth != 0 or last_sig != TT.COLON:
        return False
    if saw_qmark and colon_count == 1:
        return False
    return True

json.dumps({"is_block_header": _is_block_header(_shk_line)})
`);
        const result = JSON.parse(resultJson);
        self.postMessage({
            type: 'lex_probe_result',
            isBlockHeader: result.is_block_header === true,
            requestId: requestId
        });
    } catch (err) {
        self.postMessage({
            type: 'lex_probe_result',
            isBlockHeader: false,
            requestId: requestId
        });
        console.error('Lex probe error:', err);
    }
}

function handleHighlightRange(code, startLine, requestId) {
    if (!pyodide) return;

    try {
        const resultJson = runPythonWithGlobals(
            {_shk_src: code, _shk_start_line: startLine},
            `
import json
from shakar_ref import lexer_rd
from highlight_server import _tokens_to_highlights, _scan_comments

def _range_highlight(src: str, line_offset: int) -> dict:
    tokens = lexer_rd.tokenize(src, track_indentation=False)
    highlights = _tokens_to_highlights(tokens)
    highlights.extend(_scan_comments(src))
    for hl in highlights:
        hl["line"] = hl.get("line", 0) + line_offset
    return {"highlights": highlights}

json.dumps(_range_highlight(_shk_src, _shk_start_line))
`
        );
        const result = JSON.parse(resultJson);
        self.postMessage({
            type: 'highlight_result',
            highlights: result.highlights || [],
            startLine: startLine,
            requestId: requestId
        });
    } catch (err) {
        self.postMessage({
            type: 'highlight_result',
            highlights: [],
            startLine: startLine,
            requestId: requestId
        });
        console.error('Highlight range error:', err);
    }
}
