// Shakar Web Worker — runs Pyodide off the main thread
// Uses runPython (sync) since blocking the worker is fine.
// Keys arrive via SharedArrayBuffer (postMessage can't reach us while runPython blocks).

importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');

let pyodide = null;
let debugPyTrace = false;

// WASM highlighter (loaded alongside Pyodide)
let wasmHL = null;
let wasmApi = null;
let cacheV = Date.now();

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
            cacheV = msg.version || Date.now();
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

// Group name table (must match HL_ enum in highlight.h)
const HL_GROUP_NAMES = [
    '',            // HL_NONE
    'keyword', 'boolean', 'constant', 'number', 'unit',
    'string', 'regex', 'path', 'comment', 'operator',
    'punctuation', 'identifier', 'function', 'decorator',
    'hook', 'property', 'implicit_subject', 'type',
];

async function initWasmHL() {
    try {
        importScripts('shakar_hl_glue.js?v=' + cacheV);
        wasmHL = await ShakarHL({
            locateFile: (path) => {
                if (path.endsWith('.wasm')) return 'shakar_hl_glue.wasm?v=' + cacheV;
                return path;
            }
        });
        wasmApi = {
            srcPtr:      wasmHL.cwrap('shk_src_ptr', 'number', []),
            setSrcLen:   wasmHL.cwrap('shk_set_src_len', null, ['number']),
            highlight:   wasmHL.cwrap('shk_highlight', 'number', []),
            hlSpansPtr:  wasmHL.cwrap('shk_hl_spans_ptr', 'number', []),
            hlCount:     wasmHL.cwrap('shk_hl_count', 'number', []),
            errorLine:   wasmHL.cwrap('shk_error_line', 'number', []),
            errorCol:    wasmHL.cwrap('shk_error_col', 'number', []),
            errorPos:    wasmHL.cwrap('shk_error_pos', 'number', []),
            diagnostics: wasmHL.cwrap('shk_diagnostics', 'number', []),
            diagCount:   wasmHL.cwrap('shk_diag_count', 'number', []),
            diagLine:    wasmHL.cwrap('shk_diag_line', 'number', ['number']),
            diagColStart:wasmHL.cwrap('shk_diag_col_start', 'number', ['number']),
            diagColEnd:  wasmHL.cwrap('shk_diag_col_end', 'number', ['number']),
            diagSeverity:wasmHL.cwrap('shk_diag_severity', 'number', ['number']),
            diagMessage: wasmHL.cwrap('shk_diag_message', 'string', ['number']),
        };
    } catch (err) {
        console.warn('WASM highlighter not available, highlights disabled:', err.message);
        wasmApi = null;
    }
}

function wasmHighlight(source) {
    if (!wasmApi) return null;

    try {
        const MAX_SRC = 1024 * 1024;  // must match api.c MAX_SRC
        const ptr = wasmApi.srcPtr();
        const len = wasmHL.lengthBytesUTF8(source);
        if (len >= MAX_SRC) return null;  // too large — skip highlighting
        wasmHL.stringToUTF8(source, ptr, len + 1);
        wasmApi.setSrcLen(len);

        const count = wasmApi.highlight();
        if (count < 0) {
            // Lex error: synthesize a single error span from WASM error coordinates.
            const line = wasmApi.errorLine ? wasmApi.errorLine() : 0;
            const col = wasmApi.errorCol ? wasmApi.errorCol() : 0;
            const pos = wasmApi.errorPos ? wasmApi.errorPos() : -1;
            let lineIdxResolved = -1;
            let colStartResolved = -1;

            if (line > 0 && col > 0) {
                lineIdxResolved = line - 1;
                colStartResolved = col - 1;
            } else if (pos >= 0) {
                // Fall back to byte offset when line/col are unavailable.
                const before = source.slice(0, pos);
                lineIdxResolved = (before.match(/\n/g) || []).length;
                const lastNl = before.lastIndexOf('\n');
                colStartResolved = pos - (lastNl + 1);
            }

            if (lineIdxResolved < 0 || colStartResolved < 0) {
                // Final fallback: mark the first line if we can't resolve coordinates.
                lineIdxResolved = 0;
                colStartResolved = 0;
            }

            // Clamp to actual line length (or 1 for empty lines) to place a 1-char error span.
            let lineStart = 0;
            let idx = 0;
            let remaining = lineIdxResolved;
            while (idx < source.length && remaining > 0) {
                const nl = source.indexOf('\n', idx);
                if (nl < 0) break;
                lineStart = nl + 1;
                idx = lineStart;
                remaining--;
            }
            let lineEnd = source.indexOf('\n', lineStart);
            if (lineEnd < 0) lineEnd = source.length;
            let lineLen = lineEnd - lineStart;
            if (lineLen > 0 && source[lineEnd - 1] === '\r') {
                lineLen -= 1;
            }
            // Highlight the full line so errors are visible even on empty/short lines.
            const safeLen = Math.max(1, lineLen);
            const clampedStart = Math.min(Math.max(0, colStartResolved), safeLen - 1);
            return [{
                line: lineIdxResolved,
                col_start: clampedStart,
                col_end: Math.min(clampedStart + 1, safeLen),
                group: 'error',
            }];
        }

        // Read HlSpan array directly from WASM memory.
        // HlSpan is { int line, int col_start, int col_end, int group } = 4 ints = 16 bytes.
        const spansPtr = wasmApi.hlSpansPtr();
        const heap = new Int32Array(wasmHL.HEAP32.buffer, spansPtr, count * 4);
        const spans = new Array(count);
        for (let i = 0; i < count; i++) {
            const base = i * 4;
            spans[i] = {
                line:      heap[base],
                col_start: heap[base + 1],
                col_end:   heap[base + 2],
                group:     HL_GROUP_NAMES[heap[base + 3]] || '',
            };
        }

        // Run structural diagnostics and append error spans.
        const diagCount = wasmApi.diagnostics();
        for (let i = 0; i < diagCount; i++) {
            spans.push({
                line:      wasmApi.diagLine(i),
                col_start: wasmApi.diagColStart(i),
                col_end:   wasmApi.diagColEnd(i),
                group:     wasmApi.diagSeverity(i) === 1 ? 'error' : 'warning',
            });
        }

        return spans;
    } catch (err) {
        console.error('WASM highlight error:', err);
        return null;
    }
}

async function handleInit() {
    try {
        // Load WASM highlighter and Pyodide in parallel
        const wasmPromise = initWasmHL();
        const pyodidePromise = (async () => {
            pyodide = await loadPyodide({enableRunUntilComplete: false});
            await pyodide.loadPackage('typing-extensions');

            const response = await fetch('shakar_bundle.py?v=' + cacheV);
            if (!response.ok) throw new Error('Failed to load shakar_bundle.py');
            const bundleCode = await response.text();
            pyodide.runPython(bundleCode);
        })();

        await Promise.all([wasmPromise, pyodidePromise]);

        self.postMessage({type: 'ready', wasmHighlight: wasmApi !== null});
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
import json
from io import StringIO

def _range_from_coords(line, col, end_line, end_col):
    if not isinstance(line, int) or line <= 0:
        return None
    if not isinstance(col, int) or col <= 0:
        col = 1
    start = col - 1

    if isinstance(end_line, int) and end_line == line and isinstance(end_col, int) and end_col > col:
        end = max(start + 1, end_col - 1)
    else:
        # The web overlay renderer applies spans per line. For multi-line
        # diagnostics, keep a minimal anchor on the start line.
        end = start + 1

    return {"line": line - 1, "col_start": start, "col_end": end}

_shk_stdout = StringIO()
_shk_stderr = StringIO()
_shk_traceback = None
_shk_error_range_json = "null"
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
    _shk_error_range = None

    try:
        from shakar_ref.parser_rd import ParseError as _ParseError
        from shakar_ref.lexer_rd import LexError as _LexError
    except Exception:
        _ParseError = None
        _LexError = None

    if _ParseError is not None and isinstance(e, _ParseError):
        _shk_error_range = _range_from_coords(
            getattr(e, "line", None),
            getattr(e, "column", None),
            getattr(e, "end_line", None),
            getattr(e, "end_column", None),
        )

    if _shk_error_range is None and _LexError is not None and isinstance(e, _LexError):
        _shk_error_range = _range_from_coords(
            getattr(e, "line", None),
            getattr(e, "column", None),
            getattr(e, "end_line", None),
            getattr(e, "end_column", None),
        )

    if _shk_error_range is None:
        meta = getattr(e, "shk_meta", None)
        _shk_error_range = _range_from_coords(
            getattr(meta, "line", None) if meta is not None else None,
            getattr(meta, "column", None) if meta is not None else None,
            getattr(meta, "end_line", None) if meta is not None else None,
            getattr(meta, "end_column", None) if meta is not None else None,
        )

    _shk_error_range_json = json.dumps(_shk_error_range)
    if os.environ.get("SHAKAR_DEBUG_PY_TRACE"):
        _shk_traceback = traceback.format_exc()
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
`);

        const stdout = pyodide.runPython('_shk_stdout.getvalue()');
        const stderr = pyodide.runPython('_shk_stderr.getvalue()');
        const error = pyodide.runPython('_shk_error');
        const errorRangeJson = pyodide.runPython('_shk_error_range_json');
        const tracebackText = pyodide.runPython('_shk_traceback');
        const detailText = tracebackText || '';
        let errorRange = null;
        if (errorRangeJson) {
            try {
                errorRange = JSON.parse(errorRangeJson);
            } catch (_err) {
                errorRange = null;
            }
        }

        if (error) {
            self.postMessage({
                type: 'output',
                text: error,
                isError: true,
                traceback: detailText,
                errorRange: errorRange
            });
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
            text = text ? text + '\n' + error : error;
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
    self.postMessage({
        type: 'highlight_result',
        highlights: wasmHighlight(code) || [],
        target: target,
        requestId: requestId
    });
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
    const wasmResult = wasmHighlight(code);
    // Offset span lines to match the document position within the visible range.
    const highlights = wasmResult
        ? wasmResult.map(s => ({...s, line: s.line + startLine}))
        : [];

    self.postMessage({
        type: 'highlight_result',
        highlights: highlights,
        startLine: startLine,
        requestId: requestId
    });
}
