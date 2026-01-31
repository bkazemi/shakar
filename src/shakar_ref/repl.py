from __future__ import annotations

import ctypes
import ctypes.util
import sys
from typing import Optional

from .evaluator import eval_expr
from .lexer_rd import LexError
from .parser_rd import ParseError
from .runtime import Frame, init_stdlib
from .types import ShkNil, ShakarRuntimeError
from .runner import _parse_and_lower, _last_is_stmt

# ---------------------------------------------------------------------------
# Ctrl-Up/Down line history via ctypes
# ---------------------------------------------------------------------------
_RLCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)

try:
    import readline as _readline

    _HAVE_READLINE = True
except ModuleNotFoundError:
    _readline = None
    _HAVE_READLINE = False

_rl_lib: Optional[ctypes.CDLL] = None
_rl_name = ctypes.util.find_library("readline")
if _rl_name:
    try:
        _rl_lib = ctypes.CDLL(_rl_name)
    except OSError:
        pass

_line_hist: list[str] = []
_line_idx: int = -1


def _rl_replace_line(text: str) -> None:
    if _rl_lib is None:
        return
    encoded = text.encode("utf-8")
    try:
        _rl_lib.rl_replace_line(encoded, 0)
        for sym in ("rl_end", "rl_point"):
            try:
                ctypes.c_int.in_dll(_rl_lib, sym).value = len(encoded)
            except ValueError:
                pass
        _rl_lib.rl_redisplay()
    except AttributeError:
        pass


@_RLCB
def _ctrl_up(_count: int, _key: int) -> int:
    global _line_idx
    if _line_hist:
        _line_idx = min(_line_idx + 1, len(_line_hist) - 1)
        _rl_replace_line(_line_hist[-(1 + _line_idx)])

    return 0


@_RLCB
def _ctrl_down(_count: int, _key: int) -> int:
    global _line_idx
    if _line_idx > 0:
        _line_idx -= 1
        _rl_replace_line(_line_hist[-(1 + _line_idx)])
    elif _line_idx == 0:
        _line_idx = -1
        _rl_replace_line("")

    return 0


def _bind_line_history_keys() -> None:
    if _rl_lib is None or not _HAVE_READLINE:
        return
    try:
        _rl_lib.rl_bind_keyseq(b"\x1b[1;5A", _ctrl_up)
        _rl_lib.rl_bind_keyseq(b"\x1b[1;5B", _ctrl_down)
    except (OSError, AttributeError):
        pass


def _repl_history_replace(n_lines: int, combined: str) -> None:
    """Replace the last *n_lines* readline entries with one multiline entry."""
    if not _HAVE_READLINE:
        return
    for _ in range(n_lines):
        end = _readline.get_current_history_length()
        if end > 0:
            _readline.remove_history_item(end - 1)

    _readline.add_history(combined)


def _auto_history_enabled() -> bool:
    if not _HAVE_READLINE:
        return False
    getter = getattr(_readline, "get_auto_history", None)
    if getter is None:
        return True
    try:
        return bool(getter())
    except Exception:
        return True


def repl() -> None:
    """Interactive read-eval-print loop.

    Up recalls entire multiline blocks (with indentation).
    Ctrl-Up/Down cycle through individual dedented lines.
    """
    global _line_idx

    init_stdlib()
    frame = Frame(source="", source_path="<repl>")
    if _HAVE_READLINE:
        setter = getattr(_readline, "set_auto_history", None)
        if setter is not None:
            setter(True)
    _bind_line_history_keys()
    print("shakar repl — Ctrl-D to exit")

    while True:
        _line_idx = -1

        try:
            line = input(">>> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
            continue

        # Collect continuation lines for indented blocks.
        lines = [line]
        if line.endswith(":") or line.endswith("\\"):
            aborted = False
            while True:
                try:
                    cont = input("... ")
                except (EOFError, KeyboardInterrupt) as exc:
                    if isinstance(exc, EOFError):
                        print()
                    else:
                        print("\nKeyboardInterrupt")
                    aborted = True
                    break
                if cont == "":
                    break
                lines.append(cont)
            if aborted:
                continue

        src = "\n".join(lines)
        if not src.strip():
            continue

        auto_history = _auto_history_enabled()
        if _HAVE_READLINE:
            if len(lines) == 1:
                if not auto_history:
                    _readline.add_history(src)
            elif auto_history:
                _repl_history_replace(len(lines), src)
            else:
                _readline.add_history(src)

        # Record dedented lines for Ctrl-Up navigation.
        for l in lines:
            stripped = l.strip()
            if stripped and (not _line_hist or _line_hist[-1] != stripped):
                _line_hist.append(stripped)

        try:
            ast2 = _parse_and_lower(src)
            stmt = _last_is_stmt(ast2)
            result = eval_expr(ast2, frame, source=src)
        except (ParseError, LexError, ShakarRuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        if not stmt and not isinstance(result, ShkNil):
            print(result)
