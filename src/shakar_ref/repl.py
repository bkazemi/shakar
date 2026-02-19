"""Interactive REPL for Shakar, powered by prompt_toolkit."""

from __future__ import annotations

import os
import re
import sys
import traceback
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import clear

from .lexer_rd import LexError, tokenize
from .parser_rd import ParseError
from .repl_highlight import ShakarLexer
from .runtime import Frame, init_stdlib
from .runner import repl_eval
from .token_types import TT
from .types import ShkNil, ShakarRuntimeError
from .utils import debug_py_trace_enabled

# Zero-width and invisible characters to strip from input.
_INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\ufeff\u00a0\r]")

# Slash commands: name => (description, argument_hint).
_SLASH_CMDS = {
    "/clear": ("Clear the terminal screen", ""),
    "/py-traceback": ("Toggle Python traceback on errors", "[on|off]"),
    "/reset": ("Reset the REPL environment", ""),
}

# Block-header tokens: last significant token at depth 0 is COLON. If we saw a
# depth-0 QMARK, require at least two COLONs (ternary + block header).
_DEPTH_OPEN = {TT.LPAR, TT.LSQB, TT.LBRACE}
_DEPTH_CLOSE = {TT.RPAR, TT.RSQB, TT.RBRACE}
_LAYOUT = {TT.NEWLINE, TT.INDENT, TT.DEDENT, TT.EOF, TT.COMMENT}


def _is_block_header(line: str) -> bool:
    """Return True if *line* ends a block header (colon at depth 0)."""
    try:
        tokens = tokenize(line)
    except LexError:
        return False

    depth = 0
    last_sig = None
    colon_count = 0
    saw_qmark = False

    for tok in tokens:
        t = tok.type
        if t in _LAYOUT or t == TT.SEMI:
            continue
        if t in _DEPTH_OPEN:
            depth += 1
        elif t in _DEPTH_CLOSE:
            depth = max(depth - 1, 0)
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


class _SlashCompleter(Completer):
    """Autocomplete slash commands on the primary prompt."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        for cmd, (desc, hint) in _SLASH_CMDS.items():
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display_meta=desc,
                )


def _handle_slash(line: str, frame_box: list[Frame]) -> bool:
    """Handle slash commands. Returns True if the line was a command."""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return False

    parts = stripped.split(None, 1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/clear":
        clear()
        return True

    if cmd == "/py-traceback":
        if arg.lower() in ("on", "1", "true", "yes"):
            os.environ["SHAKAR_DEBUG_PY_TRACE"] = "1"
        elif arg.lower() in ("off", "0", "false", "no"):
            os.environ.pop("SHAKAR_DEBUG_PY_TRACE", None)
        elif arg == "":
            # Toggle.
            if debug_py_trace_enabled():
                os.environ.pop("SHAKAR_DEBUG_PY_TRACE", None)
            else:
                os.environ["SHAKAR_DEBUG_PY_TRACE"] = "1"
        else:
            print(f"Usage: /py-traceback [on|off]", file=sys.stderr)
            return True

        state = "on" if debug_py_trace_enabled() else "off"
        print(f"Python traceback: {state}")
        return True

    if cmd == "/reset":
        from .runtime import _STATIC_ONCE_CELLS

        init_stdlib()
        _STATIC_ONCE_CELLS.clear()
        frame_box[0] = Frame(source="", source_path="<repl>")
        print("Environment reset.")
        return True

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return True


def _normalize(text: str) -> str:
    """Strip invisible characters from input."""
    return _INVISIBLE_RE.sub("", text)


def _awaits_catch(text: str) -> bool:
    """Return True if *text* is a ``try:`` block with no base-level content yet.

    Empty lines are allowed indefinitely (like Python).  Once the user types
    any non-empty content at the base indentation level (catch or otherwise),
    returns False so the next empty line submits normally.
    """
    lines = text.split("\n")
    first = lines[0] if lines else ""
    try_indent = len(first) - len(first.lstrip())
    if first.strip() != "try:":
        return False

    # If a catch clause already exists at the try indentation, no need to wait.
    # If any line at the base level already appeared after the body, the user
    # had their chance — let submission proceed.
    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            continue

        indent = len(line) - len(line.lstrip())
        if indent <= try_indent:
            # Non-empty content at the base level.  If it's a catch clause
            # the try is complete; otherwise the user typed something wrong
            # and the next empty line should submit so the parser can error.
            return False

    return True


def _compute_indent(text: str) -> str:
    """Compute the auto-indent prefix for the next continuation line."""
    lines = text.split("\n")
    last = lines[-1]

    if _is_block_header(last):
        existing = len(last) - len(last.lstrip())
        return " " * (existing + 4)

    # Preserve indent of the last line.
    if last.strip():
        return " " * (len(last) - len(last.lstrip()))

    return ""


def repl() -> None:
    """Interactive read-eval-print loop with prompt_toolkit."""
    init_stdlib()
    frame = Frame(source="", source_path="<repl>")
    # Use a mutable box so /reset can swap the frame.
    frame_box: list[Frame] = [frame]

    history = InMemoryHistory()
    lexer = ShakarLexer()

    bindings = KeyBindings()

    @bindings.add("backspace")
    def _backspace(event):
        buf = event.app.current_buffer
        buf.delete_before_cursor(1)
        if buf.text.startswith("/"):
            buf.start_completion()

    @bindings.add("enter")
    def _enter(event):
        buf = event.app.current_buffer
        text = buf.text

        # Single-line, not a block header and not a backslash continuation => accept.
        if "\n" not in text:
            first_line = text
            if first_line.endswith("\\") or _is_block_header(first_line):
                indent = _compute_indent(text)
                buf.insert_text("\n" + indent)
                return

            buf.validate_and_handle()
            return

        # Multiline: if the current (last) line is empty => accept,
        # unless a try: block still needs its catch clause.
        lines = text.split("\n")
        if lines[-1].strip() == "":
            content = "\n".join(lines[:-1])
            if _awaits_catch(content):
                # Dedent to the try: level so the user can type catch:.
                first = lines[0]
                base = " " * (len(first) - len(first.lstrip()))
                buf.insert_text("\n" + base)
                return

            # Remove trailing empty line before accepting.
            buf.text = content
            buf.cursor_position = len(buf.text)
            buf.validate_and_handle()
            return

        # Still in continuation — check for backslash or block header.
        last = lines[-1]
        if last.rstrip().endswith("\\") or _is_block_header(last):
            indent = _compute_indent(text)
            buf.insert_text("\n" + indent)
            return

        # Non-empty continuation line, not a block header => insert newline
        # to allow more lines; user submits with empty line.
        indent = _compute_indent(text)
        buf.insert_text("\n" + indent)

    session: PromptSession[str] = PromptSession(
        history=history,
        lexer=lexer,
        completer=_SlashCompleter(),
        complete_while_typing=True,
        key_bindings=bindings,
        multiline=True,
        prompt_continuation="... ",
    )

    print("shakar repl — Ctrl-D to exit, / for commands")

    while True:
        try:
            text = session.prompt(">>> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            continue

        text = _normalize(text)
        if not text.strip():
            continue

        # Slash command?
        if _handle_slash(text, frame_box):
            continue

        try:
            result, stmt = repl_eval(text, frame_box[0])
        except (ParseError, LexError, ShakarRuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            if debug_py_trace_enabled() and isinstance(exc, ShakarRuntimeError):
                tb = getattr(exc, "shk_py_trace", None)
                if tb:
                    print("\nPython traceback:", file=sys.stderr)
                    print(
                        "".join(traceback.format_tb(tb)),
                        file=sys.stderr,
                        end="",
                    )
            continue

        if not stmt and not isinstance(result, ShkNil):
            print(result)
