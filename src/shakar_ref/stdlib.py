"""Built-in stdlib functions (print, etc.) registered via shakar_runtime."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from .runtime import (
    register_stdlib,
    register_module_factory,
    ShkNil,
    ShkString,
    ShkNumber,
    ShkDuration,
    ShkBool,
    ShkEnvVar,
    ShkChannel,
    ShkModule,
    ShkValue,
    StdlibFunction,
    ShakarImportError,
    ShakarTypeError,
    ShkArray,
    ShkFan,
    ShkObject,
    ShkOptional,
    ShkUnion,
)
from .runtime import ShakarRuntimeError
from .eval.helpers import is_truthy, current_function_frame, name_in_current_frame
from .utils import envvar_value_by_name


@register_stdlib("int")
def std_int(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkNumber:
    args = _with_subject(subject, args)
    if len(args) != 1:
        raise ShakarTypeError("int() expects exactly one argument")

    val = args[0]

    if isinstance(val, ShkNumber):
        return ShkNumber(int(val.value))

    if isinstance(val, ShkBool):
        return ShkNumber(int(bool(val.value)))

    if isinstance(val, ShkString):
        try:
            return ShkNumber(int(val.value))
        except ValueError:
            raise ShakarTypeError("int() expects numeric string or number")

    raise ShakarTypeError("int() expects number or numeric string")


def _render(value):
    if isinstance(value, ShkString):
        return value.value

    if isinstance(value, ShkEnvVar):
        env_val = envvar_value_by_name(value.name)
        return env_val if env_val is not None else "nil"

    if isinstance(value, (ShkNumber, ShkBool)):
        return str(value.value)

    if isinstance(value, ShkNil):
        return "nil"
    return str(value)


def _with_subject(subject: Optional[ShkValue], args: List[ShkValue]) -> List[ShkValue]:
    if subject is None:
        return list(args)
    return [subject, *args]


@register_stdlib("print", named=True)
def std_print(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkNil:
    args = _with_subject(subject, args)
    rendered = [_render(arg) for arg in args]
    sep = " "

    if named:
        unknown = set(named.keys()) - {"sep"}
        if unknown:
            raise ShakarTypeError(
                f"print: unknown named argument(s): {', '.join(sorted(unknown))}"
            )
        sep_val = named.get("sep")
        if sep_val is not None:
            sep = _render(sep_val)

    print(*rendered, sep=sep)
    return ShkNil()


@register_stdlib("tap", named=True)
def std_tap(
    frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkValue:
    args = _with_subject(subject, args)
    if len(args) < 2:
        raise ShakarTypeError("tap(value, fn[, ...args]) expects at least 2 arguments")

    value = args[0]
    callback = args[1]
    callback_args = [value, *args[2:]]

    # Deferred imports avoid runtime import cycles with evaluator modules.
    from .eval.chains import call_value
    from .evaluator import eval_node

    call_value(callback, callback_args, frame, eval_func=eval_node, named=named)
    return value


@register_stdlib("sleep")
def std_sleep(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
):
    args = _with_subject(subject, args)
    if len(args) != 1:
        raise ShakarTypeError("sleep expects exactly one argument")
    duration = args[0]

    if isinstance(duration, ShkDuration):
        seconds = max(0.0, float(duration.nanos) / 1_000_000_000.0)
    elif isinstance(duration, ShkNumber):
        milliseconds = max(0.0, float(duration.value))
        seconds = milliseconds / 1000.0
    else:
        raise ShakarTypeError("sleep expects a numeric duration")

    time.sleep(seconds)
    return ShkNil()


@register_stdlib("channel")
def std_channel(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkChannel:
    args = _with_subject(subject, args)
    if len(args) > 1:
        raise ShakarTypeError("channel() expects at most one argument")
    if not args:
        return ShkChannel(0)
    cap = args[0]
    if isinstance(cap, ShkNumber):
        capacity = int(cap.value)
        if capacity < 0:
            raise ShakarTypeError("channel() capacity cannot be negative")
        return ShkChannel(capacity)
    raise ShakarTypeError("channel() expects a numeric capacity")


@register_stdlib("error")
def std_error(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkObject:
    args = _with_subject(subject, args)
    if len(args) < 2 or len(args) > 3:
        raise ShakarTypeError("error(type, message[, data]) expects 2 or 3 arguments")
    type_arg, message_arg, *rest = args

    if not isinstance(type_arg, ShkString) or not isinstance(message_arg, ShkString):
        raise ShakarTypeError("error expects string type and message")
    data = rest[0] if rest else ShkNil()
    slots: dict[str, ShkValue] = {
        "__error__": ShkBool(True),
        "type": type_arg,
        "message": message_arg,
        "data": data,
    }

    return ShkObject(slots)


@register_stdlib("all")
def std_all(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkBool:
    args = _with_subject(subject, args)
    if not args:
        raise ShakarRuntimeError("all() expects at least one argument")

    iterable: List[ShkValue] = args if len(args) > 1 else list(_iter_coerce(args[0]))

    for val in iterable:
        if not is_truthy(val):
            return ShkBool(False)
    return ShkBool(True)


@register_stdlib("any")
def std_any(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkBool:
    args = _with_subject(subject, args)
    if not args:
        raise ShakarRuntimeError("any() expects at least one argument")

    iterable: List[ShkValue] = args if len(args) > 1 else list(_iter_coerce(args[0]))

    for val in iterable:
        if is_truthy(val):
            return ShkBool(True)
    return ShkBool(False)


@register_stdlib("mixin")
def std_mixin(
    frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkNil:
    args = _with_subject(subject, args)
    if len(args) != 1:
        raise ShakarTypeError("mixin() expects exactly one argument")

    obj = args[0]
    slots = getattr(obj, "slots", None)
    if slots is None:
        raise ShakarTypeError("mixin() expects an object or module")

    target = current_function_frame(frame) or frame

    for key in slots:
        if name_in_current_frame(target, key):
            raise ShakarImportError(f"Mixin collision: '{key}' already exists in scope")

    for key, value in slots.items():
        target.define(key, value)

    return ShkNil()


def _iter_coerce(value: ShkValue):
    """Coerce arrays/strings/objects/iterables into iteration."""
    if isinstance(value, ShkString):
        for ch in value.value:
            yield ShkString(ch)
    elif isinstance(value, ShkObject):
        for key in value.slots:
            yield ShkString(key)
    elif isinstance(value, (ShkArray, ShkFan)):
        for v in value.items:
            yield v
    else:
        raise ShakarTypeError("all/any expects iterable or multiple args")


@register_stdlib("Optional")
def std_optional(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkOptional:
    """Wrap a schema value to mark it as optional in structural matching."""
    args = _with_subject(subject, args)
    if len(args) != 1:
        raise ShakarTypeError("Optional() expects exactly one argument")
    return ShkOptional(args[0])


@register_stdlib("Union")
def std_union(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkUnion:
    """Create a union type that matches any of the provided alternatives."""
    args = _with_subject(subject, args)
    if len(args) == 0:
        raise ShakarTypeError("Union() requires at least one type alternative")
    return ShkUnion(tuple(args))


# --- Unified io module ---
# Platform-agnostic I/O that works in terminal and browser.

_TERM_RAW_STATE: Optional[list[int]] = None


def _detect_platform() -> str:
    """Detect runtime platform: 'browser' or 'terminal'."""
    try:
        import js  # Pyodide provides this in browser

        return "browser"
    except ImportError:
        return "terminal"


# Browser key input via SharedArrayBuffer.
# Key codes: left=1, right=2, down=3, up=4, space=5, q=6
_IO_KEY_NAMES = {1: "left", 2: "right", 3: "down", 4: "up", 5: " ", 6: "q"}

# Mutable read index (list so it's writable from inline runPython)
_io_key_read_idx: List[int] = [0]


def _io_read_key(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkString:
    """Read a single key. Optional timeout in ms; blocks forever if omitted."""
    args = _with_subject(subject, args)
    if len(args) > 1:
        raise ShakarTypeError("io.read_key([timeout_ms]) expects at most one argument")

    timeout_ms: Optional[float] = None
    if args:
        duration = args[0]
        if isinstance(duration, ShkDuration):
            timeout_ms = max(0.0, float(duration.nanos) / 1_000_000.0)
        elif isinstance(duration, ShkNumber):
            timeout_ms = max(0.0, float(duration.value))
        else:
            raise ShakarTypeError(
                "io.read_key([timeout_ms]) expects number or duration"
            )

    if _detect_platform() == "browser":
        return _io_browser_read_key(timeout_ms)
    else:
        return _io_term_read_key(timeout_ms)


def _io_term_read_key(timeout_ms: Optional[float]) -> ShkString:
    """Terminal implementation using select."""
    import select
    import sys
    import os

    if not sys.stdin.isatty():
        return ShkString("")

    fd = sys.stdin.fileno()

    if timeout_ms is None:
        # Blocking read
        data = os.read(fd, 1)
        if not data:
            return ShkString("")
        return ShkString(data.decode("latin-1"))

    # Timeout read
    ready, _, _ = select.select([fd], [], [], timeout_ms / 1000.0)
    if not ready:
        return ShkString("")

    data = os.read(fd, 1)
    if not data:
        return ShkString("")
    return ShkString(data.decode("latin-1"))


def _io_browser_read_key(timeout_ms: Optional[float]) -> ShkString:
    """Browser implementation reading keys from SharedArrayBuffer ring buffer.
    Main thread writes key codes; we read via Atomics.wait + Atomics.load.
    This works while runPython blocks the worker because SAB is shared memory."""
    import js

    buf = js.self.shk_key_buf
    if buf is None:
        return ShkString("")

    buf_len = int(buf.length)
    if buf_len <= 1:
        return ShkString("")
    buf_slots = buf_len - 1

    def _try_read() -> Optional[str]:
        write_idx = int(js.Atomics.load(buf, 0)) & 0xFFFFFFFF
        read_idx = _io_key_read_idx[0] & 0xFFFFFFFF
        if read_idx != write_idx:
            slot = (read_idx % buf_slots) + 1
            code = js.Atomics.load(buf, slot)
            _io_key_read_idx[0] = (read_idx + 1) & 0xFFFFFFFF
            return _IO_KEY_NAMES.get(code, "")
        return None

    # Check immediately
    key = _try_read()
    if key is not None:
        return ShkString(key)

    # Poll with Atomics.wait (blocks worker thread, not main thread)
    cur_idx = int(js.Atomics.load(buf, 0)) & 0xFFFFFFFF
    if timeout_ms is None:
        # Block indefinitely, waking every 200ms to recheck
        while True:
            js.Atomics.wait(buf, 0, cur_idx | 0, 200)
            key = _try_read()
            if key is not None:
                return ShkString(key)
            cur_idx = int(js.Atomics.load(buf, 0)) & 0xFFFFFFFF

    # Timed wait
    start = time.monotonic() * 1000
    remaining = timeout_ms
    while remaining > 0:
        wait_ms = min(remaining, 200)
        js.Atomics.wait(buf, 0, cur_idx | 0, int(wait_ms))
        key = _try_read()
        if key is not None:
            return ShkString(key)
        cur_idx = int(js.Atomics.load(buf, 0)) & 0xFFFFFFFF
        remaining = timeout_ms - (time.monotonic() * 1000 - start)

    return ShkString("")


def _io_write(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkNil:
    """Write text to output."""
    args = _with_subject(subject, args)
    if len(args) != 1 or not isinstance(args[0], ShkString):
        raise ShakarTypeError("io.write(str) expects one string argument")

    text = args[0].value

    if _detect_platform() == "browser":
        import re
        import js

        # Strip ANSI escape codes for browser
        clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
        clean = re.sub(r"\x1b\[\?[0-9;]*[a-zA-Z]", "", clean)
        js.self.shk_io_write(clean)
    else:
        import sys

        try:
            sys.stdout.buffer.write(text.encode("latin-1"))
        except UnicodeEncodeError:
            sys.stdout.write(text)
        sys.stdout.flush()

    return ShkNil()


def _io_clear(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkNil:
    """Clear the screen/output."""
    args = _with_subject(subject, args)
    if args:
        raise ShakarTypeError("io.clear() expects no arguments")

    if _detect_platform() == "browser":
        import js

        js.self.shk_io_clear()
    else:
        import sys

        # ANSI clear screen + home
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()

    return ShkNil()


def _io_overwrite(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkNil:
    """Atomic clear and write to prevent flickering."""
    args = _with_subject(subject, args)
    if len(args) != 1 or not isinstance(args[0], ShkString):
        raise ShakarTypeError("io.overwrite(str) expects one string argument")

    text = args[0].value

    if _detect_platform() == "browser":
        import re
        import js

        # Strip ANSI escape codes for browser
        clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
        clean = re.sub(r"\x1b\[\?[0-9;]*[a-zA-Z]", "", clean)
        js.self.shk_io_overwrite(clean)
    else:
        import sys

        # ANSI clear screen + home
        sys.stdout.write("\x1b[2J\x1b[H")
        try:
            sys.stdout.buffer.write(text.encode("latin-1"))
        except UnicodeEncodeError:
            sys.stdout.write(text)
        sys.stdout.flush()

    return ShkNil()


def _io_is_interactive(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkBool:
    """Check if interactive I/O is available."""
    args = _with_subject(subject, args)
    if args:
        raise ShakarTypeError("io.is_interactive() expects no arguments")

    if _detect_platform() == "browser":
        return ShkBool(True)
    else:
        import sys

        return ShkBool(sys.stdin.isatty() and sys.stdout.isatty())


def _io_raw(
    _frame,
    subject: Optional[ShkValue],
    args: List[ShkValue],
    named: Optional[Dict[str, ShkValue]] = None,
) -> ShkBool:
    """Set raw mode (terminal only, no-op in browser)."""
    args = _with_subject(subject, args)
    if len(args) != 1:
        raise ShakarTypeError("io.raw(on) expects one boolean argument")
    if not isinstance(args[0], ShkBool):
        raise ShakarTypeError("io.raw(on) expects a boolean")

    if _detect_platform() == "browser":
        return ShkBool(True)

    import sys

    if not sys.stdin.isatty():
        return ShkBool(False)

    import termios
    import tty

    global _TERM_RAW_STATE
    enable = bool(args[0].value)
    fd = sys.stdin.fileno()

    if enable:
        if _TERM_RAW_STATE is None:
            _TERM_RAW_STATE = termios.tcgetattr(fd)
            tty.setraw(fd)
        return ShkBool(True)

    if _TERM_RAW_STATE is not None:
        termios.tcsetattr(fd, termios.TCSADRAIN, _TERM_RAW_STATE)
        _TERM_RAW_STATE = None
    return ShkBool(True)


def _build_io_module() -> ShkModule:
    """Build the unified io module."""
    slots: dict[str, ShkValue] = {
        "read_key": StdlibFunction(fn=_io_read_key, name="read_key"),  # 0 or 1 args
        "write": StdlibFunction(fn=_io_write, arity=1, name="write"),
        "clear": StdlibFunction(fn=_io_clear, arity=0, name="clear"),
        "overwrite": StdlibFunction(fn=_io_overwrite, arity=1, name="overwrite"),
        "is_interactive": StdlibFunction(
            fn=_io_is_interactive, arity=0, name="is_interactive"
        ),
        "raw": StdlibFunction(fn=_io_raw, arity=1, name="raw"),
        "platform": ShkString(_detect_platform()),
    }
    return ShkModule(slots=slots, name="io")


register_module_factory("io", _build_io_module)
