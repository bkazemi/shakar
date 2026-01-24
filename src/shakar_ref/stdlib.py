"""Built-in stdlib functions (print, etc.) registered via shakar_runtime."""

from __future__ import annotations

import time
from typing import List, Optional

from .runtime import (
    register_stdlib,
    register_module_factory,
    ShkNull,
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
    ShkObject,
    ShkOptional,
    ShkUnion,
)
from .runtime import ShakarRuntimeError
from .eval.helpers import is_truthy, current_function_frame, name_in_current_frame
from .utils import envvar_value_by_name


@register_stdlib("int")
def std_int(_frame, args: List[ShkValue]) -> ShkNumber:
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

    if isinstance(value, ShkNull):
        return "null"
    return str(value)


@register_stdlib("print")
def std_print(_frame, args: List[ShkValue]) -> ShkNull:
    rendered = [_render(arg) for arg in args]
    print(*rendered)
    return ShkNull()


@register_stdlib("sleep")
def std_sleep(_frame, args: List[ShkValue]):
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
    return ShkNull()


@register_stdlib("channel")
def std_channel(_frame, args: List[ShkValue]) -> ShkChannel:
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
def std_error(_frame, args: List[ShkValue]) -> ShkObject:
    if len(args) < 2 or len(args) > 3:
        raise ShakarTypeError("error(type, message[, data]) expects 2 or 3 arguments")
    type_arg, message_arg, *rest = args

    if not isinstance(type_arg, ShkString) or not isinstance(message_arg, ShkString):
        raise ShakarTypeError("error expects string type and message")
    data = rest[0] if rest else ShkNull()
    slots: dict[str, ShkValue] = {
        "__error__": ShkBool(True),
        "type": type_arg,
        "message": message_arg,
        "data": data,
    }

    return ShkObject(slots)


@register_stdlib("all")
def std_all(_frame, args: List[ShkValue]) -> ShkBool:
    if not args:
        raise ShakarRuntimeError("all() expects at least one argument")

    iterable: List[ShkValue] = args if len(args) > 1 else list(_iter_coerce(args[0]))

    for val in iterable:
        if not is_truthy(val):
            return ShkBool(False)
    return ShkBool(True)


@register_stdlib("any")
def std_any(_frame, args: List[ShkValue]) -> ShkBool:
    if not args:
        raise ShakarRuntimeError("any() expects at least one argument")

    iterable: List[ShkValue] = args if len(args) > 1 else list(_iter_coerce(args[0]))

    for val in iterable:
        if is_truthy(val):
            return ShkBool(True)
    return ShkBool(False)


@register_stdlib("mixin")
def std_mixin(frame, args: List[ShkValue]) -> ShkNull:
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

    return ShkNull()


def _iter_coerce(value: ShkValue):
    """Coerce arrays/strings/objects/iterables into iteration."""
    if isinstance(value, ShkString):
        for ch in value.value:
            yield ShkString(ch)
    elif isinstance(value, ShkObject):
        for key in value.slots:
            yield ShkString(key)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield v
    elif hasattr(value, "items") and isinstance(value.items, list):
        for v in value.items:
            yield v
    else:
        raise ShakarTypeError("all/any expects iterable or multiple args")


@register_stdlib("Optional")
def std_optional(_frame, args: List[ShkValue]) -> ShkOptional:
    """Wrap a schema value to mark it as optional in structural matching."""
    if len(args) != 1:
        raise ShakarTypeError("Optional() expects exactly one argument")
    return ShkOptional(args[0])


@register_stdlib("Union")
def std_union(_frame, args: List[ShkValue]) -> ShkUnion:
    """Create a union type that matches any of the provided alternatives."""
    if len(args) == 0:
        raise ShakarTypeError("Union() requires at least one type alternative")
    return ShkUnion(tuple(args))


def _term_read_key(_frame, args: List[ShkValue]) -> ShkString:
    if args:
        raise ShakarTypeError("term.read_key() expects no arguments")
    import sys

    data = sys.stdin.buffer.read(1)
    if not data:
        return ShkString("")
    return ShkString(data.decode("latin-1"))


def _term_read_key_timeout(_frame, args: List[ShkValue]) -> ShkString:
    if len(args) != 1:
        raise ShakarTypeError("term.read_key_timeout(ms) expects one argument")
    duration = args[0]

    if isinstance(duration, ShkDuration):
        seconds = max(0.0, float(duration.nanos) / 1_000_000_000.0)
    elif isinstance(duration, ShkNumber):
        seconds = max(0.0, float(duration.value)) / 1000.0
    else:
        raise ShakarTypeError("term.read_key_timeout(ms) expects number or duration")

    import select
    import sys
    import os

    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], seconds)
    if not ready:
        return ShkString("")

    data = os.read(fd, 1)
    if not data:
        return ShkString("")
    return ShkString(data.decode("latin-1"))


_TERM_RAW_STATE: Optional[list[int]] = None


def _term_raw(_frame, args: List[ShkValue]) -> ShkBool:
    if len(args) != 1:
        raise ShakarTypeError("term.raw(on) expects one boolean argument")
    if not isinstance(args[0], ShkBool):
        raise ShakarTypeError("term.raw(on) expects a boolean")

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


def _term_is_interactive(_frame, args: List[ShkValue]) -> ShkBool:
    if args:
        raise ShakarTypeError("term.is_interactive() expects no arguments")
    import sys

    return ShkBool(sys.stdin.isatty() and sys.stdout.isatty())


def _term_write(_frame, args: List[ShkValue]) -> ShkNull:
    if len(args) != 1 or not isinstance(args[0], ShkString):
        raise ShakarTypeError("term.write(str) expects one string argument")
    import sys

    try:
        sys.stdout.buffer.write(args[0].value.encode("latin-1"))
    except UnicodeEncodeError:
        sys.stdout.write(args[0].value)
    sys.stdout.flush()
    return ShkNull()


def _build_term_module() -> ShkModule:
    slots: dict[str, ShkValue] = {
        "read_key": StdlibFunction(fn=_term_read_key, arity=0),
        "read_key_timeout": StdlibFunction(fn=_term_read_key_timeout, arity=1),
        "is_interactive": StdlibFunction(fn=_term_is_interactive, arity=0),
        "write": StdlibFunction(fn=_term_write, arity=1),
        "raw": StdlibFunction(fn=_term_raw, arity=1),
    }
    return ShkModule(slots=slots, name="term")


register_module_factory("term", _build_term_module)
