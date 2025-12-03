from __future__ import annotations

import importlib
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
from .tree import Node
from .types import (
    ShkNull, ShkNumber, ShkString, ShkBool, ShkArray, ShkObject, ShkCommand,
    ShkSelector, SelectorIndex, SelectorSlice, SelectorPart,
    ShkFn, ShkDecorator, DecoratorConfigured, DecoratorContinuation,
    BoundMethod, BuiltinMethod, Descriptor, StdlibFunction, DeferEntry,
    ShkValue, DotValue, Frame, ShkType, ShkOptional,
    ShakarRuntimeError, ShakarTypeError, ShakarArityError, ShakarKeyError,
    ShakarIndexError, ShakarMethodNotFound, CommandError, ShakarAssertionError,
    ShakarReturnSignal, ShakarBreakSignal, ShakarContinueSignal,
    Method, MethodRegistry, Builtins,
    is_shk_value, _ensure_shk_value, StdlibFn
)

_STDLIB_INITIALIZED = False

def init_stdlib() -> None:
    """Load stdlib modules (idempotent) so register_stdlib hooks run."""
    global _STDLIB_INITIALIZED

    if _STDLIB_INITIALIZED:
        return

    for module_name in ("shakar_ref.stdlib",):
        try:
            importlib.import_module(module_name)
            _STDLIB_INITIALIZED = True
            return
        except ModuleNotFoundError:
            continue

    stdlib_expect_arity: Dict[str, int] = {}

    known_arity: Dict[Tuple[str, str], Tuple[str, int]] = {
        ("array","map"):     ("exact", 1),
        ("array","filter"):  ("exact", 1),
        ("array","zipWith"): ("exact", 2),
        ("string","len"):    ("exact", 0),
        ("string","trim"):   ("exact", 0),
        ("string","lower"):  ("exact", 0),
        ("string","upper"):  ("exact", 0),
        ("string","hasPrefix"): ("exact", 1),
        ("string","hasSuffix"): ("exact", 1),
        ("string","isAscii"): ("exact", 0),
        ("object","items"):  ("exact", 0),
    }

def register_method(registry: MethodRegistry, name: str):
    def dec(fn: Callable[..., ShkValue]):
        registry[name] = fn
        return fn

    return dec

def register_array(name: str):
    return register_method(Builtins.array_methods, name)

def register_string(name: str):
    return register_method(Builtins.string_methods, name)

def register_object(name: str):
    return register_method(Builtins.object_methods, name)

def register_command(name: str):
    return register_method(Builtins.command_methods, name)

def register_stdlib(name: str, *, arity: Optional[int] = None):
    def dec(fn: StdlibFn):
        Builtins.stdlib_functions[name] = StdlibFunction(fn=fn, arity=arity)
        return fn

    return dec

def _string_expect_arity(method: str, args: List[ShkValue], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(f"string.{method} expects {expected} argument(s); got {len(args)}")

def _string_arg(method: str, arg: ShkValue) -> str:
    if isinstance(arg, ShkString):
        return arg.value

    raise ShakarTypeError(f"string.{method} expects a string argument")

@register_string("join")
def _string_join(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    items: List[ShkValue]
    if len(args) == 1 and isinstance(args[0], ShkArray):
        items = args[0].items
    else:
        items = args

    from .eval.common import stringify
    strings = []

    for item in items:
        strings.append(stringify(item))

    return ShkString(recv.value.join(strings))

@register_string("trim")
def _string_trim(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    _string_expect_arity("trim", args, 0)

    return ShkString(recv.value.strip())

@register_string("lower")
def _string_lower(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    _string_expect_arity("lower", args, 0)

    return ShkString(recv.value.lower())

@register_string("upper")
def _string_upper(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkString:
    _string_expect_arity("upper", args, 0)

    return ShkString(recv.value.upper())

@register_string("hasPrefix")
def _string_has_prefix(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkBool:
    _string_expect_arity("hasPrefix", args, 1)
    prefix = _string_arg("hasPrefix", args[0])

    return ShkBool(recv.value.startswith(prefix))

@register_string("hasSuffix")
def _string_has_suffix(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkBool:
    _string_expect_arity("hasSuffix", args, 1)
    suffix = _string_arg("hasSuffix", args[0])

    return ShkBool(recv.value.endswith(suffix))

@register_string("isAscii")
def _string_is_ascii(_frame: Frame, recv: ShkString, args: List[ShkValue]) -> ShkBool:
    _string_expect_arity("isAscii", args, 0)

    return ShkBool(recv.value.isascii())

@register_command("run")
def _command_run(_frame: Frame, recv: ShkCommand, args: List[ShkValue]) -> ShkValue:
    if args:
        raise ShakarArityError(f"command.run expects 0 args; got {len(args)}")
    cmd = recv.render()

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except OSError as exc:
        raise ShakarRuntimeError(f"Command execution failed: {exc}") from exc

    if result.returncode != 0:
        raise CommandError(cmd, result.returncode, result.stdout, result.stderr)

    return ShkString(result.stdout)

def call_builtin_method(recv: ShkValue, name: str, args: List[ShkValue], frame: 'Frame') -> ShkValue:
    registry_by_type: Dict[type, MethodRegistry] = {
        ShkArray: Builtins.array_methods,
        ShkString: Builtins.string_methods,
        ShkObject: Builtins.object_methods,
        ShkCommand: Builtins.command_methods,
    }

    registry = registry_by_type.get(type(recv))
    if registry:
        handler = registry.get(name)
        if handler is not None:
            return handler(frame, recv, args)

    raise ShakarMethodNotFound(recv, name)

def call_shkfn(fn: ShkFn, positional: List[ShkValue], subject: Optional[ShkValue], caller_frame: 'Frame') -> ShkValue:
    """
    Subjectful call semantics:
    - subject is available to callee as frame.dot
    - If fn.params is None, treat as unary subject-lambda: ignore positional, evaluate body with dot bound.
    - Else: arity must match len(fn.params); bind params; dot=subject.
    """

    if fn.decorators:
        return _call_shkfn_with_decorators(fn, positional, subject, caller_frame)

    return _call_shkfn_raw(fn, positional, subject, caller_frame)

def _call_shkfn_raw(fn: ShkFn, positional: List[ShkValue], subject: Optional[ShkValue], caller_frame: 'Frame') -> ShkValue:
    _ = caller_frame
    from .evaluator import eval_node  # local import to avoid cycle

    if fn.params is None:
        if subject is None:
            if not positional:
                raise ShakarArityError("Subject-only amp_lambda expects a subject argument")
            subject, *positional = positional

        if positional:
            raise ShakarArityError(f"Subject-only amp_lambda does not take positional args; got {len(positional)} extra")

        callee_frame = Frame(parent=fn.frame, dot=subject)
        callee_frame.mark_function_frame()

        try:
            return _ensure_shk_value(eval_node(fn.body, callee_frame))
        except ShakarReturnSignal as signal:
            return signal.value

    callee_frame = Frame(parent=fn.frame, dot=subject)

    if len(positional) != len(fn.params):
        raise ShakarArityError(f"Function expects {len(fn.params)} args; got {len(positional)}")

    for name, val in zip(fn.params, positional):
        callee_frame.define(name, val)

    callee_frame.mark_function_frame()

    try:
        return _ensure_shk_value(eval_node(fn.body, callee_frame))
    except ShakarReturnSignal as signal:
        return signal.value

def _call_shkfn_with_decorators(fn: ShkFn, positional: List[ShkValue], subject: Optional[ShkValue], caller_frame: 'Frame') -> ShkValue:
    chain = fn.decorators or ()
    args = ShkArray(list(positional))

    return _run_decorator_chain(fn, chain, 0, args, subject, caller_frame)

def _run_decorator_chain(
    fn: ShkFn,
    chain: Tuple[DecoratorConfigured, ...],
    index: int,
    args_value: ShkArray,
    subject: Optional[ShkValue],
    caller_frame: 'Frame',
) -> ShkValue:
    if index >= len(chain):
        return _call_shkfn_raw(fn, list(args_value.items), subject, caller_frame)
    continuation = DecoratorContinuation(
        fn=fn,
        decorators=chain,
        index=index + 1,
        subject=subject,
        caller_frame=caller_frame,
    )
    inst = chain[index]

    return _execute_decorator_instance(inst, continuation, args_value, subject, caller_frame)

def _execute_decorator_instance(
    inst: DecoratorConfigured,
    continuation: DecoratorContinuation,
    args_value: ShkArray,
    subject: Optional[ShkValue],
    caller_frame: 'Frame',
) -> ShkValue:
    from .evaluator import eval_node  # defer to avoid cycle
    deco_frame = Frame(parent=inst.decorator.frame, dot=subject)
    deco_frame.mark_function_frame()
    params = inst.decorator.params or []

    for name, val in zip(params, inst.args):
        deco_frame.define(name, val)
    deco_frame.define('f', continuation)
    deco_frame.define('args', args_value)

    try:
        eval_node(inst.decorator.body, deco_frame)
    except ShakarReturnSignal as signal:
        return signal.value

    updated_args_value = deco_frame.get('args')

    return continuation.invoke(updated_args_value)

def _coerce_decorator_args(value: ShkValue) -> ShkArray:
    if isinstance(value, ShkArray):
        return value

    raise ShakarTypeError("Decorator args must be an array value")

# Runtime type constants for structural matching
TYPE_INT = ShkType("Int", ShkNumber)
TYPE_FLOAT = ShkType("Float", ShkNumber)
TYPE_STR = ShkType("Str", ShkString)
TYPE_BOOL = ShkType("Bool", ShkBool)
TYPE_ARRAY = ShkType("Array", ShkArray)
TYPE_OBJECT = ShkType("Object", ShkObject)
TYPE_NIL = ShkType("Nil", ShkNull)

Builtins.type_constants = {
    "Int": TYPE_INT,
    "Float": TYPE_FLOAT,
    "Str": TYPE_STR,
    "Bool": TYPE_BOOL,
    "Array": TYPE_ARRAY,
    "Object": TYPE_OBJECT,
    "Nil": TYPE_NIL,
}
