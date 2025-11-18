from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------- Value Model (only Sh* -> Shk*) ----------

@dataclass
class ShkNull:
    def __repr__(self) -> str:
        return "null"

@dataclass
class ShkNumber:
    value: float
    def __repr__(self) -> str:
        v = self.value
        return str(int(v)) if v.is_integer() else str(v)

@dataclass
class ShkString:
    value: str
    def __repr__(self) -> str:
        return f'"{self.value}"'

@dataclass
class ShkBool:
    value: bool
    def __repr__(self) -> str:
        return "true" if self.value else "false"

@dataclass
class ShkArray:
    items: List[Any]
    def __repr__(self) -> str:
        return "[" + ", ".join(repr(x) for x in self.items) + "]"

@dataclass
class ShkObject:
    slots: Dict[str, Any]
    def __repr__(self) -> str:
        pairs = []

        for k, v in self.slots.items():
            pairs.append(f"{k}: {repr(v)}")

        return "{ " + ", ".join(pairs) + " }"

@dataclass
class SelectorIndex:
    value: Any

@dataclass
class SelectorSlice:
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]
    clamp: bool
    exclusive_stop: bool = False

SelectorPart = Union[SelectorIndex, SelectorSlice]

@dataclass
class ShkSelector:
    parts: List[SelectorPart]
    def __repr__(self) -> str:
        descr = []

        for part in self.parts:
            if isinstance(part, SelectorIndex):
                descr.append(str(part.value))
            else:
                bits = []
                bits.append("" if part.start is None else str(part.start))
                bits.append("" if part.stop is None else ("<" + str(part.stop) if part.exclusive_stop else str(part.stop)))

                if part.step is not None:
                    bits.append(str(part.step))

                while len(bits) < 3:
                    bits.append("")
                descr.append(":".join(bits[:3]).rstrip(":"))

        return "selector{" + ", ".join(descr) + "}"

@dataclass
class ShkDecorator:
    params: Optional[List[str]]
    body: Any
    frame: 'Frame'

@dataclass
class DecoratorConfigured:
    decorator: ShkDecorator
    args: List[Any]

@dataclass
class ShkFn:
    params: Optional[List[str]]  # None for subject-only amp-lambda
    body: Any                    # AST node
    frame: 'Frame'                   # Closure frame
    decorators: Optional[Tuple[DecoratorConfigured, ...]] = None
    kind: str = "fn"
    def __repr__(self) -> str:
        body_label = getattr(self.body, 'data', type(self.body).__name__)
        label = "amp-fn" if self.kind == "amp" else self.kind

        if self.params is None:
            param_desc = "subject"
        else:
            param_desc = self.params if self.params else "nullary"

        return f"<{label} params={param_desc} body={body_label}>"

@dataclass
class BoundMethod:
    fn: ShkFn
    subject: Any

@dataclass
class BuiltinMethod:
    name: str
    subject: Any

@dataclass
class Descriptor:
    getter: Any = None  # ShkFn or None
    setter: Any = None  # ShkFn or None

@dataclass
class DecoratorContinuation:
    fn: ShkFn
    decorators: Tuple[DecoratorConfigured, ...]
    index: int
    subject: Any
    caller_frame: 'Frame'

    def invoke(self, args_value: Any) -> Any:
        args = _coerce_decorator_args(args_value)
        return _run_decorator_chain(self.fn, self.decorators, self.index, args, self.subject, self.caller_frame)

@dataclass
class DeferEntry:
    """Scheduled defer thunk plus optional metadata for dependency ordering."""
    thunk: Callable[[], None]
    label: Optional[str] = None
    deps: List[str] = field(default_factory=list)

class Frame:
    def __init__(self, parent: Optional['Frame']=None, dot: Any=None, source: Optional[str]=None):
        self.parent = parent
        self.vars: Dict[str, Any] = {}
        self.dot = dot
        self._defer_stack: List[List[DeferEntry]] = []
        self._is_function_frame = False
        self._active_error: Optional[ShakarRuntimeError] = None

        if parent is None and Builtins.stdlib_functions:
            for name, std in Builtins.stdlib_functions.items():
                self.vars[name] = std

        if source is not None:
            self.source = source
        elif parent is not None and hasattr(parent, 'source'):
            self.source = parent.source
        else:
            self.source = None

    def define(self, name: str, val: Any) -> None:
        self.vars[name] = val

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]

        if self.parent is not None:
            return self.parent.get(name)

        raise ShakarRuntimeError(f"Name '{name}' not found")

    def set(self, name: str, val: Any) -> None:
        if name in self.vars:
            self.vars[name] = val
            return

        if self.parent is not None:
            self.parent.set(name, val)
            return

        raise ShakarRuntimeError(f"Name '{name}' not found")

    def push_defer_frame(self) -> None:
        self._defer_stack.append([])

    def pop_defer_frame(self) -> List[DeferEntry]:
        if not self._defer_stack:
            return []

        return self._defer_stack.pop()

    def current_defer_frame(self) -> List[DeferEntry]:
        if not self._defer_stack:
            raise ShakarRuntimeError("Cannot use defer outside of a block")

        return self._defer_stack[-1]

    def has_defer_frame(self) -> bool:
        return bool(self._defer_stack)

    def mark_function_frame(self) -> None:
        self._is_function_frame = True

    def is_function_frame(self) -> bool:
        return self._is_function_frame

# ---------- Exceptions (keep Shakar* canonical) ----------

class ShakarRuntimeError(Exception):
    pass

class ShakarTypeError(ShakarRuntimeError):
    pass

class ShakarArityError(ShakarRuntimeError):
    pass

class ShakarKeyError(ShakarRuntimeError):
    def __init__(self, key: str):
        super().__init__(f"Key '{key}' not found")
        self.key = key

class ShakarIndexError(ShakarRuntimeError):
    def __init__(self, message: str = "Index out of bounds"):
        super().__init__(message)

class ShakarMethodNotFound(ShakarRuntimeError):
    def __init__(self, recv: Any, name: str):
        super().__init__(f"{type(recv).__name__} has no builtin method '{name}'")
        self.receiver = recv
        self.name = name

class ShakarAssertionError(ShakarRuntimeError):
    pass

class ShakarReturnSignal(Exception):
    """Internal control-flow exception used to implement `return`."""
    def __init__(self, value: Any):
        self.value = value

class ShakarBreakSignal(Exception):
    """Internal control flow for `break`."""

class ShakarContinueSignal(Exception):
    """Internal control flow for `continue`."""

# ---------- Built-in method registry ----------

@dataclass(frozen=True)
class StdlibFunction:
    fn: Callable[['Frame', List[Any]], Any]
    arity: Optional[int] = None

class Builtins:
    array_methods: Dict[str, Callable[['Frame', 'ShkArray', List[Any]], Any]] = {}
    string_methods: Dict[str, Callable[['Frame', 'ShkString', List[Any]], Any]] = {}
    object_methods: Dict[str, Callable[['Frame', 'ShkObject', List[Any]], Any]] = {}
    stdlib_functions: Dict[str, StdlibFunction] = {}

_STDLIB_INITIALIZED = False

def init_stdlib() -> None:
    """Load stdlib modules (idempotent) so register_stdlib hooks run."""
    global _STDLIB_INITIALIZED

    if _STDLIB_INITIALIZED:
        return

    for module_name in ("rt.shakar_stdlib", "shakar_stdlib"):
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

def register_array(name: str):
    def dec(fn):
        Builtins.array_methods[name] = fn
        return fn

    return dec

def register_string(name: str):
    def dec(fn):
        Builtins.string_methods[name] = fn
        return fn

    return dec

def register_object(name: str):
    def dec(fn):
        Builtins.object_methods[name] = fn
        return fn

    return dec

def register_stdlib(name: str, *, arity: int | None = None):
    def dec(fn: Callable[['Frame', List[Any]], Any]):
        Builtins.stdlib_functions[name] = StdlibFunction(fn=fn, arity=arity)
        return fn

    return dec

def _string_expect_arity(method: str, args: List[Any], expected: int) -> None:
    if len(args) != expected:
        raise ShakarArityError(f"string.{method} expects {expected} argument(s); got {len(args)}")

def _string_arg(method: str, arg: Any) -> str:
    if isinstance(arg, ShkString):
        return arg.value

    raise ShakarTypeError(f"string.{method} expects a string argument")

@register_string("trim")
def _string_trim(_frame: Frame, recv: ShkString, args: List[Any]) -> ShkString:
    _string_expect_arity("trim", args, 0)

    return ShkString(recv.value.strip())

@register_string("lower")
def _string_lower(_frame: Frame, recv: ShkString, args: List[Any]) -> ShkString:
    _string_expect_arity("lower", args, 0)

    return ShkString(recv.value.lower())

@register_string("upper")
def _string_upper(_frame: Frame, recv: ShkString, args: List[Any]) -> ShkString:
    _string_expect_arity("upper", args, 0)

    return ShkString(recv.value.upper())

@register_string("hasPrefix")
def _string_has_prefix(_frame: Frame, recv: ShkString, args: List[Any]) -> ShkBool:
    _string_expect_arity("hasPrefix", args, 1)
    prefix = _string_arg("hasPrefix", args[0])

    return ShkBool(recv.value.startswith(prefix))

@register_string("hasSuffix")
def _string_has_suffix(_frame: Frame, recv: ShkString, args: List[Any]) -> ShkBool:
    _string_expect_arity("hasSuffix", args, 1)
    suffix = _string_arg("hasSuffix", args[0])

    return ShkBool(recv.value.endswith(suffix))

@register_string("isAscii")
def _string_is_ascii(_frame: Frame, recv: ShkString, args: List[Any]) -> ShkBool:
    _string_expect_arity("isAscii", args, 0)

    return ShkBool(recv.value.isascii())

def call_builtin_method(recv: Any, name: str, args: List[Any], frame: 'Frame') -> Any:
    if isinstance(recv, ShkArray):
        fn = Builtins.array_methods.get(name)
        if fn: return fn(frame, recv, args)

    if isinstance(recv, ShkString):
        fn = Builtins.string_methods.get(name)
        if fn: return fn(frame, recv, args)

    if isinstance(recv, ShkObject):
        fn = Builtins.object_methods.get(name)
        if fn: return fn(frame, recv, args)

    raise ShakarMethodNotFound(recv, name)

def call_shkfn(fn: ShkFn, positional: List[Any], subject: Any, caller_frame: 'Frame') -> Any:
    """
    Subjectful call semantics:
    - subject is available to callee as frame.dot
    - If fn.params is None, treat as unary subject-lambda: ignore positional, evaluate body with dot bound.
    - Else: arity must match len(fn.params); bind params; dot=subject.
    """

    if fn.decorators:
        return _call_shkfn_with_decorators(fn, positional, subject, caller_frame)

    return _call_shkfn_raw(fn, positional, subject, caller_frame)

def _call_shkfn_raw(fn: ShkFn, positional: List[Any], subject: Any, caller_frame: 'Frame') -> Any:
    _ = caller_frame
    from shakar_eval import eval_node  # local import to avoid cycle

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
            return eval_node(fn.body, callee_frame)
        except ShakarReturnSignal as signal:
            return signal.value

    callee_frame = Frame(parent=fn.frame, dot=subject)

    if len(positional) != len(fn.params):
        raise ShakarArityError(f"Function expects {len(fn.params)} args; got {len(positional)}")

    for name, val in zip(fn.params, positional):
        callee_frame.define(name, val)

    callee_frame.mark_function_frame()

    try:
        return eval_node(fn.body, callee_frame)
    except ShakarReturnSignal as signal:
        return signal.value

def _call_shkfn_with_decorators(fn: ShkFn, positional: List[Any], subject: Any, caller_frame: 'Frame') -> Any:
    chain = fn.decorators or ()
    args = ShkArray(list(positional))

    return _run_decorator_chain(fn, chain, 0, args, subject, caller_frame)

def _run_decorator_chain(
    fn: ShkFn,
    chain: Tuple[DecoratorConfigured, ...],
    index: int,
    args_value: ShkArray,
    subject: Any,
    caller_frame: 'Frame',
) -> Any:
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
    subject: Any,
    caller_frame: 'Frame',
) -> Any:
    from shakar_eval import eval_node  # defer to avoid cycle
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

def _coerce_decorator_args(value: Any) -> ShkArray:
    if isinstance(value, ShkArray):
        return value

    raise ShakarTypeError("Decorator args must be an array value")
