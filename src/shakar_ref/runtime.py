from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from typing_extensions import Protocol, TypeAlias, TypeGuard

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
    items: List['ShkValue']
    def __repr__(self) -> str:
        return "[" + ", ".join(repr(x) for x in self.items) + "]"

@dataclass
class ShkCommand:
    segments: List[str]
    def render(self) -> str:
        return "".join(self.segments)
    def __repr__(self) -> str:
        return f"sh<{self.render()}>"

@dataclass
class ShkObject:
    slots: Dict[str, 'ShkValue']
    def __repr__(self) -> str:
        pairs = []

        for k, v in self.slots.items():
            pairs.append(f"{k}: {repr(v)}")

        return "{ " + ", ".join(pairs) + " }"

@dataclass
class SelectorIndex:
    value: 'ShkValue'

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
    args: List['ShkValue']

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
            param_desc = ", ".join(self.params) if self.params else "nullary"

        return f"<{label} params={param_desc} body={body_label}>"

@dataclass
class BoundMethod:
    fn: ShkFn
    subject: 'ShkValue'

@dataclass
class BuiltinMethod:
    name: str
    subject: 'ShkValue'

@dataclass
class Descriptor:
    getter: ShkFn | None = None
    setter: ShkFn | None = None

@dataclass
class DecoratorContinuation:
    fn: ShkFn
    decorators: Tuple[DecoratorConfigured, ...]
    index: int
    subject: 'ShkValue | None'
    caller_frame: 'Frame'

    def invoke(self, args_value: ShkValue) -> 'ShkValue':
        args = _coerce_decorator_args(args_value)
        return _run_decorator_chain(self.fn, self.decorators, self.index, args, self.subject, self.caller_frame)

@dataclass
class DeferEntry:
    """Scheduled defer thunk plus optional metadata for dependency ordering."""
    thunk: Callable[[], None]
    label: Optional[str] = None
    deps: List[str] = field(default_factory=list)

StdlibFn = Callable[['Frame', List['ShkValue']], 'ShkValue']

@dataclass(frozen=True)
class StdlibFunction:
    fn: StdlibFn
    arity: Optional[int] = None

ShkValue: TypeAlias = (
    ShkNull
    | ShkNumber
    | ShkString
    | ShkBool
    | ShkArray
    | ShkObject
    | ShkFn
    | ShkDecorator
    | DecoratorConfigured
    | DecoratorContinuation
    | Descriptor
    | ShkSelector
    | ShkCommand
    | BoundMethod
    | BuiltinMethod
    | StdlibFunction
)

DotValue: TypeAlias = ShkValue | None

class Frame:
    def __init__(self, parent: Optional['Frame']=None, dot: DotValue=None, source: Optional[str]=None):
        self.parent = parent
        self.vars: Dict[str, ShkValue] = {}
        self.dot: DotValue = dot
        self._defer_stack: List[List[DeferEntry]] = []
        self._is_function_frame = False
        self._active_error: Optional[ShakarRuntimeError] = None
        self.source: Optional[str]

        if parent is None and Builtins.stdlib_functions:
            for name, std in Builtins.stdlib_functions.items():
                self.vars[name] = std

        if source is not None:
            self.source = source
        elif parent is not None and hasattr(parent, 'source'):
            self.source = parent.source
        else:
            self.source = None

    def define(self, name: str, val: ShkValue) -> None:
        self.vars[name] = val

    def get(self, name: str) -> ShkValue:
        if name in self.vars:
            return self.vars[name]

        if self.parent is not None:
            return self.parent.get(name)

        raise ShakarRuntimeError(f"Name '{name}' not found")

    def set(self, name: str, val: ShkValue) -> None:
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
    shk_type: str | None
    shk_data: Any
    shk_payload: Any

    def __init__(self, message: str):
        super().__init__(message)
        self.shk_type = None
        self.shk_data = None
        self.shk_payload = None

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
    def __init__(self, recv: ShkValue, name: str):
        super().__init__(f"{type(recv).__name__} has no builtin method '{name}'")
        self.receiver = recv
        self.name = name

class CommandError(ShakarRuntimeError):
    def __init__(self, cmd: str, code: int, stdout: str, stderr: str):
        super().__init__(f"Command failed with exit code {code}: {cmd}")
        self.cmd = cmd
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self.shk_payload = ShkObject({
            "cmd": ShkString(cmd),
            "code": ShkNumber(float(code)),
            "stdout": ShkString(stdout),
            "stderr": ShkString(stderr),
        })
        self.shk_type = "CommandError"

class ShakarAssertionError(ShakarRuntimeError):
    pass

class ShakarReturnSignal(Exception):
    """Internal control-flow exception used to implement `return`."""
    def __init__(self, value: ShkValue):
        self.value = value

class ShakarBreakSignal(Exception):
    """Internal control flow for `break`."""

class ShakarContinueSignal(Exception):
    """Internal control flow for `continue`."""

_SHK_VALUE_TYPES: Tuple[type[Any], ...] = (
    ShkNull,
    ShkNumber,
    ShkString,
    ShkBool,
    ShkArray,
    ShkObject,
    ShkFn,
    ShkDecorator,
    DecoratorConfigured,
    DecoratorContinuation,
    Descriptor,
    ShkSelector,
    ShkCommand,
    BoundMethod,
    BuiltinMethod,
    StdlibFunction,
)

def is_shk_value(value: Any) -> TypeGuard[ShkValue]:
    return isinstance(value, _SHK_VALUE_TYPES)

def _ensure_shk_value(value: Any) -> ShkValue:
    if value is None:
        return ShkNull()
    if is_shk_value(value):
        return value
    raise ShakarTypeError(f"Unexpected value type {type(value).__name__}")

R_contra = TypeVar("R_contra", bound="ShkValue", contravariant=True)

class Method(Protocol[R_contra]):
    def __call__(self, frame: 'Frame', recv: R_contra, args: List['ShkValue']) -> 'ShkValue': ...

MethodRegistry = Dict[str, Method[Any]]

class Builtins:
    array_methods: MethodRegistry = {}
    string_methods: MethodRegistry = {}
    object_methods: MethodRegistry = {}
    command_methods: MethodRegistry = {}
    stdlib_functions: Dict[str, StdlibFunction] = {}

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

def register_stdlib(name: str, *, arity: int | None = None):
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

def call_shkfn(fn: ShkFn, positional: List[ShkValue], subject: ShkValue | None, caller_frame: 'Frame') -> ShkValue:
    """
    Subjectful call semantics:
    - subject is available to callee as frame.dot
    - If fn.params is None, treat as unary subject-lambda: ignore positional, evaluate body with dot bound.
    - Else: arity must match len(fn.params); bind params; dot=subject.
    """

    if fn.decorators:
        return _call_shkfn_with_decorators(fn, positional, subject, caller_frame)

    return _call_shkfn_raw(fn, positional, subject, caller_frame)

def _call_shkfn_raw(fn: ShkFn, positional: List[ShkValue], subject: ShkValue | None, caller_frame: 'Frame') -> ShkValue:
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

def _call_shkfn_with_decorators(fn: ShkFn, positional: List[ShkValue], subject: ShkValue | None, caller_frame: 'Frame') -> ShkValue:
    chain = fn.decorators or ()
    args = ShkArray(list(positional))

    return _run_decorator_chain(fn, chain, 0, args, subject, caller_frame)

def _run_decorator_chain(
    fn: ShkFn,
    chain: Tuple[DecoratorConfigured, ...],
    index: int,
    args_value: ShkArray,
    subject: ShkValue | None,
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
    subject: ShkValue | None,
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
