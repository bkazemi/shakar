from __future__ import annotations

from typing import Any, Callable, List

from lark import Tree

from ..runtime import (
    BoundMethod,
    BuiltinMethod,
    DecoratorConfigured,
    DecoratorContinuation,
    Frame,
    ShkDecorator,
    ShkFn,
    ShkSelector,
    SelectorIndex,
    ShakarArityError,
    ShakarMethodNotFound,
    ShakarRuntimeError,
    ShakarTypeError,
    StdlibFunction,
    call_builtin_method,
    call_shkfn,
)
from ..tree import child_by_label, is_token_node, is_tree_node, tree_children, tree_label
from .bind import FanContext, RebindContext, apply_fan_op
from .common import expect_ident_token as _expect_ident_token
from .mutation import get_field_value, index_value, slice_value
from .selector import evaluate_selectorlist, apply_selectors_to_value

EvalFunc = Callable[[Any, Frame], Any]

def eval_args_node(args_node: Any, frame: Frame, eval_func: EvalFunc) -> List[Any]:
    def label(node: Tree) -> str | None:
        return tree_label(node)

    def flatten(node: Any) -> List[Any]:
        if is_tree_node(node):
            tag = label(node)

            if tag in {'args', 'arglist', 'arglistnamedmixed'}:
                out: List[Any] = []

                for ch in node.children:
                    out.extend(flatten(ch))
                return out

            if tag in {'argitem', 'arg'} and node.children:
                return flatten(node.children[0])

            if tag == 'namedarg' and node.children:
                return flatten(node.children[-1])

        return [node]

    if is_tree_node(args_node):
        return [eval_func(n, frame) for n in flatten(args_node)]

    if isinstance(args_node, list):
        res: List[Any] = []
        for n in args_node:
            res.extend(flatten(n))
        return [eval_func(n, frame) for n in res]

    return []

def evaluate_index_operand(index_node: Tree, frame: Frame, eval_func: EvalFunc) -> Any:
    selectorlist = child_by_label(index_node, 'selectorlist')

    if selectorlist is not None:
        selectors = evaluate_selectorlist(selectorlist, frame, eval_func)

        if len(selectors) == 1 and isinstance(selectors[0], SelectorIndex):
            return selectors[0].value

        return ShkSelector(selectors)

    expr_node = _index_expr_from_children(index_node.children)
    return eval_func(expr_node, frame)

def apply_slice(recv: Any, arms: List[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    def arm_to_py(node: Any) -> int | None:
        if tree_label(node) == 'emptyexpr':
            return None

        value = eval_func(node, frame)
        if hasattr(value, "value"):
            return int(value.value)
        return None

    start, stop, step = map(arm_to_py, arms)

    return slice_value(recv, start, stop, step)

def apply_index_operation(recv: Any, op: Tree, frame: Frame, eval_func: EvalFunc) -> Any:
    selectorlist = child_by_label(op, 'selectorlist')

    if selectorlist is None:
        expr_node = _index_expr_from_children(op.children)
        idx_val = eval_func(expr_node, frame)
        return index_value(recv, idx_val, frame)

    selectors = evaluate_selectorlist(selectorlist, frame, eval_func)

    if len(selectors) == 1 and isinstance(selectors[0], SelectorIndex):
        return index_value(recv, selectors[0].value, frame)

    return apply_selectors_to_value(recv, selectors)

def apply_op(recv: Any, op: Tree, frame: Frame, eval_func: EvalFunc) -> Any:
    if isinstance(recv, FanContext):
        return apply_fan_op(recv, op, frame, apply_op=apply_op, eval_func=eval_func)

    context = None

    if isinstance(recv, RebindContext):
        context = recv
        recv = context.value

    d = op.data

    if d in {'field', 'fieldsel'}:
        field_name = _expect_ident_token(op.children[0], "Field access")
        result = get_field_value(recv, field_name, frame)
    elif d == 'index':
        result = apply_index_operation(recv, op, frame, eval_func)
    elif d == 'slicesel':
        result = apply_slice(recv, op.children, frame, eval_func)
    elif d == 'fieldfan':
        return apply_fan_op(recv, op, frame, apply_op=apply_op, eval_func=eval_func)
    elif d == 'call':
        args = eval_args_node(op.children[0] if op.children else None, frame, eval_func)
        result = call_value(recv, args, frame, eval_func)
    elif d == 'method':
        method_name = _expect_ident_token(op.children[0], "Method call")
        args = eval_args_node(op.children[1] if len(op.children) > 1 else None, frame, eval_func)

        try:
            result = call_builtin_method(recv, method_name, args, frame)
        except ShakarMethodNotFound:
            cal = get_field_value(recv, method_name, frame)

            if isinstance(cal, BoundMethod):
                result = call_shkfn(cal.fn, args, subject=cal.subject, caller_frame=frame)
            elif isinstance(cal, ShkFn):
                result = call_shkfn(cal, args, subject=recv, caller_frame=frame)
            else:
                raise
    else:
        raise ShakarRuntimeError(f"Unknown chain op: {d}")

    if context is not None:
        context.value = result

        if not isinstance(result, (BuiltinMethod, BoundMethod, ShkFn)):
            context.setter(result)
        return context

    return result

def call_value(cal: Any, args: List[Any], frame: Frame, eval_func: EvalFunc) -> Any:
    match cal:
        case BoundMethod(fn=fn, subject=subject):
            return call_shkfn(fn, args, subject=subject, caller_frame=frame)
        case BuiltinMethod(name=name, subject=subject):
            return call_builtin_method(subject, name, args, frame)
        case StdlibFunction(fn=fn, arity=arity):
            if arity is not None and len(args) != arity:
                raise ShakarArityError(f"Function expects {arity} args; got {len(args)}")
            return fn(frame, args)
        case DecoratorContinuation():
            if len(args) != 1:
                raise ShakarArityError("Decorator continuation expects exactly 1 argument (the args array)")
            return cal.invoke(args[0])
        case ShkDecorator():
            params = cal.params or []

            if len(args) != len(params):
                raise ShakarArityError(f"Decorator expects {len(params)} args; got {len(args)}")
            return DecoratorConfigured(decorator=cal, args=list(args))
        case ShkFn():
            return call_shkfn(cal, args, subject=None, caller_frame=frame)
        case _:
            raise ShakarTypeError(f"Cannot call value of type {type(cal).__name__}")

def _index_expr_from_children(children: List[Any]) -> Any:
    queue = list(children)

    while queue:
        node = queue.pop(0)

        if is_token_node(node):
            continue

        if not is_tree_node(node):
            return node

        tag = tree_label(node)
        if tag in {'selectorlist', 'selector', 'indexsel'}:
            queue.extend(node.children)
            continue

        return node

    raise ShakarRuntimeError("Malformed index expression")
