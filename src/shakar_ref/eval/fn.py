from __future__ import annotations

from typing import Any, List

from lark import Tree

from ..runtime import DecoratorConfigured, Frame, ShkDecorator, ShkFn, ShkNull, ShakarRuntimeError, ShakarTypeError
from ..tree import is_tree_node, tree_children, tree_label
from .common import expect_ident_token as _expect_ident_token, ident_token_value as _ident_token_value

def extract_param_names(params_node: Any, context: str="parameter list") -> List[str]:
    if params_node is None:
        return []

    names: List[str] = []

    for p in tree_children(params_node):
        name = _ident_token_value(p)

        if name is not None:
            names.append(name)
            continue
        if getattr(p, "type", None) == 'COMMA':
            continue

        raise ShakarRuntimeError(f"Unsupported parameter node in {context}: {p}")

    return names

def eval_fn_def(children: List[Any], frame: Frame, eval_func) -> Any:
    if not children:
        raise ShakarRuntimeError("Malformed function definition")

    name = _expect_ident_token(children[0], "Function name")
    params_node = None
    body_node = None
    decorators_node = None

    for node in children[1:]:
        if params_node is None and is_tree_node(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif body_node is None and is_tree_node(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node
        elif decorators_node is None and is_tree_node(node) and tree_label(node) == 'decorator_list':
            decorators_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params = extract_param_names(params_node, context="function definition")
    fn_value = ShkFn(params=params, body=body_node, frame=Frame(parent=frame, dot=None))

    if decorators_node is not None:
        instances = evaluate_decorator_list(decorators_node, frame, eval_func)
        if instances:
            fn_value.decorators = tuple(reversed(instances))

    frame.define(name, fn_value)

    return ShkNull()

def eval_decorator_def(children: List[Any], frame: Frame) -> Any:
    if not children:
        raise ShakarRuntimeError("Malformed decorator definition")

    name = _expect_ident_token(children[0], "Decorator name")
    params_node = None
    body_node = None

    for node in children[1:]:
        if params_node is None and is_tree_node(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif body_node is None and is_tree_node(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params = extract_param_names(params_node, context="decorator definition")
    decorator = ShkDecorator(params=params, body=body_node, frame=Frame(parent=frame, dot=None))
    frame.define(name, decorator)

    return ShkNull()

def eval_anonymous_fn(children: List[Any], frame: Frame) -> ShkFn:
    params_node = None
    body_node = None

    for node in children:
        if is_tree_node(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif is_tree_node(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params = extract_param_names(params_node, context="anonymous function")

    return ShkFn(params=params, body=body_node, frame=Frame(parent=frame, dot=None))

def eval_amp_lambda(n: Tree, frame: Frame) -> ShkFn:
    if len(n.children) == 1:
        return ShkFn(params=None, body=n.children[0], frame=Frame(parent=frame, dot=None), kind="amp")

    if len(n.children) == 2:
        params_node, body = n.children
        params = extract_param_names(params_node, context="amp_lambda")
        return ShkFn(params=params, body=body, frame=Frame(parent=frame, dot=None), kind="amp")

    raise ShakarRuntimeError("amp_lambda malformed")

def evaluate_decorator_list(node: Tree, frame: Frame, eval_func) -> List[DecoratorConfigured]:
    configured: List[DecoratorConfigured] = []

    for entry in tree_children(node):
        kids = tree_children(entry)
        expr_node = kids[0] if kids else None

        if expr_node is None:
            continue

        value = eval_func(expr_node, frame)
        configured.append(_coerce_decorator_instance(value))

    return configured

def _coerce_decorator_instance(value: Any) -> DecoratorConfigured:
    match value:
        case DecoratorConfigured():
            return DecoratorConfigured(decorator=value.decorator, args=list(value.args))
        case ShkDecorator(params=params):
            arity = len(params) if params is not None else 0

            if arity:
                raise ShakarRuntimeError("Decorator requires arguments; call it with parentheses")
            return DecoratorConfigured(decorator=value, args=[])
        case _:
            raise ShakarTypeError("Decorator expression must evaluate to a decorator")

