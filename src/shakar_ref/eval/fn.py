from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..tree import Tree, Tok
from ..tree import Node
from ..token_types import TT

from ..runtime import (
    DecoratorConfigured,
    Frame,
    ShkDecorator,
    ShkFn,
    ShkNil,
    ShkValue,
    ShakarRuntimeError,
    ShakarTypeError,
)
from ..tree import (
    Tree,
    is_tree,
    tree_children,
    tree_label,
    child_by_label,
    is_token,
    is_inline_body,
)
from .common import (
    expect_ident_token as _expect_ident_token,
    extract_function_signature,
    token_string,
)
from .helpers import closure_frame

EvalFunc = Callable[[Node, Frame], ShkValue]
_DECORATOR_RESERVED_BINDINGS = frozenset({"f", "args"})


def eval_hook_stmt(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    children = tree_children(n)
    if len(children) != 2:
        raise ShakarRuntimeError("Malformed hook statement")

    name_tok = children[0]
    if not is_token(name_tok):
        raise ShakarRuntimeError("Hook name must be a string token")

    # Parse the string token to get the actual name (handles raw strings, etc.)
    # token_string expects (token, frame) but frame is unused for string literals
    token_string(name_tok, None)

    # We evaluate the handler (amp_lambda) to ensure it's valid,
    # but we don't execute it or register it yet.
    handler_node = children[1]
    _ = eval_func(handler_node, frame)

    return ShkNil()


def eval_fn_def(children: List[Node], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not children:
        raise ShakarRuntimeError("Malformed function definition")

    name = _expect_ident_token(children[0], "Function name")
    params_node = None
    body_node = None
    return_contract_node = None
    decorators_node = None

    for node in children[1:]:
        if is_tree(node):
            label = tree_label(node)
            if params_node is None and label == "paramlist":
                params_node = node
            elif body_node is None and label == "body":
                body_node = node
            elif return_contract_node is None and label == "return_contract":
                return_contract_node = node
            elif decorators_node is None and label == "decorator_list":
                decorators_node = node

    if body_node is None:
        body_node = Tree("body", [], attrs={"inline": True})

    params, varargs, defaults, contracts, spread_contracts = extract_function_signature(
        params_node, context="function definition"
    )

    # Extract return contract expression if present
    return_contract_expr = None
    if return_contract_node is not None:
        contract_children = tree_children(return_contract_node)
        if contract_children:
            return_contract_expr = contract_children[0]

    final_body = (
        _inject_contract_assertions(body_node, contracts, spread_contracts)
        if contracts or spread_contracts
        else body_node
    )
    fn_value = ShkFn(
        params=params,
        body=final_body,
        frame=closure_frame(frame),
        return_contract=return_contract_expr,
        vararg_indices=varargs,
        param_defaults=defaults,
        name=name,
    )

    if decorators_node is not None:
        instances = evaluate_decorator_list(decorators_node, frame, eval_func)
        if instances:
            fn_value.decorators = tuple(reversed(instances))

    frame.define(name, fn_value)

    return ShkNil()


def _inject_contract_assertions(
    body: Node,
    contracts: Dict[str, Node],
    spread_contracts: Dict[str, Node],
) -> Node:
    """Inject assert statements for parameter contracts at the start of the function body"""
    assertions = _build_contract_assertions(contracts, spread_contracts)

    if not assertions:
        return body

    body_label = tree_label(body)

    if body_label is None:
        raise ShakarRuntimeError("Malformed function body")

    # For inline bodies, we need to convert to block to preserve semantics
    # because inline body only evaluates the first child
    if is_inline_body(body):
        body_children = tree_children(body)
        new_children = assertions + list(body_children)
        return Tree("body", new_children, attrs={"inline": False})

    # For block body, just prepend assertions
    body_children = tree_children(body)
    new_children = assertions + list(body_children)
    return Tree("body", new_children, attrs={"inline": False})


def _build_contract_assertions(
    contracts: Dict[str, Node],
    spread_contracts: Dict[str, Node],
) -> List[Node]:
    assertions: List[Node] = []

    for param_name, contract_expr in contracts.items():
        assertion = Tree(
            "assert",
            [
                Tree(
                    "compare",
                    [
                        Tok(TT.IDENT, param_name, 0, 0),
                        Tree("cmpop", [Tok(TT.TILDE, "~", 0, 0)]),
                        contract_expr,
                    ],
                )
            ],
        )
        assertions.append(assertion)

    for param_name, contract_expr in spread_contracts.items():
        assertion = Tree(
            "assert",
            [
                Tree(
                    "compare",
                    [
                        Tree("subject", [Tok(TT.DOT, ".", 0, 0)]),
                        Tree("cmpop", [Tok(TT.TILDE, "~", 0, 0)]),
                        contract_expr,
                    ],
                )
            ],
        )
        loop_body = Tree("body", [assertion], attrs={"inline": False})
        loop = Tree(
            "forsubject",
            [
                Tok(TT.FOR, "for", 0, 0),
                Tok(TT.IDENT, param_name, 0, 0),
                loop_body,
            ],
        )
        assertions.append(loop)

    return assertions


def eval_decorator_def(children: List[Node], frame: Frame) -> ShkValue:
    if not children:
        raise ShakarRuntimeError("Malformed decorator definition")

    name = _expect_ident_token(children[0], "Decorator name")
    params_node = None
    body_node = None

    for node in children[1:]:
        if params_node is None and is_tree(node) and tree_label(node) == "paramlist":
            params_node = node
        elif body_node is None and is_tree(node) and tree_label(node) == "body":
            body_node = node

    if body_node is None:
        body_node = Tree("body", [], attrs={"inline": True})

    params, varargs, defaults, _contracts, _spread_contracts = (
        extract_function_signature(params_node, context="decorator definition")
    )
    for param_name in params:
        if param_name in _DECORATOR_RESERVED_BINDINGS:
            raise ShakarRuntimeError(
                f"Decorator parameter '{param_name}' is reserved for decorator context"
            )

    decorator = ShkDecorator(
        params=params,
        body=body_node,
        frame=closure_frame(frame),
        vararg_indices=varargs,
        param_defaults=defaults,
    )
    frame.define(name, decorator)

    return ShkNil()


def eval_anonymous_fn(children: List[Node], frame: Frame) -> ShkFn:
    params_node = None
    body_node = None
    return_contract_node = None

    for node in children:
        if is_tree(node) and tree_label(node) == "paramlist":
            params_node = node
        elif is_tree(node) and tree_label(node) == "body":
            body_node = node
        elif is_tree(node) and tree_label(node) == "return_contract":
            return_contract_node = node

    if body_node is None:
        body_node = Tree("body", [], attrs={"inline": True})

    params, varargs, defaults, contracts, spread_contracts = extract_function_signature(
        params_node, context="anonymous function"
    )

    # Extract return contract expression if present
    return_contract_expr = None
    if return_contract_node is not None:
        contract_children = tree_children(return_contract_node)
        if contract_children:
            return_contract_expr = contract_children[0]

    final_body = (
        _inject_contract_assertions(body_node, contracts, spread_contracts)
        if contracts or spread_contracts
        else body_node
    )

    return ShkFn(
        params=params,
        body=final_body,
        frame=closure_frame(frame),
        return_contract=return_contract_expr,
        vararg_indices=varargs,
        param_defaults=defaults,
        name=None,
    )


def eval_amp_lambda(n: Tree, frame: Frame) -> ShkFn:
    if len(n.children) == 1:
        return ShkFn(
            params=None,
            body=n.children[0],
            frame=closure_frame(frame),
            kind="amp",
            name=None,
        )

    if len(n.children) == 2:
        params_node, body = n.children
        params, varargs, defaults, contracts, spread_contracts = (
            extract_function_signature(params_node, context="amp_lambda")
        )
        if contracts or spread_contracts:
            assertions = _build_contract_assertions(contracts, spread_contracts)
            body = Tree("body", assertions + [body], attrs={"inline": False})
        return ShkFn(
            params=params,
            body=body,
            frame=closure_frame(frame),
            kind="amp",
            vararg_indices=varargs,
            param_defaults=defaults,
            name=None,
        )

    raise ShakarRuntimeError("amp_lambda malformed")


def evaluate_decorator_list(
    node: Tree, frame: Frame, eval_func: EvalFunc
) -> List[DecoratorConfigured]:
    configured: List[DecoratorConfigured] = []

    for entry in tree_children(node):
        kids = tree_children(entry)
        expr_node = kids[0] if kids else None

        if expr_node is None:
            continue

        value = eval_func(expr_node, frame)
        configured.append(_coerce_decorator_instance(value))

    return configured


def _coerce_decorator_instance(value: ShkValue) -> DecoratorConfigured:
    match value:
        case DecoratorConfigured():
            return DecoratorConfigured(decorator=value.decorator, args=list(value.args))
        case ShkDecorator(
            params=params, vararg_indices=varargs, param_defaults=defaults
        ):
            param_list = params or []
            vararg_list = varargs or []
            default_list = defaults or []
            if len(default_list) < len(param_list):
                default_list = default_list + [None] * (
                    len(param_list) - len(default_list)
                )

            spread_index = vararg_list[0] if vararg_list else None
            required = 0
            for idx in range(len(param_list)):
                if spread_index is not None and idx == spread_index:
                    continue
                if default_list[idx] is None:
                    required += 1

            if required > 0:
                raise ShakarRuntimeError(
                    "Decorator requires arguments; call it with parentheses"
                )
            return DecoratorConfigured(decorator=value, args=[])
        case _:
            raise ShakarTypeError("Decorator expression must evaluate to a decorator")
