from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..tree import Tree, Tok
from ..tree import Node
from ..token_types import TT

from ..runtime import DecoratorConfigured, Frame, ShkDecorator, ShkFn, ShkNull, ShkValue, ShakarRuntimeError, ShakarTypeError
from ..tree import is_tree, is_token, tree_children, tree_label, child_by_label
from .common import expect_ident_token as _expect_ident_token, ident_token_value as _ident_token_value, extract_param_names

EvalFunc = Callable[[Node, Frame], ShkValue]

def extract_param_contracts(params_node: Optional[Node]) -> Dict[str, Node]:
    """Extract parameter contracts from function definition"""
    if params_node is None:
        return {}

    contracts: Dict[str, Node] = {}

    for p in tree_children(params_node):
        if not is_tree(p) or tree_label(p) != 'param':
            continue

        children = tree_children(p)
        if len(children) < 2:
            continue

        param_name = _ident_token_value(children[0])
        if param_name is None:
            continue

        contract_node = child_by_label(p, 'contract')
        if contract_node is not None:
            contract_children = tree_children(contract_node)
            if contract_children:
                contracts[param_name] = contract_children[0]

    return contracts

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
            if params_node is None and label == 'paramlist':
                params_node = node
            elif body_node is None and label in {'inlinebody', 'indentblock'}:
                body_node = node
            elif return_contract_node is None and label == 'return_contract':
                return_contract_node = node
            elif decorators_node is None and label == 'decorator_list':
                decorators_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params, varargs = extract_param_names(params_node, context="function definition")
    contracts = extract_param_contracts(params_node)

    # Extract return contract expression if present
    return_contract_expr = None
    if return_contract_node is not None:
        contract_children = tree_children(return_contract_node)
        if contract_children:
            return_contract_expr = contract_children[0]

    final_body = _inject_contract_assertions(body_node, contracts) if contracts else body_node
    fn_value = ShkFn(
        params=params,
        body=final_body,
        frame=Frame(parent=frame, dot=None),
        return_contract=return_contract_expr,
        vararg_indices=varargs,
    )

    if decorators_node is not None:
        instances = evaluate_decorator_list(decorators_node, frame, eval_func)
        if instances:
            fn_value.decorators = tuple(reversed(instances))

    frame.define(name, fn_value)

    return ShkNull()

def _inject_contract_assertions(body: Node, contracts: Dict[str, Node]) -> Node:
    """Inject assert statements for parameter contracts at the start of the function body"""
    if not contracts:
        return body

    assertions: List[Node] = []
    for param_name, contract_expr in contracts.items():
        assertion = Tree('assert', [
            Tree('compare', [
                Tok(TT.IDENT, param_name, 0, 0),
                Tree('cmpop', [Tok(TT.TILDE, '~', 0, 0)]),
                contract_expr
            ])
        ])
        assertions.append(assertion)

    body_label = tree_label(body)

    if body_label is None:
        raise ShakarRuntimeError("Malformed function body")

    # For inline bodies, we need to convert to an indentblock to preserve semantics
    # because inlinebody only evaluates the first child
    if body_label == 'inlinebody':
        body_children = tree_children(body)
        new_children = assertions + list(body_children)
        return Tree('indentblock', new_children)

    # For indentblock, just prepend assertions
    body_children = tree_children(body)
    new_children = assertions + list(body_children)
    return Tree(body_label, new_children)

def eval_decorator_def(children: List[Node], frame: Frame) -> ShkValue:
    if not children:
        raise ShakarRuntimeError("Malformed decorator definition")

    name = _expect_ident_token(children[0], "Decorator name")
    params_node = None
    body_node = None

    for node in children[1:]:
        if params_node is None and is_tree(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif body_node is None and is_tree(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params, varargs = extract_param_names(params_node, context="decorator definition")
    decorator = ShkDecorator(params=params, body=body_node, frame=Frame(parent=frame, dot=None), vararg_indices=varargs)
    frame.define(name, decorator)

    return ShkNull()

def eval_anonymous_fn(children: List[Node], frame: Frame) -> ShkFn:
    params_node = None
    body_node = None
    return_contract_node = None

    for node in children:
        if is_tree(node) and tree_label(node) == 'paramlist':
            params_node = node
        elif is_tree(node) and tree_label(node) in {'inlinebody', 'indentblock'}:
            body_node = node
        elif is_tree(node) and tree_label(node) == 'return_contract':
            return_contract_node = node

    if body_node is None:
        body_node = Tree('inlinebody', [])

    params, varargs = extract_param_names(params_node, context="anonymous function")
    contracts = extract_param_contracts(params_node)

    # Extract return contract expression if present
    return_contract_expr = None
    if return_contract_node is not None:
        contract_children = tree_children(return_contract_node)
        if contract_children:
            return_contract_expr = contract_children[0]

    final_body = _inject_contract_assertions(body_node, contracts) if contracts else body_node

    return ShkFn(
        params=params,
        body=final_body,
        frame=Frame(parent=frame, dot=None),
        return_contract=return_contract_expr,
        vararg_indices=varargs,
    )

def eval_amp_lambda(n: Tree, frame: Frame) -> ShkFn:
    if len(n.children) == 1:
        return ShkFn(params=None, body=n.children[0], frame=Frame(parent=frame, dot=None), kind="amp")

    if len(n.children) == 2:
        params_node, body = n.children
        params, varargs = extract_param_names(params_node, context="amp_lambda")
        return ShkFn(params=params, body=body, frame=Frame(parent=frame, dot=None), kind="amp", vararg_indices=varargs)

    raise ShakarRuntimeError("amp_lambda malformed")

def evaluate_decorator_list(node: Tree, frame: Frame, eval_func: EvalFunc) -> List[DecoratorConfigured]:
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
        case ShkDecorator(params=params, vararg_indices=varargs):
            param_list = params or []
            vararg_list = varargs or []
            min_arity = len(param_list) - len(vararg_list)

            if min_arity > 0:
                raise ShakarRuntimeError("Decorator requires arguments; call it with parentheses")
            return DecoratorConfigured(decorator=value, args=[])
        case _:
            raise ShakarTypeError("Decorator expression must evaluate to a decorator")
