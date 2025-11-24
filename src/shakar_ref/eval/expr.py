from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Set

from lark import Token

from ..runtime import (
    Frame,
    ShkArray,
    ShkBool,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkValue,
    ShkSelector,
    ShkString,
    ShakarKeyError,
    ShakarIndexError,
    ShakarRuntimeError,
    ShakarTypeError,
)
from ..tree import Node, Tree, is_tree, node_meta, tree_children, tree_label
from ..utils import shk_equals
from .bind import FanContext, RebindContext
from .chains import apply_op as chain_apply_op, evaluate_index_operand
from .common import require_number, stringify, token_kind
from .helpers import is_truthy, retargets_anchor
from .selector import selector_iter_values

EvalFunc = Callable[[Node, Frame], ShkValue]

def normalize_unary_op(op_node: Node, frame: Frame) -> Node | str:
    if tree_label(op_node) == 'unaryprefixop':
        if op_node.children:
            return normalize_unary_op(op_node.children[0], frame)
        src = getattr(frame, 'source', None)
        meta = node_meta(op_node)

        if src is not None and meta is not None:
            return src[meta.start_pos:meta.end_pos]
        return ''
    return op_node

def eval_unary(op_node: Node, rhs_node: Tree, frame: Frame, eval_func: EvalFunc, apply_op_func=chain_apply_op) -> ShkValue:
    op_norm = normalize_unary_op(op_node, frame)
    op_value = op_norm.value if isinstance(op_norm, Token) else op_norm

    if op_value in ('++', '--'):
        from .bind import resolve_assignable_node, apply_numeric_delta

        context = resolve_assignable_node(
            rhs_node,
            frame,
            eval_func=eval_func,
            apply_op=apply_op_func,
            evaluate_index_operand=evaluate_index_operand,
        )
        if isinstance(context, FanContext):
            raise ShakarRuntimeError("Increment target must end with a field or index")
        _, new_val = apply_numeric_delta(context, 1 if op_value == '++' else -1)
        return new_val

    rhs = eval_func(rhs_node, frame)

    match op_norm:
        case Token(type='PLUS') | '+':
            raise ShakarRuntimeError("unary + not supported")
        case Token(type='MINUS') | '-':
            rhs_num = _coerce_number(rhs)
            return ShkNumber(-rhs_num)
        case Token(type='TILDE') | '~':
            raise ShakarRuntimeError("bitwise ~ not supported yet")
        case Token(type='NOT') | 'not':
            return ShkBool(not is_truthy(rhs))
        case Token(type='NEG') | '!':
            return ShkBool(not is_truthy(rhs))
        case _:
            raise ShakarRuntimeError("Unsupported unary op")

def eval_infix(children: List[Tree], frame: Frame, eval_func: EvalFunc, right_assoc_ops: Set[str] | None=None) -> ShkValue:
    if not children:
        return ShkNull()

    if right_assoc_ops and _all_ops_in(children, right_assoc_ops):
        vals = [eval_func(children[i], frame) for i in range(0, len(children), 2)]
        acc = vals[-1]

        for i in range(len(vals)-2, -1, -1):
            lhs, rhs = vals[i], acc
            require_number(lhs); require_number(rhs)
            acc = ShkNumber(lhs.value ** rhs.value)
        return acc

    it = iter(children)
    acc = eval_func(next(it), frame)

    for x in it:
        op = as_op(x)
        rhs = eval_func(next(it), frame)
        acc = apply_binary_operator(op, acc, rhs)

    return acc

def eval_compare(children: List[Tree], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not children:
        return ShkNull()

    if len(children) == 1:
        return eval_func(children[0], frame)

    subject = eval_func(children[0], frame)
    idx = 1
    joiner = 'and'
    last_comp: Optional[str] = None
    agg: Optional[bool] = None

    while idx < len(children):
        node = children[idx]
        tok = token_kind(node)

        if tok in {'AND', 'and'}:
            joiner = 'and'
            idx += 1
            continue
        if tok in {'OR', 'or'}:
            joiner = 'or'
            idx += 1
            continue

        label = tree_label(node)
        if label == 'cmpop':
            comp = as_op(node)
            last_comp = comp
            idx += 1

            if idx >= len(children):
                raise ShakarRuntimeError("Missing right-hand side for comparison")
            rhs = eval_func(children[idx], frame)
            idx += 1
        else:
            if last_comp is None:
                raise ShakarRuntimeError("Comparator required in comparison chain")

            comp = last_comp
            rhs = eval_func(node, frame)
            idx += 1

        leg_val = _compare_values(comp, subject, rhs)

        if agg is None:
            agg = leg_val
        else:
            agg = (agg and leg_val) if joiner == 'and' else (agg or leg_val)

    return ShkBool(bool(agg)) if agg is not None else ShkBool(True)

def eval_logical(kind: str, children: List[Tree], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not children:
        return ShkNull()

    normalized = 'and' if 'and' in kind else 'or'
    prev_dot = frame.dot

    try:
        last_val: ShkValue
        if normalized == 'and':
            last_val = ShkBool(True)

            for child in children:
                if token_kind(child) in {'AND', 'OR'}:
                    continue

                val = eval_func(child, frame)
                frame.dot = val if retargets_anchor(child) else frame.dot
                last_val = val

                if not is_truthy(val):
                    return val
            return last_val

        last_val = ShkBool(False)

        for child in children:
            if token_kind(child) in {'AND', 'OR'}:
                continue

            val = eval_func(child, frame)
            frame.dot = val if retargets_anchor(child) else frame.dot
            last_val = val

            if is_truthy(val):
                return val
        return last_val
    finally:
        frame.dot = prev_dot

def eval_nullish(children: List[Tree], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    exprs = [child for child in children if not (isinstance(child, Token) and child.value == '??')]

    if not exprs:
        return ShkNull()

    current = eval_func(exprs[0], frame)

    for expr in exprs[1:]:
        if not isinstance(current, ShkNull):
            return current
        current = eval_func(expr, frame)

    return current

def eval_nullsafe(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not is_tree(node) or tree_label(node) != 'explicit_chain':
        return eval_func(node, frame)

    children = tree_children(node)
    if not children:
        return ShkNull()

    head = children[0]
    current = eval_func(head, frame)

    if isinstance(current, ShkNull):
        return ShkNull()

    for op in children[1:]:
        try:
            current = chain_apply_op(current, op, frame, eval_func)
        except (ShakarRuntimeError, ShakarTypeError) as err:
            if _nullsafe_recovers(err, current):
                return ShkNull()
            raise

        if isinstance(current, RebindContext):
            current = current.value

        if isinstance(current, ShkNull):
            return ShkNull()
    return current

def eval_ternary(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if len(n.children) != 3:
        raise ShakarRuntimeError("Malformed ternary expression")

    cond_node, true_node, false_node = n.children
    cond_val = eval_func(cond_node, frame)

    if is_truthy(cond_val):
        return eval_func(true_node, frame)

    return eval_func(false_node, frame)

def _all_ops_in(children: List[Tree], allowed: Set[str]) -> bool:
    return all(as_op(children[i]) in allowed for i in range(1, len(children), 2))

def as_op(x: Node) -> str:
    if isinstance(x, Token):
        return str(x.value)

    label = tree_label(x) if is_tree(x) else None
    if label is not None:
        if label in ('addop', 'mulop', 'powop') and len(x.children) == 1 and isinstance(x.children[0], Token):
            return str(x.children[0].value)

        if label == 'cmpop':
            tokens: list[str] = [str(tok.value) for tok in x.children if isinstance(tok, Token)]
            if not tokens:
                raise ShakarRuntimeError("Empty comparison operator")

            if tokens[0] == '!' and len(tokens) > 1:
                return '!' + ''.join(tokens[1:])

            if len(tokens) > 1:
                return " ".join(tokens)
            return tokens[0]

    raise ShakarRuntimeError(f"Expected operator token, got {x!r}")

def apply_binary_operator(op: str, lhs: ShkValue, rhs: ShkValue) -> ShkValue:
    match op:
        case '+':
            if isinstance(lhs, ShkString) or isinstance(rhs, ShkString):
                return ShkString(stringify(lhs) + stringify(rhs))
            require_number(lhs); require_number(rhs)
            return ShkNumber(lhs.value + rhs.value)
        case '-':
            require_number(lhs); require_number(rhs)
            return ShkNumber(lhs.value - rhs.value)
        case '*':
            require_number(lhs); require_number(rhs)
            return ShkNumber(lhs.value * rhs.value)
        case '/':
            require_number(lhs); require_number(rhs)
            return ShkNumber(lhs.value / rhs.value)
        case '//':
            require_number(lhs); require_number(rhs)
            return ShkNumber(math.floor(lhs.value / rhs.value))
        case '%':
            require_number(lhs); require_number(rhs)
            return ShkNumber(lhs.value % rhs.value)
        case '**':
            require_number(lhs); require_number(rhs)
            return ShkNumber(lhs.value ** rhs.value)
    raise ShakarRuntimeError(f"Unknown operator {op}")

def _compare_values(op: str, lhs: ShkValue, rhs: ShkValue) -> bool:
    if isinstance(rhs, ShkSelector):
        return _compare_with_selector(op, lhs, rhs)

    match op:
        case '==':
            return shk_equals(lhs, rhs)
        case '!=':
            return not shk_equals(lhs, rhs)
        case '<':
            return _coerce_number(lhs) < _coerce_number(rhs)
        case '<=':
            return _coerce_number(lhs) <= _coerce_number(rhs)
        case '>':
            return _coerce_number(lhs) > _coerce_number(rhs)
        case '>=':
            return _coerce_number(lhs) >= _coerce_number(rhs)
        case 'is':
            return shk_equals(lhs, rhs)
        case '!is' | 'is not':
            return not shk_equals(lhs, rhs)
        case 'in':
            return _contains(rhs, lhs)
        case 'not in' | '!in':
            return not _contains(rhs, lhs)
        case _:
            raise ShakarRuntimeError(f"Unknown comparator {op}")

def _selector_values(selector: ShkSelector) -> List[ShkNumber]:
    values = selector_iter_values(selector)
    if not values:
        raise ShakarRuntimeError("Selector literal produced no values")

    return [val for val in values if isinstance(val, ShkNumber)]

def _coerce_number(value: ShkValue) -> float:
    if isinstance(value, ShkNumber):
        return value.value

    raise ShakarTypeError("Expected number")

def _compare_with_selector(op: str, lhs: ShkValue, selector: ShkSelector) -> bool:
    values = _selector_values(selector)

    if op == '==':
        return all(shk_equals(lhs, val) for val in values)

    if op == '!=':
        return any(not shk_equals(lhs, val) for val in values)

    lhs_num = _coerce_number(lhs)
    rhs_nums = [_coerce_number(val) for val in values]

    match op:
        case '<':
            return all(lhs_num < num for num in rhs_nums)
        case '<=':
            return all(lhs_num <= num for num in rhs_nums)
        case '>':
            return all(lhs_num > num for num in rhs_nums)
        case '>=':
            return all(lhs_num >= num for num in rhs_nums)
        case _:
            raise ShakarTypeError(f"Unsupported comparator '{op}' for selector literal")

def _contains(container: ShkValue, item: ShkValue) -> bool:
    match container:
        case ShkArray(items=items):
            return any(shk_equals(element, item) for element in items)
        case ShkString(value=text):
            if isinstance(item, ShkString):
                return item.value in text
            raise ShakarTypeError("String membership requires a string value")
        case ShkObject(slots=slots):
            if isinstance(item, ShkString):
                return item.value in slots
            raise ShakarTypeError("Object membership requires a string key")
        case _:
            raise ShakarTypeError(f"Unsupported container type for 'in': {type(container).__name__}")

def _nullsafe_recovers(err: Exception, recv: ShkValue) -> bool:
    if isinstance(recv, ShkNull):
        return True
    return isinstance(err, (ShakarKeyError, ShakarIndexError))
