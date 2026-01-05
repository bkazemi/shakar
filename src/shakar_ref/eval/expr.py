from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Set

from ..tree import Tok
from ..token_types import TT

from ..runtime import (
    Frame,
    ShkArray,
    ShkBool,
    ShkNull,
    ShkNumber,
    ShkObject,
    ShkPath,
    ShkValue,
    ShkSelector,
    ShkString,
    ShkRegex,
    ShakarKeyError,
    ShakarIndexError,
    ShakarRuntimeError,
    ShakarTypeError,
    regex_match_value,
)
from ..tree import Node, Tree, is_tree, is_token, node_meta, tree_children, tree_label
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
    op_value = op_norm.value if isinstance(op_norm, Tok) else op_norm

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
        case Tok(type=TT.PLUS) | '+':
            raise ShakarRuntimeError("unary + not supported")
        case Tok(type=TT.MINUS) | '-':
            rhs_num = _coerce_number(rhs)
            return ShkNumber(-rhs_num)
        case Tok(type=TT.TILDE) | '~':
            raise ShakarRuntimeError("bitwise ~ not supported yet")
        case Tok(type=TT.NOT) | 'not':
            return ShkBool(not is_truthy(rhs))
        case Tok(type=TT.NEG) | '!':
            return ShkBool(not is_truthy(rhs))
        case _:
            raise ShakarRuntimeError("Unsupported unary op")

def eval_infix(children: List[Tree], frame: Frame, eval_func: EvalFunc, right_assoc_ops: Optional[Set[str]]=None) -> ShkValue:
    if not children:
        return ShkNull()

    arithmetic_ops: Set[str] = {"+", "-", "*", "/", "//", "%", "+>", "^", "**"}

    def _should_retarget(node: Node, next_op: Optional[str]) -> bool:
        if not retargets_anchor(node):
            return False

        # Avoid clobbering anchor for plain identifiers in arithmetic chains so subjectful anchors survive (e.g., acc + .upper()).
        if token_kind(node) == "IDENT" and (next_op in arithmetic_ops or next_op is None):
            return False

        return True

    if right_assoc_ops and _all_ops_in(children, right_assoc_ops):
        vals: List[ShkValue] = []
        ops = [as_op(children[i]) for i in range(1, len(children), 2)]
        for idx in range(0, len(children), 2):
            operand_node = children[idx]
            next_op = ops[idx] if idx < len(ops) else None
            val = eval_func(operand_node, frame)
            if _should_retarget(operand_node, next_op):
                frame.dot = val
            vals.append(val)

        acc = vals[-1]

        for i in range(len(vals)-2, -1, -1):
            lhs = require_number(vals[i])
            rhs = require_number(acc)
            acc = ShkNumber(lhs.value ** rhs.value)
        return acc

    operands = children[0::2]
    ops = [as_op(op_node) for op_node in children[1::2]]

    acc = eval_func(operands[0], frame)
    if _should_retarget(operands[0], ops[0] if ops else None):
        frame.dot = acc

    for idx in range(1, len(operands)):
        op = ops[idx - 1]
        rhs_node = operands[idx]
        rhs = eval_func(rhs_node, frame)
        if _should_retarget(rhs_node, ops[idx] if idx < len(ops) else None):
            frame.dot = rhs
        acc = apply_binary_operator(op, acc, rhs)

    return acc

def eval_compare(children: List[Tree], frame: Frame, eval_func: EvalFunc) -> ShkValue:
    if not children:
        return ShkNull()

    if len(children) == 1:
        return eval_func(children[0], frame)

    if len(children) == 3 and tree_label(children[1]) == 'cmpop' and as_op(children[1]) == '~~':
        lhs_node, _, rhs_node = children
        lhs = eval_func(lhs_node, frame)
        if retargets_anchor(lhs_node):
            frame.dot = lhs
        rhs = eval_func(rhs_node, frame)
        if retargets_anchor(rhs_node):
            frame.dot = rhs
        return _regex_match(lhs, rhs)

    first_node = children[0]
    subject = eval_func(first_node, frame)
    if retargets_anchor(first_node):
        frame.dot = subject
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
            rhs_node = children[idx]
            rhs = eval_func(rhs_node, frame)
            if retargets_anchor(rhs_node):
                frame.dot = rhs
            idx += 1
        else:
            if last_comp is None:
                raise ShakarRuntimeError("Comparator required in comparison chain")

            comp = last_comp
            rhs = eval_func(node, frame)
            if retargets_anchor(node):
                frame.dot = rhs
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
    exprs = [child for child in children if not (isinstance(child, Tok) and child.value == '??')]

    if not exprs:
        return ShkNull()

    current = eval_func(exprs[0], frame)

    for expr in exprs[1:]:
        if not isinstance(current, ShkNull):
            return current
        current = eval_func(expr, frame)

    return current

def eval_nullsafe(node: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue:
    target = node
    if is_tree(node) and tree_label(node) == 'nullsafe' and node.children:
        target = node.children[0]

    if not is_tree(target) or tree_label(target) != 'explicit_chain':
        try:
            return eval_func(target, frame)
        except (ShakarKeyError, ShakarIndexError):
            return ShkNull()

    children = tree_children(target)
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
    if isinstance(x, Tok):
        return str(x.value)

    label = tree_label(x) if is_tree(x) else None
    if label is not None:
        if label in ('addop', 'mulop', 'powop') and len(x.children) == 1 and isinstance(x.children[0], Tok):
            return str(x.children[0].value)

        if label == 'cmpop':
            tokens: list[str] = [str(tok.value) for tok in x.children if isinstance(tok, Tok)]
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
            if isinstance(lhs, ShkArray) and isinstance(rhs, ShkArray):
                return ShkArray(lhs.items + rhs.items)
            if isinstance(lhs, ShkString) or isinstance(rhs, ShkString):
                return ShkString(stringify(lhs) + stringify(rhs))
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value + rhs_num.value)
        case '-':
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value - rhs_num.value)
        case '*':
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value * rhs_num.value)
        case '/':
            if isinstance(lhs, ShkPath):
                if isinstance(rhs, ShkPath):
                    return ShkPath(str(lhs.as_path() / rhs.as_path()))
                if isinstance(rhs, ShkString):
                    return ShkPath(str(lhs.as_path() / rhs.value))
                raise ShakarTypeError("Path join expects a path or string on the right")
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value / rhs_num.value)
        case '//':
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(math.floor(lhs_num.value / rhs_num.value))
        case '%':
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value % rhs_num.value)
        case '**':
            lhs_num = require_number(lhs)
            rhs_num = require_number(rhs)
            return ShkNumber(lhs_num.value ** rhs_num.value)
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
        case '~':
            from .match import match_structure
            return match_structure(lhs, rhs)
        case '~~':
            return is_truthy(_regex_match(lhs, rhs))
        case _:
            raise ShakarRuntimeError(f"Unknown comparator {op}")

def _regex_match(lhs: ShkValue, rhs: ShkValue) -> ShkValue:
    if not isinstance(lhs, ShkString):
        raise ShakarTypeError("Regex match expects a string on the left")
    if not isinstance(rhs, ShkRegex):
        raise ShakarTypeError("Regex match expects a regex on the right")
    return regex_match_value(rhs, lhs.value)

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
