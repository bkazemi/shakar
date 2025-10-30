
from __future__ import annotations

from typing import Any, List, Optional
from lark import Tree, Token

from shakar_runtime import (
    Env, ShkNumber, ShkString, ShkBool, ShkNull, ShkArray, ShkObject, Descriptor, ShkFn, BoundMethod,
    ShakarRuntimeError, ShakarTypeError, ShakarArityError,
    call_builtin_method, call_shkfn, Builtins
)

# ---------------- Public API ----------------

def eval_expr(ast: Any, env: Optional[Env]=None, source: Optional[str]=None) -> Any:
    if env is None:
        env = Env(source=source)
    else:
        if source is not None:
            env.source = source
        elif not hasattr(env, 'source'):
            env.source = None
    return eval_node(ast, env)

# ---------------- Core evaluator ----------------

def eval_node(n: Any, env: Env) -> Any:
    if not isinstance(n, (Tree, Token)):
        return n
    if isinstance(n, Token):
        return _eval_token(n, env)

    d = n.data
    match d:
        # common wrapper nodes (delegate to single child)
        case 'start_noindent' | 'start_indented' | 'stmtlist':
            return _eval_program(n.children, env)
        case 'stmt':
            return _eval_program(n.children, env)
        case 'literal' | 'primary' | 'expr' | 'expr_nc':
            if len(n.children) == 1:
                return eval_node(n.children[0], env)
            if len(n.children) == 0 and d == 'literal':
                return _eval_keyword_literal(n)
            raise ShakarRuntimeError(f"Unsupported wrapper shape {d} with {len(n.children)} children")
        case 'array':
            return ShkArray([eval_node(c, env) for c in n.children])
        case 'object':
            return _eval_object(n, env)
        case 'unary' | 'unary_nc':
            op, rhs_node = n.children
            rhs = eval_node(rhs_node, env)
            return _eval_unary(op, rhs, env)
        case 'pow' | 'pow_nc':
            return _eval_infix(n.children, env, right_assoc_ops={'**', 'POW'})
        case 'mul' | 'mul_nc' | 'add' | 'add_nc':
            return _eval_infix(n.children, env)
        case 'explicit_chain':
            head, *ops = n.children
            val = eval_node(head, env)
            for op in ops:
                val = _apply_op(val, op, env)
            return val
        case 'implicit_chain':
            return _eval_implicit_chain(n.children, env)
        case 'call':
            args_node = n.children[0] if n.children else None
            args = _eval_args_node(args_node, env)
            cal = env.get('')  # unreachable in practice
            return _call_value(cal, args, env)
        case 'amp_lambda':
            return _eval_amp_lambda(n, env)
        case 'compare' | 'compare_nc':
            return _eval_compare(n.children, env)
        case 'and' | 'or' | 'and_nc' | 'or_nc':
            return _eval_logical(d, n.children, env)
        case 'walrus' | 'walrus_nc':
            return _eval_walrus(n.children, env)
        case 'assignstmt':
            return _eval_assign_stmt(n.children, env)
        case 'subject':
            return _get_subject(env)
        case 'keyexpr' | 'keyexpr_nc':
            return eval_node(n.children[0], env) if n.children else ShkNull()
        case 'destructure':
            return _eval_destructure(n, env, create=False, allow_broadcast=False)
        case 'destructure_walrus':
            return _eval_destructure(n, env, create=True, allow_broadcast=True)
        case 'inlinebody':
            return _eval_inline_body(n, env)
        case 'indentblock':
            return _eval_indent_block(n, env)
        case 'onelineguard':
            return _eval_oneline_guard(n.children, env)
        case _:
            raise ShakarRuntimeError(f"Unknown node: {d}")

# ---------------- Tokens ----------------

def _eval_token(t: Token, env: Env) -> Any:
    match t.type:
        case 'NUMBER':
            return ShkNumber(float(t.value))
        case 'STRING':
            v = t.value
            if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                v = v[1:-1]
            return ShkString(v)
        case 'TRUE':
            return ShkBool(True)
        case 'FALSE':
            return ShkBool(False)
        case 'IDENT':
            return env.get(t.value)
        case _:
            raise ShakarRuntimeError(f"Unhandled token {t.type}:{t.value}")

def _eval_keyword_literal(node: Tree) -> Any:
    meta = getattr(node, "meta", None)
    if meta is None:
        raise ShakarRuntimeError("Missing metadata for literal")
    end = getattr(meta, "end_pos", None)
    start = getattr(meta, "start_pos", None)
    if start is None or end is None:
        raise ShakarRuntimeError("Missing source span for literal")
    width = end - start
    match width:
        case 3:
            return ShkNull()
        case 4:
            return ShkBool(True)
        case 5:
            return ShkBool(False)
    raise ShakarRuntimeError("Unknown literal")

def _eval_program(children: List[Any], env: Env) -> Any:
    result: Any = ShkNull()
    skip_tokens = {'SEMI', '_NL', 'INDENT', 'DEDENT'}
    for child in children:
        if isinstance(child, Token) and child.type in skip_tokens:
            continue
        result = eval_node(child, env)
    return result

def _get_subject(env: Env) -> Any:
    if env.dot is None:
        raise ShakarRuntimeError("No subject available for '.'")
    return env.dot

def _eval_implicit_chain(ops: List[Any], env: Env) -> Any:
    val = _get_subject(env)
    for op in ops:
        val = _apply_op(val, op, env)
    return val

def _eval_inline_body(node: Any, env: Env) -> Any:
    if isinstance(node, Tree) and node.data == 'inlinebody':
        for child in node.children:
            if isinstance(child, Tree) and child.data == 'stmtlist':
                return _eval_program(child.children, env)
        if not node.children:
            return ShkNull()
        return eval_node(node.children[0], env)
    return eval_node(node, env)

def _eval_indent_block(node: Tree, env: Env) -> Any:
    return _eval_program(node.children, env)

def _eval_oneline_guard(children: List[Any], env: Env) -> Any:
    branches: List[Tree] = []
    else_body: Tree | None = None
    for child in children:
        if isinstance(child, Tree):
            if child.data == 'guardbranch':
                branches.append(child)
            elif child.data == 'inlinebody':
                else_body = child
    outer_dot = env.dot
    try:
        for branch in branches:
            if not isinstance(branch, Tree) or len(branch.children) != 2:
                raise ShakarRuntimeError("Malformed guard branch")
            cond_node, body_node = branch.children
            env.dot = outer_dot
            cond_val = eval_node(cond_node, env)
            if _is_truthy(cond_val):
                result = _eval_inline_body(body_node, env)
                env.dot = outer_dot
                return result
            env.dot = outer_dot
        if else_body is not None:
            env.dot = outer_dot
            result = _eval_inline_body(else_body, env)
            env.dot = outer_dot
            return result
        return ShkNull()
    finally:
        env.dot = outer_dot

# ---------------- Assignment ----------------

def _eval_walrus(children: List[Any], env: Env) -> Any:
    if len(children) != 2:
        raise ShakarRuntimeError("Malformed walrus expression")
    name, value_node = children
    if not isinstance(name, Token) or name.type != 'IDENT':
        raise ShakarRuntimeError("Walrus target must be an identifier")
    value = eval_node(value_node, env)
    return _assign_ident(name.value, value, env, create=True)

def _eval_assign_stmt(children: List[Any], env: Env) -> Any:
    if len(children) < 2:
        raise ShakarRuntimeError("Malformed assignment statement")
    lvalue_node = children[0]
    value_node = children[-1]
    value = eval_node(value_node, env)
    _assign_lvalue(lvalue_node, value, env, create=False)
    return ShkNull()

def _eval_destructure(n: Tree, env: Env, create: bool, allow_broadcast: bool) -> Any:
    if len(n.children) != 2:
        raise ShakarRuntimeError("Malformed destructure")
    pattern_list, rhs_node = n.children
    patterns = [c for c in getattr(pattern_list, 'children', []) if isinstance(c, Tree) and c.data == 'pattern']
    if not patterns:
        raise ShakarRuntimeError("Empty destructure pattern")
    values, result = _evaluate_destructure_rhs(rhs_node, env, len(patterns), allow_broadcast)
    for pat, val in zip(patterns, values):
        _assign_pattern(pat, val, env, create, allow_broadcast)
    return result if allow_broadcast else ShkNull()

def _assign_ident(name: str, value: Any, env: Env, create: bool) -> Any:
    try:
        env.set(name, value)
    except ShakarRuntimeError:
        if create:
            env.define(name, value)
        else:
            raise
    return value

def _assign_lvalue(node: Any, value: Any, env: Env, create: bool) -> Any:
    if not isinstance(node, Tree) or node.data != 'lvalue':
        raise ShakarRuntimeError("Invalid assignment target")
    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")
    head, *ops = node.children
    if not ops and isinstance(head, Token) and head.type == 'IDENT':
        return _assign_ident(head.value, value, env, create=create)
    target = eval_node(head, env)
    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    if isinstance(final_op, Tree):
        match final_op.data:
            case 'field' | 'fieldsel':
                name_tok = final_op.children[0]
                assert isinstance(name_tok, Token) and name_tok.type == 'IDENT'
                return _set_field(target, name_tok.value, value, env, create=create)
            case 'lv_index':
                return _set_index(target, final_op, value, env)
    raise ShakarRuntimeError("Unsupported assignment target")

def _evaluate_destructure_rhs(rhs_node: Any, env: Env, target_count: int, allow_broadcast: bool) -> tuple[list[Any], Any]:
    if isinstance(rhs_node, Tree) and rhs_node.data == 'pack':
        vals = [eval_node(child, env) for child in rhs_node.children]
        result = ShkArray(vals)
    else:
        single = eval_node(rhs_node, env)
        vals = [single]
        result = single
    if len(vals) == 1 and target_count > 1:
        seq = _coerce_sequence(vals[0], target_count)
        if seq is not None:
            vals = list(seq)
            if isinstance(result, list) or isinstance(result, tuple):
                result = ShkArray(vals)
        elif allow_broadcast:
            vals = [vals[0]] * target_count
        else:
            raise ShakarRuntimeError("Destructure arity mismatch")
    elif len(vals) != target_count:
        raise ShakarRuntimeError("Destructure arity mismatch")
    return vals, result

def _assign_pattern(pattern: Tree, value: Any, env: Env, create: bool, allow_broadcast: bool) -> None:
    if not isinstance(pattern, Tree) or pattern.data != 'pattern' or not pattern.children:
        raise ShakarRuntimeError("Malformed pattern")
    target = pattern.children[0]
    if isinstance(target, Token) and target.type == 'IDENT':
        if create:
            env.define(target.value, value)
        else:
            _assign_ident(target.value, value, env, create=False)
        return
    if isinstance(target, Tree) and target.data == 'pattern_list':
        subpatterns = [c for c in getattr(target, 'children', []) if isinstance(c, Tree) and c.data == 'pattern']
        if not subpatterns:
            raise ShakarRuntimeError("Empty nested pattern")
        seq = _coerce_sequence(value, len(subpatterns))
        if seq is None:
            if allow_broadcast and len(subpatterns) > 1:
                seq = [value] * len(subpatterns)
            else:
                raise ShakarRuntimeError("Destructure expects a sequence")
        for sub_pat, sub_val in zip(subpatterns, seq):
            _assign_pattern(sub_pat, sub_val, env, create, allow_broadcast)
        return
    raise ShakarRuntimeError("Unsupported pattern element")

def _coerce_sequence(value: Any, expected_len: int) -> list[Any] | None:
    if isinstance(value, ShkArray):
        items = list(value.items)
    elif isinstance(value, list):
        items = list(value)
    elif isinstance(value, tuple):
        items = list(value)
    else:
        return None
    if len(items) != expected_len:
        raise ShakarRuntimeError("Destructure arity mismatch")
    return items

# ---------------- Comparison ----------------

def _eval_compare(children: List[Any], env: Env) -> Any:
    if not children:
        return ShkNull()

    if len(children) == 1:
        return eval_node(children[0], env)

    lhs = eval_node(children[0], env)
    idx = 1
    while idx < len(children):
        op_node = children[idx]
        op = _as_op(op_node)
        idx += 1
        if idx >= len(children):
            raise ShakarRuntimeError("Missing right-hand side for comparison")
        rhs = eval_node(children[idx], env)
        idx += 1
        if not _compare_values(op, lhs, rhs):
            return ShkBool(False)
        lhs = rhs
    return ShkBool(True)

def _compare_values(op: str, lhs: Any, rhs: Any) -> bool:
    match op:
        case '==':
            return _shk_equals(lhs, rhs)
        case '!=':
            return not _shk_equals(lhs, rhs)
        case '<':
            _require_number(lhs); _require_number(rhs)
            return lhs.value < rhs.value
        case '<=':
            _require_number(lhs); _require_number(rhs)
            return lhs.value <= rhs.value
        case '>':
            _require_number(lhs); _require_number(rhs)
            return lhs.value > rhs.value
        case '>=':
            _require_number(lhs); _require_number(rhs)
            return lhs.value >= rhs.value
        case 'is':
            return _shk_equals(lhs, rhs)
        case '!is' | 'is not':
            return not _shk_equals(lhs, rhs)
        case 'in':
            return _contains(rhs, lhs)
        case 'not in' | '!in':
            return not _contains(rhs, lhs)
        case _:
            raise ShakarRuntimeError(f"Unknown comparator {op}")

def _contains(container: Any, item: Any) -> bool:
    match container:
        case ShkArray(items=items):
            return any(_shk_equals(element, item) for element in items)
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

def _shk_equals(lhs: Any, rhs: Any) -> bool:
    if type(lhs) is not type(rhs):
        return False
    match lhs:
        case ShkNull():
            return True
        case ShkNumber(value=a):
            return a == rhs.value
        case ShkString(value=a):
            return a == rhs.value
        case ShkBool(value=a):
            return a == rhs.value
        case ShkArray(items=items):
            rhs_items = rhs.items
            if len(items) != len(rhs_items):
                return False
            return all(_shk_equals(a, b) for a, b in zip(items, rhs_items))
        case ShkObject(slots=slots):
            rhs_slots = rhs.slots
            if slots.keys() != rhs_slots.keys():
                return False
            return all(_shk_equals(slots[k], rhs_slots[k]) for k in slots)
        case ShkFn():
            return lhs is rhs
        case Descriptor():
            return lhs is rhs
        case _:
            return lhs is rhs

def _eval_logical(kind: str, children: List[Any], env: Env) -> Any:
    if not children:
        return ShkNull()
    normalized = 'and' if 'and' in kind else 'or'
    prev_dot = env.dot
    try:
        if normalized == 'and':
            last_val: Any = ShkBool(True)
            for child in children:
                val = eval_node(child, env)
                if _retargets_anchor(child):
                    env.dot = val
                last_val = val
                if not _is_truthy(val):
                    return val
            return last_val
        last_val: Any = ShkBool(False)
        for child in children:
            val = eval_node(child, env)
            if _retargets_anchor(child):
                env.dot = val
            last_val = val
            if _is_truthy(val):
                return val
        return last_val
    finally:
        env.dot = prev_dot

def _is_truthy(val: Any) -> bool:
    match val:
        case ShkBool(value=b):
            return b
        case ShkNull():
            return False
        case ShkNumber(value=num):
            return num != 0
        case ShkString(value=s):
            return bool(s)
        case ShkArray(items=items):
            return bool(items)
        case ShkObject(slots=slots):
            return bool(slots)
        case _:
            return True

def _retargets_anchor(node: Any) -> bool:
    if isinstance(node, Token):
        return node.type not in {'SEMI', '_NL', 'INDENT', 'DEDENT'}
    if isinstance(node, Tree):
        return node.data not in {'implicit_chain', 'subject'}
    return True

# ---------------- Arithmetic ----------------

def _normalize_unary_op(op_node: Any, env: Env) -> Any:
    if isinstance(op_node, Tree) and op_node.data == 'unaryprefixop':
        if op_node.children:
            return _normalize_unary_op(op_node.children[0], env)
        src = getattr(env, 'source', None)
        meta = getattr(op_node, 'meta', None)
        if src is not None and meta is not None:
            return src[meta.start_pos:meta.end_pos]
        return ''
    return op_node

def _eval_unary(op_tok_or_str: Any, rhs: Any, env: Env) -> Any:
    op_tok_or_str = _normalize_unary_op(op_tok_or_str, env)
    match op_tok_or_str:
        case Token(type='PLUS') | '+':
          #return rhs
          raise ShakarRuntimeError("unary + not supported")
        case Token(type='MINUS') | '-':
            _require_number(rhs)
            return ShkNumber(-rhs.value)
        case Token(type='TILDE') | '~':
            raise ShakarRuntimeError("bitwise ~ not supported yet")
        case Token(type='NOT') | 'not':
            return ShkBool(not _is_truthy(rhs))
        case Token(type='NEG') | '!':
            return ShkBool(not _is_truthy(rhs))
        case _:
            if op_tok_or_str in ('', None):
                if isinstance(rhs, ShkNumber):
                    return ShkNumber(-rhs.value)
                return ShkBool(not _is_truthy(rhs))
            raise ShakarRuntimeError("Unsupported unary op")

def _eval_infix(children: List[Any], env: Env, right_assoc_ops: set|None=None) -> Any:
    if not children:
        return ShkNull()

    if right_assoc_ops and _all_ops_in(children, right_assoc_ops):
        vals = [eval_node(children[i], env) for i in range(0, len(children), 2)]
        acc = vals[-1]
        for i in range(len(vals)-2, -1, -1):
            lhs, rhs = vals[i], acc
            _require_number(lhs); _require_number(rhs)
            acc = ShkNumber(lhs.value ** rhs.value)
        return acc

    it = iter(children)
    acc = eval_node(next(it), env)
    for x in it:
        op = _as_op(x)
        rhs = eval_node(next(it), env)
        match op:
            case '+':
                if isinstance(acc, ShkString) or isinstance(rhs, ShkString):
                    acc = ShkString(str(getattr(acc, 'value', acc)) + str(getattr(rhs, 'value', rhs)))
                else:
                    _require_number(acc); _require_number(rhs)
                    acc = ShkNumber(acc.value + rhs.value)
            case '-':
                _require_number(acc); _require_number(rhs)
                acc = ShkNumber(acc.value - rhs.value)
            case '*':
                _require_number(acc); _require_number(rhs)
                acc = ShkNumber(acc.value * rhs.value)
            case '/':
                _require_number(acc); _require_number(rhs)
                acc = ShkNumber(acc.value / rhs.value)
            case '%':
                _require_number(acc); _require_number(rhs)
                acc = ShkNumber(acc.value % rhs.value)
            case _:
                raise ShakarRuntimeError(f"Unknown operator {op}")
    return acc

def _all_ops_in(children: List[Any], allowed: set) -> bool:
    for i in range(1, len(children), 2):
        op = _as_op(children[i])
        if op not in allowed:
            return False
    return True

def _as_op(x: Any) -> str:
    if isinstance(x, Token):
        return x.value
    if isinstance(x, Tree):
        # e.g. Tree('addop', [Token('PLUS','+')]) or mulop/powop
        if x.data in ('addop','mulop','powop') and len(x.children) == 1 and isinstance(x.children[0], Token):
            return x.children[0].value
        if x.data == 'cmpop':
            tokens = [tok.value for tok in x.children if isinstance(tok, Token)]
            if not tokens:
                raise ShakarRuntimeError("Empty comparison operator")
            if tokens[0] == '!' and len(tokens) > 1:
                return '!' + ''.join(tokens[1:])
            if len(tokens) > 1:
                return " ".join(tokens)
            return tokens[0]

    raise ShakarRuntimeError(f"Expected operator token, got {x!r}")

def _require_number(v: Any) -> None:
    if not isinstance(v, ShkNumber):
        raise ShakarTypeError("Expected number")

# ---------------- Chains ----------------

def _apply_op(recv: Any, op: Tree, env: Env) -> Any:
    d = op.data
    match d:
        case 'field' | 'fieldsel':
            name_tok = op.children[0]
            assert isinstance(name_tok, Token) and name_tok.type == 'IDENT'
            return _get_field(recv, name_tok.value, env)
        case 'index':
            expr_node = _index_expr_from_children(op.children)
            idx = eval_node(expr_node, env)
            return _index(recv, idx, env)
        case 'slicesel':
            return _slice(recv, op.children, env)
        case 'call':
            args = _eval_args_node(op.children[0] if op.children else None, env)
            return _call_value(recv, args, env)
        case 'method':
            name_tok = op.children[0]; assert isinstance(name_tok, Token) and name_tok.type == 'IDENT'
            args = _eval_args_node(op.children[1] if len(op.children)>1 else None, env)
            try:
                return call_builtin_method(recv, name_tok.value, args, env)
            except Exception:
                cal = _get_field(recv, name_tok.value, env)
                if isinstance(cal, BoundMethod):
                    return call_shkfn(cal.fn, args, subject=cal.subject, caller_env=env)
                if isinstance(cal, ShkFn):
                    return call_shkfn(cal, args, subject=recv, caller_env=env)
                raise
        case _:
            raise ShakarRuntimeError(f"Unknown chain op: {d}")

def _eval_args_node(args_node: Any, env: Env) -> List[Any]:
    def label(node: Tree) -> str | None:
        data = getattr(node, 'data', None)
        if isinstance(data, Token):
            return data.value
        return data

    def flatten(node: Any) -> List[Any]:
        if isinstance(node, Tree):
            tag = label(node)
            if tag in {'args', 'arglist', 'arglistnamedmixed'}:
                out: List[Any] = []
                for ch in node.children:
                    out.extend(flatten(ch))
                return out
            if tag in {'argitem', 'arg'} and node.children:
                return flatten(node.children[0])
            if tag == 'namedarg' and node.children:
                # ignore the name for now; evaluate the value node
                return flatten(node.children[-1])
        return [node]

    if isinstance(args_node, Tree):
        return [eval_node(n, env) for n in flatten(args_node)]
    if isinstance(args_node, list):
        res: List[Any] = []
        for n in args_node:
            res.extend(flatten(n))
        return [eval_node(n, env) for n in res]
    return []

def _get_field(recv: Any, name: str, env: Env) -> Any:
    match recv:
        case ShkObject(slots=slots):
            if name in slots:
                slot = slots[name]
                if isinstance(slot, Descriptor):
                    if slot.getter is None:
                        return ShkNull()
                    return call_shkfn(slot.getter, [], subject=recv, caller_env=env)
                if isinstance(slot, ShkFn):
                    return BoundMethod(slot, recv)
                return slot
            raise ShakarRuntimeError(f"Key '{name}' not found")
        case ShkArray(items=_):
            if name == "len":
                return ShkNumber(float(len(recv.items)))
            raise ShakarTypeError(f"Array has no field '{name}'")
        case ShkString(value=_):
            if name == "len":
                return ShkNumber(float(len(recv.value)))
            raise ShakarTypeError(f"String has no field '{name}'")
        case ShkFn():
            raise ShakarTypeError("Function has no fields")
        case _:
            raise ShakarTypeError(f"Unsupported field access on {type(recv).__name__}")

def _index(recv: Any, idx: Any, env: Env) -> Any:
    match recv:
        case ShkArray(items=items):
            if isinstance(idx, ShkNumber):
                return items[int(idx.value)]
            raise ShakarTypeError("Array index must be a number")
        case ShkString(value=s):
            if isinstance(idx, ShkNumber):
                return ShkString(s[int(idx.value)])
            raise ShakarTypeError("String index must be a number")
        case ShkObject(slots=slots):
            if isinstance(idx, ShkString):
                key = idx.value
            elif isinstance(idx, ShkNumber):
                key = str(int(idx.value))
            else:
                key = str(idx)
            if key in slots:
                val = slots[key]
                if isinstance(val, Descriptor):
                    getter = val.getter
                    if getter is None:
                        return ShkNull()
                    return call_shkfn(getter, [], subject=recv, caller_env=env)
                return val
            raise ShakarRuntimeError(f"Key '{key}' not found")
        case _:
            raise ShakarTypeError("Unsupported index operation")

def _index_expr_from_children(children: List[Any]) -> Any:
    queue = list(children)
    while queue:
        node = queue.pop(0)
        if isinstance(node, Token):
            continue
        if not isinstance(node, Tree):
            return node
        tag = node.data
        if tag in {'selectorlist', 'selector', 'indexsel'}:
            queue.extend(node.children)
            continue
        return node
    raise ShakarRuntimeError("Malformed index expression")

# Helpers for assignment mutation
def _set_field(recv: Any, name: str, value: Any, env: Env, create: bool) -> Any:
    match recv:
        case ShkObject(slots=slots):
            slot = slots.get(name)
            if isinstance(slot, Descriptor):
                setter = slot.setter
                if setter is None:
                    raise ShakarRuntimeError(f"Property '{name}' is read-only")
                call_shkfn(setter, [value], subject=recv, caller_env=env)
                return value
            slots[name] = value
            return value
        case _:
            raise ShakarTypeError(f"Cannot set field '{name}' on {type(recv).__name__}")

def _set_index(recv: Any, index_node: Tree, value: Any, env: Env) -> Any:
    index_val = _extract_index_value(index_node, env)
    match recv:
        case ShkArray(items=items):
            if isinstance(index_val, ShkNumber):
                idx = int(index_val.value)
            elif isinstance(index_val, int):
                idx = index_val
            else:
                raise ShakarTypeError("Array index must be an integer")
            items[idx] = value
            return value
        case ShkObject(slots=slots):
            key = index_val
            if isinstance(key, ShkString):
                key = key.value
            slots[str(key)] = value
            return value
        case _:
            raise ShakarTypeError("Unsupported index assignment target")

def _extract_index_value(index_node: Tree, env: Env) -> Any:
    expr_node = _index_expr_from_children(index_node.children)
    operand = eval_node(expr_node, env)
    if isinstance(operand, ShkNumber):
        return int(operand.value)
    if isinstance(operand, ShkString):
        return operand.value
    return operand
    raise ShakarRuntimeError("Malformed index selector")

def _slice(recv: Any, arms: List[Any], env: Env) -> Any:
    def arm_to_py(t):
        if isinstance(t, Tree) and t.data == 'emptyexpr':
            return None
        v = eval_node(t, env)
        return int(v.value) if isinstance(v, ShkNumber) else None
    start, stop, step = map(arm_to_py, arms)
    s = slice(start, stop, step)
    match recv:
        case ShkArray(items=items):
            return ShkArray(items[s])
        case ShkString(value=sval):
            return ShkString(sval[s])
        case _:
            raise ShakarTypeError("Slice only supported on arrays/strings")

def _call_value(cal: Any, args: List[Any], env: Env) -> Any:
    match cal:
        case BoundMethod(fn=fn, subject=subject):
            return call_shkfn(fn, args, subject=subject, caller_env=env)
        case ShkFn():
            return call_shkfn(cal, args, subject=None, caller_env=env)
        case _:
            raise ShakarTypeError(f"Cannot call value of type {type(cal).__name__}")

# ---------------- Objects ----------------

def _eval_object(n: Tree, env: Env) -> ShkObject:
    slots: dict[str, Any] = {}

    def _install_descriptor(name: str, getter: ShkFn|None=None, setter: ShkFn|None=None) -> None:
        existing = slots.get(name)
        if isinstance(existing, Descriptor):
            if getter is not None:
                existing.getter = getter
            if setter is not None:
                existing.setter = setter
            slots[name] = existing
        else:
            slots[name] = Descriptor(getter=getter, setter=setter)

    def _extract_params(params_node: Tree|None) -> List[str]:
        if params_node is None:
            return []
        names: List[str] = []
        queue = list(getattr(params_node, 'children', []))
        while queue:
            node = queue.pop(0)
            ident = _unwrap_ident(node)
            if ident is not None:
                names.append(ident)
                continue
            if isinstance(node, Tree):
                queue.extend(node.children)
        return names

    def _unwrap_ident(node: Any) -> str | None:
        cur = node
        seen = set()
        while isinstance(cur, Tree) and cur.children and id(cur) not in seen:
            seen.add(id(cur))
            if len(cur.children) != 1:
                break
            cur = cur.children[0]
        if isinstance(cur, Token) and cur.type == 'IDENT':
            return cur.value
        return None

    def _maybe_method_signature(key_node: Any) -> tuple[str, List[str]] | None:
        if not isinstance(key_node, Tree) or key_node.data != 'key_expr':
            return None
        target = key_node.children[0] if key_node.children else None
        chain = None
        if isinstance(target, Tree):
            if getattr(target, 'data', None) == 'explicit_chain':
                chain = target
            else:
                for ch in getattr(target, 'children', []):
                    if isinstance(ch, Tree) and getattr(ch, 'data', None) == 'explicit_chain':
                        chain = ch
                        break
        if chain is None or len(chain.children) != 2:
            return None
        head, call_node = chain.children
        if not isinstance(head, Token) or head.type != 'IDENT':
            return None
        if not (isinstance(call_node, Tree) and call_node.data == 'call'):
            return None
        args_node = call_node.children[0] if call_node.children else None
        params: List[str] = []
        if isinstance(args_node, Tree):
            queue = list(args_node.children)
            while queue:
                raw = queue.pop(0)
                if isinstance(raw, Tree) and getattr(raw, 'data', None) in {'namedarg', 'kwarg'}:
                    return None
                if isinstance(raw, Tree) and raw.children and getattr(raw, 'data', None) not in {'args','arglist','arglistnamedmixed','argitem','arg'}:
                    # allow deeper structures by re-queueing children
                    queue.extend(raw.children)
                    continue
                ident = _unwrap_ident(raw)
                if ident is None:
                    if isinstance(raw, Tree):
                        queue.extend(raw.children)
                    else:
                        return None
                else:
                    params.append(ident)
        elif args_node is not None:
            ident = _unwrap_ident(args_node)
            if ident is None:
                return None
            params.append(ident)
        return (head.value, params)

    def handle_item(item: Tree) -> None:
        match item.data:
            case 'obj_field':
                key_node, val_node = item.children
                method_sig = _maybe_method_signature(key_node)
                if method_sig:
                    name, params = method_sig
                    method_fn = ShkFn(params=params, body=val_node, env=Env(parent=env))
                    slots[name] = method_fn
                    return
                key = _eval_key(key_node, env)
                val = eval_node(val_node, env)
                slots[str(key)] = val
            case 'obj_get':
                name_tok, body = item.children
                if name_tok is None:
                    raise ShakarRuntimeError("Getter missing name")
                key = name_tok.value
                getter_fn = ShkFn(params=None, body=body, env=Env(parent=env))
                _install_descriptor(key, getter=getter_fn)
            case 'obj_set':
                name_tok, param_tok, body = item.children
                if name_tok is None or param_tok is None:
                    raise ShakarRuntimeError("Setter missing name or parameter")
                key = name_tok.value
                setter_fn = ShkFn(params=[param_tok.value], body=body, env=Env(parent=env))
                _install_descriptor(key, setter=setter_fn)
            case 'obj_method':
                name_tok, params_node, body = item.children
                if name_tok is None:
                    raise ShakarRuntimeError("Method missing name")
                param_names = _extract_params(params_node)
                method_fn = ShkFn(params=param_names, body=body, env=Env(parent=env))
                slots[name_tok.value] = method_fn
            case _:
                raise ShakarRuntimeError(f"Unknown object item {item.data}")

    for child in n.children:
        if isinstance(child, Token):
            continue
        if child.data == 'object_items':
            for item in child.children:
                if isinstance(item, Tree) and item.data != 'obj_sep':
                    handle_item(item)
        else:
            if child.data == 'obj_sep':
                continue
            handle_item(child)
    return ShkObject(slots)

def _eval_key(k: Any, env: Env) -> Any:
    if isinstance(k, Tree):
        match k.data:
            case 'key_ident':
                t = k.children[0]; return t.value
            case 'key_string':
                t = k.children[0]; s = t.value
                if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
                    s = s[1:-1]
                return s
            case 'key_expr':
                v = eval_node(k.children[0], env)
                return v.value if isinstance(v, ShkString) else v

    if isinstance(k, Token) and k.type in ('IDENT','STRING'):
        return k.value.strip('"').strip("'")

    return eval_node(k, env)

def _eval_amp_lambda(n: Tree, env: Env) -> ShkFn:
    if len(n.children) == 1:
        return ShkFn(params=None, body=n.children[0], env=Env(parent=env, dot=None))

    if len(n.children) == 2:
        params_node, body = n.children
        params: List[str] = []
        for p in getattr(params_node, 'children', []):
            if isinstance(p, Token) and p.type == 'IDENT':
                params.append(p.value)
            else:
                raise ShakarRuntimeError(f"Unsupported param node in amp_lambda: {p}")
        return ShkFn(params=params, body=body, env=Env(parent=env, dot=None))

    raise ShakarRuntimeError("amp_lambda malformed")
