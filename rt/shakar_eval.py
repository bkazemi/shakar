
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional
import math
from lark import Tree, Token

from contextlib import contextmanager

from shakar_runtime import (
    Env, ShkNumber, ShkString, ShkBool, ShkNull, ShkArray, ShkObject, Descriptor, ShkFn, BoundMethod, BuiltinMethod,
    ShkSelector, SelectorIndex, SelectorSlice,
    ShakarRuntimeError, ShakarTypeError, ShakarArityError, ShakarKeyError, ShakarIndexError, ShakarMethodNotFound,
    ShakarAssertionError,
    call_builtin_method, call_shkfn, Builtins
)

from shakar_utils import (
    value_in_list,
    shk_equals,
    sequence_items,
    fanout_values,
    normalize_object_key,
)
from shakar_tree import (
    tree_label,
    tree_children,
    node_meta,
    child_by_label,
    child_by_labels,
    first_child,
    is_tree_node,
    is_token_node,
)

from eval.selector import (
    eval_selectorliteral,
    evaluate_selectorlist,
    clone_selector_parts,
    apply_selectors_to_value,
    selector_iter_values,
)

from eval.destructure import (
    evaluate_destructure_rhs,
    assign_pattern as destructure_assign_pattern,
    infer_implicit_binders as destructure_infer_implicit_binders,
    apply_comp_binders as destructure_apply_comp_binders,
)

from eval.mutation import (
    set_field_value,
    set_index_value,
    index_value,
    slice_value,
    get_field_value,
)

class _RebindContext:
    __slots__ = ("value", "setter")

    def __init__(self, value: Any, setter: Callable[[Any], None]) -> None:
        self.value = value
        self.setter = setter

class _FanContext:
    __slots__ = ("contexts", "values")

    def __init__(self, contexts: List[_RebindContext]) -> None:
        self.contexts = contexts
        self.values = [ctx.value for ctx in contexts]

    def update_from_contexts(self) -> None:
        self.values = [ctx.value for ctx in self.contexts]

    def snapshot(self) -> List[Any]:
        return list(self.values)

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

def _token_kind(node: Any) -> Optional[str]:
    return node.type if is_token_node(node) else None

def _expect_ident_token(node: Any, context: str) -> str:
    if is_token_node(node) and _token_kind(node) == 'IDENT':
        return node.value
    raise ShakarRuntimeError(f"{context} must be an identifier")

def _ident_token_value(node: Any) -> Optional[str]:
    if is_token_node(node) and _token_kind(node) == 'IDENT':
        return node.value
    return None

def _is_literal_node(node: Any) -> bool:
    return not isinstance(node, (Tree, Token))

def _get_source_segment(node: Any, env: Env) -> Optional[str]:
    source = getattr(env, 'source', None)
    if source is None:
        return None
    meta = node_meta(node)
    if meta is None:
        return None
    start = getattr(meta, "start_pos", None)
    end = getattr(meta, "end_pos", None)
    if start is None or end is None:
        return None
    return source[start:end]

def eval_node(n: Any, env: Env) -> Any:
    if _is_literal_node(n):
        return n
    if is_token_node(n):
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
            return _eval_unary(op, rhs_node, env)
        case 'pow' | 'pow_nc':
            return _eval_infix(n.children, env, right_assoc_ops={'**', 'POW'})
        case 'mul' | 'mul_nc' | 'add' | 'add_nc':
            return _eval_infix(n.children, env)
        case 'explicit_chain':
            head, *ops = n.children
            if ops and tree_label(ops[-1]) in {'incr', 'decr'}:
                tail = ops[-1]
                context = _resolve_chain_assignment(head, ops[:-1], env)
                delta = 1 if tree_label(tail) == 'incr' else -1
                if isinstance(context, _FanContext):
                    raise ShakarRuntimeError("++/-- not supported on field fan assignments")
                old_val, _ = _apply_numeric_delta(context, delta)
                return old_val
            val = eval_node(head, env)
            for op in ops:
                val = _apply_op(val, op, env)
            if isinstance(val, _RebindContext):
                final = val.value
                val.setter(final)
                return final
            if isinstance(val, _FanContext):
                return ShkArray(val.snapshot())
            return val
        case 'implicit_chain':
            return _eval_implicit_chain(n.children, env)
        case 'listcomp':
            return _eval_listcomp(n, env)
        case 'setcomp':
            return _eval_setcomp(n, env)
        case 'setliteral':
            return _eval_setliteral(n, env)
        case 'dictcomp':
            return _eval_dictcomp(n, env)
        case 'selectorliteral':
            return eval_selectorliteral(n, env, eval_node)
        case 'group':
            return _eval_group(n, env)
        case 'ternary':
            return _eval_ternary(n, env)
        case 'rebind_primary':
            return _eval_rebind_primary(n, env)
        case 'call':
            args_node = n.children[0] if n.children else None
            args = _eval_args_node(args_node, env)
            cal = env.get('')  # unreachable in practice
            return _call_value(cal, args, env)
        case 'amp_lambda':
            return _eval_amp_lambda(n, env)
        case 'compare' | 'compare_nc':
            return _eval_compare(n.children, env)
        case 'nullish':
            return _eval_nullish(n.children, env)
        case 'nullsafe':
            return _eval_nullsafe(n.children, env)
        case 'and' | 'or' | 'and_nc' | 'or_nc':
            return _eval_logical(d, n.children, env)
        case 'walrus' | 'walrus_nc':
            return _eval_walrus(n.children, env)
        case 'assignstmt':
            return _eval_assign_stmt(n.children, env)
        case 'compound_assign':
            return _eval_compound_assign(n.children, env)
        case 'assert':
            return _eval_assert(n.children, env)
        case 'bind' | 'bind_nc':
            return _eval_apply_assign(n.children, env)
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
    meta = node_meta(node)
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
        tok_kind = _token_kind(child)
        if tok_kind in skip_tokens:
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
    if tree_label(node) == 'inlinebody':
        for child in tree_children(node):
            if tree_label(child) == 'stmtlist':
                return _eval_program(child.children, env)
        if not tree_children(node):
            return ShkNull()
        return eval_node(node.children[0], env)
    return eval_node(node, env)

def _eval_indent_block(node: Tree, env: Env) -> Any:
    return _eval_program(node.children, env)

def _eval_oneline_guard(children: List[Any], env: Env) -> Any:
    branches: List[Tree] = []
    else_body: Tree | None = None
    for child in children:
        data = tree_label(child)
        if data == 'guardbranch':
            branches.append(child)
        elif data == 'inlinebody':
            else_body = child
    outer_dot = env.dot
    for branch in branches:
        if not is_tree_node(branch) or len(branch.children) != 2:
            raise ShakarRuntimeError("Malformed guard branch")
        cond_node, body_node = branch.children
        with _temporary_subject(env, outer_dot):
            cond_val = eval_node(cond_node, env)
        if _is_truthy(cond_val):
            with _temporary_subject(env, outer_dot):
                return _eval_inline_body(body_node, env)
    if else_body is not None:
        with _temporary_subject(env, outer_dot):
            return _eval_inline_body(else_body, env)
    env.dot = outer_dot
    return ShkNull()

# ---------------- Assignment ----------------

def _eval_walrus(children: List[Any], env: Env) -> Any:
    if len(children) != 2:
        raise ShakarRuntimeError("Malformed walrus expression")
    name_node, value_node = children
    name = _expect_ident_token(name_node, "Walrus target")
    value = eval_node(value_node, env)
    return _assign_ident(name, value, env, create=True)

def _eval_assign_stmt(children: List[Any], env: Env) -> Any:
    if len(children) < 2:
        raise ShakarRuntimeError("Malformed assignment statement")
    lvalue_node = children[0]
    value_node = children[-1]
    value = eval_node(value_node, env)
    _assign_lvalue(lvalue_node, value, env, create=False)
    return ShkNull()

def _eval_assert(children: List[Any], env: Env) -> Any:
    if not children:
        raise ShakarRuntimeError("Malformed assert statement")
    cond_val = eval_node(children[0], env)
    if _is_truthy(cond_val):
        return ShkNull()
    message = f"Assertion failed: {_assert_source_snippet(children[0], env)}"
    if len(children) > 1:
        msg_val = eval_node(children[1], env)
        message = _stringify(msg_val)
    raise ShakarAssertionError(message)

def _eval_compound_assign(children: List[Any], env: Env) -> Any:
    if len(children) < 3:
        raise ShakarRuntimeError("Malformed compound assignment")
    lvalue_node = children[0]
    rhs_node = children[-1]
    op_token = None
    for child in children[1:-1]:
        if is_token_node(child):
            op_token = child
            break
    if op_token is None:
        raise ShakarRuntimeError("Compound assignment missing operator")
    op_value = op_token.value
    op_symbol = _COMPOUND_ASSIGN_OPERATORS.get(op_value)
    if op_symbol is None:
        raise ShakarRuntimeError(f"Unsupported compound operator {op_value}")
    current_value = _read_lvalue_value(lvalue_node, env)
    rhs_value = eval_node(rhs_node, env)
    new_value = _apply_binary_operator(op_symbol, current_value, rhs_value)
    _assign_lvalue(lvalue_node, new_value, env, create=False)
    return ShkNull()

def _eval_apply_assign(children: List[Any], env: Env) -> Any:
    lvalue_node = None
    rhs_node = None
    for child in children:
        if tree_label(child) == 'lvalue':
            lvalue_node = child
        elif is_tree_node(child):
            rhs_node = child
    if lvalue_node is None or rhs_node is None:
        raise ShakarRuntimeError("Malformed apply-assign expression")
    return _apply_assign(lvalue_node, rhs_node, env)

def _eval_destructure(n: Tree, env: Env, create: bool, allow_broadcast: bool) -> Any:
    if len(n.children) != 2:
        raise ShakarRuntimeError("Malformed destructure")
    pattern_list, rhs_node = n.children
    patterns = [c for c in tree_children(pattern_list) if tree_label(c) == 'pattern']
    if not patterns:
        raise ShakarRuntimeError("Empty destructure pattern")
    values, result = evaluate_destructure_rhs(eval_node, rhs_node, env, len(patterns), allow_broadcast)
    for pat, val in zip(patterns, values):
        _assign_pattern_value(pat, val, env, create, allow_broadcast)
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

def _assign_pattern_value(pattern: Tree, value: Any, env: Env, create: bool, allow_broadcast: bool) -> None:
    def _assign_ident_wrapper(name: str, val: Any, target_env: Env, create_flag: bool) -> None:
        _assign_ident(name, val, target_env, create=create_flag)
    destructure_assign_pattern(eval_node, _assign_ident_wrapper, pattern, value, env, create, allow_broadcast)

def _apply_comp_binders_wrapper(binders: list[dict[str, Any]], mode: str, element: Any, iter_env: Env, outer_env: Env) -> None:
    destructure_apply_comp_binders(
        lambda pattern, val, target_env, create, allow_broadcast: _assign_pattern_value(pattern, val, target_env, create, allow_broadcast),
        binders,
        mode,
        element,
        iter_env,
        outer_env,
    )

def _assign_lvalue(node: Any, value: Any, env: Env, create: bool) -> Any:
    if not is_tree_node(node) or tree_label(node) != 'lvalue':
        raise ShakarRuntimeError("Invalid assignment target")
    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")
    head, *ops = node.children
    if not ops and is_token_node(head) and _token_kind(head) == 'IDENT':
        return _assign_ident(head.value, value, env, create=create)
    target = eval_node(head, env)
    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case 'field' | 'fieldsel':
            field_name = _expect_ident_token(final_op.children[0], "Field assignment")
            return set_field_value(target, field_name, value, env, create=create)
        case 'lv_index':
            idx_val = _evaluate_index_operand(final_op, env)
            return set_index_value(target, idx_val, value, env)
        case 'fieldfan':
            fieldlist_node = child_by_label(final_op, 'fieldlist')
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")
            names = [tok.value for tok in fieldlist_node.children if _token_kind(tok) == 'IDENT']
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            vals = fanout_values(value, len(names))
            for name, val in zip(names, vals):
                set_field_value(target, name, val, env, create=create)
            return value
    raise ShakarRuntimeError("Unsupported assignment target")

def _read_lvalue_value(node: Any, env: Env) -> Any:
    if not is_tree_node(node) or tree_label(node) != 'lvalue':
        raise ShakarRuntimeError("Invalid lvalue")
    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")
    head, *ops = node.children
    if not ops and is_token_node(head) and _token_kind(head) == 'IDENT':
        return env.get(head.value)
    target = eval_node(head, env)
    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case 'field' | 'fieldsel':
            field_name = _expect_ident_token(final_op.children[0], "Field value access")
            return get_field_value(target, field_name, env)
        case 'lv_index':
            idx_val = _evaluate_index_operand(final_op, env)
            return index_value(target, idx_val, env)
    raise ShakarRuntimeError("Compound assignment not supported for this target")

def _apply_assign(lvalue_node: Tree, rhs_node: Tree, env: Env) -> Any:
    head, *ops = lvalue_node.children
    if not ops and _token_kind(head) == 'IDENT':
        target = env.get(head.value)
        old_val = target
        rhs_env = Env(parent=env, dot=old_val)
        new_val = eval_node(rhs_node, rhs_env)
        _assign_ident(head.value, new_val, env, create=False)
        return new_val
    target = eval_node(head, env)
    if isinstance(target, _RebindContext):
        target = target.value
    if not ops:
        raise ShakarRuntimeError("Malformed apply-assign target")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    label = tree_label(final_op)
    match label:
        case 'field' | 'fieldsel':
            field_name = _expect_ident_token(final_op.children[0], "Field assignment")
            old_val = get_field_value(target, field_name, env)
            rhs_env = Env(parent=env, dot=old_val)
            new_val = eval_node(rhs_node, rhs_env)
            set_field_value(target, field_name, new_val, env, create=False)
            return new_val
        case 'lv_index':
            idx_val = _evaluate_index_operand(final_op, env)
            old_val = index_value(target, idx_val, env)
            rhs_env = Env(parent=env, dot=old_val)
            new_val = eval_node(rhs_node, rhs_env)
            set_index_value(target, idx_val, new_val, env)
            return new_val
        case 'fieldfan':
            fieldlist_node = child_by_label(final_op, 'fieldlist')
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")
            names = [tok.value for tok in fieldlist_node.children if _token_kind(tok) == 'IDENT']
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            results: list[Any] = []
            for name in names:
                old_val = get_field_value(target, name, env)
                rhs_env = Env(parent=env, dot=old_val)
                new_val = eval_node(rhs_node, rhs_env)
                set_field_value(target, name, new_val, env, create=False)
                results.append(new_val)
            return ShkArray(results)
    raise ShakarRuntimeError("Unsupported apply-assign target")

def _evaluate_index_operand(index_node: Tree, env: Env) -> Any:
    expr_node = _index_expr_from_children(index_node.children)
    return eval_node(expr_node, env)

def _collect_free_identifiers(node: Any, callback) -> None:
    skip_nodes = {'field', 'fieldsel', 'fieldfan', 'fieldlist', 'key_ident', 'key_string'}

    def walk(n: Any) -> None:
        if is_token_node(n):
            if _token_kind(n) == 'IDENT':
                callback(n.value)
            return
        if is_tree_node(n):
            if tree_label(n) == 'amp_lambda':
                return
            if tree_label(n) in skip_nodes:
                return
            for ch in n.children:
                walk(ch)

    walk(node)

def _prepare_comprehension(n: Tree, env: Env, head_nodes: list[Any]) -> tuple[Any, list[dict[str, Any]], str, Tree | None]:
    comphead = child_by_label(n, 'comphead')
    if comphead is None:
        raise ShakarRuntimeError("Malformed comprehension")
    ifclause = child_by_label(n, 'ifclause')
    iter_expr_node, binders, mode = _parse_comphead(comphead)
    if not binders:
        implicit_names = destructure_infer_implicit_binders(
            head_nodes,
            ifclause,
            env,
            lambda expr, callback: _collect_free_identifiers(expr, callback)
        )
        for name in implicit_names:
            pattern = Tree('pattern', [Token('IDENT', name)])
            binders.append({'pattern': pattern, 'hoist': False})
    iter_val = eval_node(iter_expr_node, env)
    return iter_val, binders, mode, ifclause

def _iterate_comprehension(n: Tree, env: Env, head_nodes: list[Any]) -> Iterable[tuple[Any, Env]]:
    iter_val, binders, mode, ifclause = _prepare_comprehension(n, env, head_nodes)
    outer_dot = env.dot
    try:
        for element in _iterable_values(iter_val):
            iter_env = Env(parent=env, dot=element)
            _apply_comp_binders_wrapper(binders, mode, element, iter_env, env)
            if ifclause is not None:
                cond_node = ifclause.children[-1] if ifclause.children else None
                if cond_node is None:
                    raise ShakarRuntimeError("Malformed comprehension guard")
                cond_val = eval_node(cond_node, iter_env)
                if not _is_truthy(cond_val):
                    continue
            yield element, iter_env
    finally:
        env.dot = outer_dot

def _eval_listcomp(n: Tree, env: Env) -> ShkArray:
    body = n.children[0] if n.children else None
    if body is None:
        raise ShakarRuntimeError("Malformed list comprehension")
    items = [eval_node(body, iter_env) for _, iter_env in _iterate_comprehension(n, env, [body])]
    return ShkArray(items)

def _eval_setcomp(n: Tree, env: Env) -> ShkArray:
    body = n.children[0] if n.children else None
    if body is None:
        raise ShakarRuntimeError("Malformed set comprehension")
    items: list[Any] = []
    for _, iter_env in _iterate_comprehension(n, env, [body]):
        result = eval_node(body, iter_env)
        if not value_in_list(items, result):
            items.append(result)
    return ShkArray(items)

def _eval_setliteral(n: Tree, env: Env) -> ShkArray:
    items: list[Any] = []
    for child in tree_children(n):
        val = eval_node(child, env)
        if not value_in_list(items, val):
            items.append(val)
    return ShkArray(items)

def _eval_dictcomp(n: Tree, env: Env) -> ShkObject:
    if len(n.children) < 3:
        raise ShakarRuntimeError("Malformed dict comprehension")
    key_node = n.children[0]
    value_node = n.children[1]
    slots: dict[str, Any] = {}
    for _, iter_env in _iterate_comprehension(n, env, [key_node, value_node]):
        key_val = eval_node(key_node, iter_env)
        value_val = eval_node(value_node, iter_env)
        key_str = normalize_object_key(key_val)
        slots[key_str] = value_val
    return ShkObject(slots)

def _parse_comphead(node: Tree) -> tuple[Any, list[dict[str, Any]], str]:
    overspec = child_by_label(node, 'overspec')
    if overspec is None:
        raise ShakarRuntimeError("Malformed comprehension head")
    return _parse_overspec(overspec)

def _parse_overspec(node: Tree) -> tuple[Any, list[dict[str, Any]], str]:
    children = list(node.children)
    binders: list[dict[str, Any]] = []
    if not children:
        raise ShakarRuntimeError("Malformed overspec")
    first = children[0]
    if tree_label(first) == 'binderlist':
        mode = 'list'
        if len(children) < 2:
            raise ShakarRuntimeError("Binder list requires a source")
        iter_expr_node = children[1]
        for bp in first.children:
            bp_label = tree_label(bp)
            if bp_label == 'binderpattern' and bp.children:
                pattern_node = bp.children[0]
                pattern_label = tree_label(pattern_node)
                if pattern_label == 'pattern' and pattern_node.children:
                    child = pattern_node.children[0]
                    if tree_label(child) == 'pattern_list':
                        raise ShakarRuntimeError("Binder list cannot use parentheses")
                binders.append({'pattern': bp.children[0], 'hoist': False})
            elif bp_label == 'hoist' and bp.children:
                tok = bp.children[0]
                pattern = Tree('pattern', [tok])
                binders.append({'pattern': pattern, 'hoist': True})
        return iter_expr_node, binders, mode
    iter_expr_node = children[0]
    if len(children) > 1:
        pattern = children[1]
        binders.append({'pattern': pattern, 'hoist': False})
        mode = 'single'
    else:
        mode = 'none'
    return iter_expr_node, binders, mode

def _iterable_values(value: Any) -> list[Any]:
    match value:
        case ShkNull():
            return []
        case ShkArray(items=items):
            return list(items)
        case ShkString(value=s):
            return [ShkString(ch) for ch in s]
        case ShkObject(slots=slots):
            return [ShkString(k) for k in slots.keys()]
        case ShkSelector():
            return selector_iter_values(value)
        case _:
            raise ShakarTypeError(f"Cannot iterate over {type(value).__name__}")
# ---------------- Comparison ----------------

def _eval_compare(children: List[Any], env: Env) -> Any:
    if not children:
        return ShkNull()

    if len(children) == 1:
        return eval_node(children[0], env)

    subject = eval_node(children[0], env)
    idx = 1
    joiner = 'and'
    last_comp: Optional[str] = None
    agg: Optional[bool] = None

    while idx < len(children):
        node = children[idx]
        token_kind = _token_kind(node)
        if token_kind == 'AND':
            joiner = 'and'
            idx += 1
            continue
        if token_kind == 'OR':
            joiner = 'or'
            idx += 1
            continue

        label = tree_label(node)
        if label == 'cmpop':
            comp = _as_op(node)
            last_comp = comp
            idx += 1
            if idx >= len(children):
                raise ShakarRuntimeError("Missing right-hand side for comparison")
            rhs = eval_node(children[idx], env)
            idx += 1
        else:
            if last_comp is None:
                raise ShakarRuntimeError("Comparator required in comparison chain")
            comp = last_comp
            rhs = eval_node(node, env)
            idx += 1

        leg_val = _compare_values(comp, subject, rhs)
        if agg is None:
            agg = leg_val
        else:
            agg = (agg and leg_val) if joiner == 'and' else (agg or leg_val)

    return ShkBool(bool(agg)) if agg is not None else ShkBool(True)

def _compare_values(op: str, lhs: Any, rhs: Any) -> bool:
    if isinstance(rhs, ShkSelector):
        return _compare_with_selector(op, lhs, rhs)
    match op:
        case '==':
            return shk_equals(lhs, rhs)
        case '!=':
            return not shk_equals(lhs, rhs)
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
    return values

def _coerce_number(value: Any) -> float:
    if isinstance(value, ShkNumber):
        return value.value
    raise ShakarTypeError("Expected number")

def _compare_with_selector(op: str, lhs: Any, selector: ShkSelector) -> bool:
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

def _contains(container: Any, item: Any) -> bool:
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

def _eval_nullish(children: List[Any], env: Env) -> Any:
    exprs = [child for child in children if not (is_token_node(child) and _token_kind(child) == 'NULLISH')]
    if not exprs:
        return ShkNull()
    current = eval_node(exprs[0], env)
    for expr in exprs[1:]:
        if not isinstance(current, ShkNull):
            return current
        current = eval_node(expr, env)
    return current

def _eval_nullsafe(children: List[Any], env: Env) -> Any:
    if not children:
        return ShkNull()
    expr = children[0]
    saved_dot = env.dot
    try:
        return _eval_chain_nullsafe(expr, env)
    finally:
        env.dot = saved_dot

def _eval_chain_nullsafe(node: Any, env: Env) -> Any:
    if not is_tree_node(node) or tree_label(node) != 'explicit_chain':
        return eval_node(node, env)
    children = tree_children(node)
    if not children:
        return ShkNull()
    head = children[0]
    current = eval_node(head, env)
    if isinstance(current, ShkNull):
        return ShkNull()
    for op in children[1:]:
        try:
            current = _apply_op(current, op, env)
        except (ShakarRuntimeError, ShakarTypeError) as err:
            if _nullsafe_recovers(err, current):
                return ShkNull()
            raise
        if isinstance(current, _RebindContext):
            current = current.value
        if isinstance(current, ShkNull):
            return ShkNull()
    return current

def _nullsafe_recovers(err: Exception, recv: Any) -> bool:
    if isinstance(recv, ShkNull):
        return True
    return isinstance(err, (ShakarKeyError, ShakarIndexError))

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
    if is_token_node(node):
        return _token_kind(node) not in {'SEMI', '_NL', 'INDENT', 'DEDENT'}
    if is_tree_node(node):
        return tree_label(node) not in {'implicit_chain', 'subject', 'group'}
    return True

# ---------------- Arithmetic ----------------

def _normalize_unary_op(op_node: Any, env: Env) -> Any:
    if tree_label(op_node) == 'unaryprefixop':
        if op_node.children:
            return _normalize_unary_op(op_node.children[0], env)
        src = getattr(env, 'source', None)
        meta = node_meta(op_node)
        if src is not None and meta is not None:
            return src[meta.start_pos:meta.end_pos]
        return ''
    return op_node

def _eval_unary(op_node: Any, rhs_node: Any, env: Env) -> Any:
    op_norm = _normalize_unary_op(op_node, env)
    op_value = op_norm.value if isinstance(op_norm, Token) else op_norm

    if op_value in ('++', '--'):
        context = _resolve_assignable_node(rhs_node, env)
        _, new_val = _apply_numeric_delta(context, 1 if op_value == '++' else -1)
        return new_val

    rhs = eval_node(rhs_node, env)

    match op_norm:
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
        acc = _apply_binary_operator(op, acc, rhs)
    return acc

def _all_ops_in(children: List[Any], allowed: set) -> bool:
    for i in range(1, len(children), 2):
        op = _as_op(children[i])
        if op not in allowed:
            return False
    return True

def _as_op(x: Any) -> str:
    if is_token_node(x):
        return x.value
    label = tree_label(x)
    if label is not None:
        # e.g. Tree('addop', [Token('PLUS','+')]) or mulop/powop
        if label in ('addop', 'mulop', 'powop') and len(x.children) == 1 and is_token_node(x.children[0]):
            return x.children[0].value
        if label == 'cmpop':
            tokens = [tok.value for tok in x.children if is_token_node(tok)]
            if not tokens:
                raise ShakarRuntimeError("Empty comparison operator")
            if tokens[0] == '!' and len(tokens) > 1:
                return '!' + ''.join(tokens[1:])
            if len(tokens) > 1:
                return " ".join(tokens)
            return tokens[0]

    raise ShakarRuntimeError(f"Expected operator token, got {x!r}")

_COMPOUND_ASSIGN_OPERATORS: dict[str, str] = {
    '+=': '+',
    '-=': '-',
    '*=': '*',
    '/=': '/',
    '//=': '//',
    '%=': '%',
    '**=': '**',
}

def _stringify(value: Any) -> str:
    if isinstance(value, ShkString):
        return value.value
    inner = getattr(value, 'value', value)
    return str(inner)

def _assert_source_snippet(node: Any, env: Env) -> str:
    src = getattr(env, 'source', None)
    if src is not None:
        start, end = _node_source_span(node)
        if start is not None and end is not None and 0 <= start < end <= len(src):
            snippet = src[start:end].strip()
            if snippet:
                return snippet
    rendered = _render_expr(node)
    return rendered if rendered else "<expr>"

def _apply_binary_operator(op: str, lhs: Any, rhs: Any) -> Any:
    match op:
        case '+':
            if isinstance(lhs, ShkString) or isinstance(rhs, ShkString):
                return ShkString(_stringify(lhs) + _stringify(rhs))
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value + rhs.value)
        case '-':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value - rhs.value)
        case '*':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value * rhs.value)
        case '/':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value / rhs.value)
        case '//':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(math.floor(lhs.value / rhs.value))
        case '%':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value % rhs.value)
        case '**':
            _require_number(lhs); _require_number(rhs)
            return ShkNumber(lhs.value ** rhs.value)
    raise ShakarRuntimeError(f"Unknown operator {op}")

def _render_expr(node: Any) -> str:
    if is_token_node(node):
        return node.value
    if not is_tree_node(node):
        return str(node)
    parts: List[str] = []
    for child in tree_children(node):
        rendered = _render_expr(child)
        if rendered:
            parts.append(rendered)
    return " ".join(parts)

def _node_source_span(node: Any) -> tuple[int | None, int | None]:
    meta = node_meta(node)
    start = getattr(meta, 'start_pos', None)
    end = getattr(meta, 'end_pos', None)
    if start is not None and end is not None:
        return start, end
    if is_tree_node(node):
        child_spans = [_node_source_span(child) for child in tree_children(node)]
        child_starts = [s for s, _ in child_spans if s is not None]
        child_ends = [e for _, e in child_spans if e is not None]
        if child_starts and child_ends:
            return min(child_starts), max(child_ends)
    return None, None

def _require_number(v: Any) -> None:
    if not isinstance(v, ShkNumber):
        raise ShakarTypeError("Expected number")

def _make_ident_context(name: str, env: Env) -> _RebindContext:
    value = env.get(name)
    def setter(new_value: Any) -> None:
        _assign_ident(name, new_value, env, create=False)
        env.dot = new_value
    return _RebindContext(value, setter)

def _resolve_assignable_node(node: Any, env: Env) -> Any:
    while is_tree_node(node) and tree_label(node) in {'primary', 'group', 'group_expr'} and len(node.children) == 1:
        node = node.children[0]

    if is_token_node(node) and _token_kind(node) == 'IDENT':
        return _make_ident_context(node.value, env)

    if is_tree_node(node):
        label = tree_label(node)
        if label == 'rebind_primary':
            ctx = eval_node(node, env)
            if isinstance(ctx, _RebindContext):
                return ctx
            raise ShakarRuntimeError("Rebind primary did not produce a context")
        if label == 'explicit_chain':
            if not node.children:
                raise ShakarRuntimeError("Malformed explicit chain")
            head = node.children[0]
            ops = list(node.children[1:])
            return _resolve_chain_assignment(head, ops, env)
        if label in {'expr', 'expr_nc'} and node.children:
            return _resolve_assignable_node(node.children[0], env)

    raise ShakarRuntimeError("Increment target must be assignable")

def _resolve_chain_assignment(head_node: Any, ops: List[Any], env: Env) -> Any:
    if not ops:
        return _resolve_assignable_node(head_node, env)

    current = eval_node(head_node, env)
    if isinstance(current, _RebindContext):
        current = current.value

    for idx, op in enumerate(ops):
        is_last = idx == len(ops) - 1
        label = tree_label(op)

        if is_last and label in {'field', 'fieldsel'}:
            field_name = _expect_ident_token(op.children[0], "Field access")
            owner = current
            value = get_field_value(owner, field_name, env)
            def setter(new_value: Any, owner: Any=owner, field_name: str=field_name) -> None:
                set_field_value(owner, field_name, new_value, env, create=False)
                env.dot = new_value
            return _RebindContext(value, setter)

        if is_last and label == 'index':
            owner = current
            idx_value = _evaluate_index_operand(op, env)
            value = index_value(owner, idx_value, env)
            def setter(new_value: Any, owner: Any=owner, idx_value: Any=idx_value) -> None:
                set_index_value(owner, idx_value, new_value, env)
                env.dot = new_value
            return _RebindContext(value, setter)

        if is_last and label == 'fieldfan':
            return _build_fieldfan_context(current, op, env)

        current = _apply_op(current, op, env)
        if isinstance(current, _RebindContext):
            current = current.value

    raise ShakarRuntimeError("Increment target must end with a field or index")

def _apply_numeric_delta(ref: _RebindContext, delta: int) -> tuple[Any, Any]:
    current = ref.value
    _require_number(current)
    new_val = ShkNumber(current.value + delta)
    ref.setter(new_val)
    ref.value = new_val
    return current, new_val

def _build_fieldfan_context(owner: Any, fan_node: Tree, env: Env) -> _FanContext:
    fieldlist_node = child_by_label(fan_node, 'fieldlist')
    if fieldlist_node is None:
        raise ShakarRuntimeError("Malformed field fan")
    names = [tok.value for tok in tree_children(fieldlist_node) if is_token_node(tok) and _token_kind(tok) == 'IDENT']
    if not names:
        raise ShakarRuntimeError("Field fan requires at least one identifier")
    contexts: List[_RebindContext] = []
    for name in names:
        value = get_field_value(owner, name, env)
        def setter(new_value: Any, owner: Any=owner, field_name: str=name) -> None:
            set_field_value(owner, field_name, new_value, env, create=False)
            env.dot = new_value
        contexts.append(_RebindContext(value, setter))
    return _FanContext(contexts)

def _apply_fan_op(fan: _FanContext, op: Tree, env: Env) -> _FanContext:
    new_contexts: List[_RebindContext] = []
    new_values: List[Any] = []
    has_contexts = False
    has_values = False
    for ctx in fan.contexts:
        res = _apply_op(ctx, op, env)
        if isinstance(res, _FanContext):
            if res.contexts:
                new_contexts.extend(res.contexts)
                has_contexts = True
            else:
                new_values.extend(res.values)
                has_values = True
        elif isinstance(res, _RebindContext):
            new_contexts.append(res)
            has_contexts = True
        else:
            new_values.append(res)
            has_values = True
    if not fan.contexts and not new_contexts and not new_values:
        return fan
    if has_contexts and has_values:
        raise ShakarRuntimeError("Mixed field fan results not supported")
    if has_contexts:
        fan.contexts = new_contexts
        fan.update_from_contexts()
    else:
        fan.contexts = []
        fan.values = new_values
    return fan

# ---------------- Chains ----------------

def _apply_op(recv: Any, op: Tree, env: Env) -> Any:
    if isinstance(recv, _FanContext):
        return _apply_fan_op(recv, op, env)

    context = None
    if isinstance(recv, _RebindContext):
        context = recv
        recv = context.value

    d = op.data
    if d in {'field', 'fieldsel'}:
        field_name = _expect_ident_token(op.children[0], "Field access")
        result = get_field_value(recv, field_name, env)
    elif d == 'index':
        result = _apply_index_operation(recv, op, env)
    elif d == 'slicesel':
        result = _apply_slice(recv, op.children, env)
    elif d == 'fieldfan':
        return _build_fieldfan_context(recv, op, env)
    elif d == 'call':
        args = _eval_args_node(op.children[0] if op.children else None, env)
        result = _call_value(recv, args, env)
    elif d == 'method':
        method_name = _expect_ident_token(op.children[0], "Method call")
        args = _eval_args_node(op.children[1] if len(op.children)>1 else None, env)
        try:
            result = call_builtin_method(recv, method_name, args, env)
        except ShakarMethodNotFound:
            cal = get_field_value(recv, method_name, env)
            if isinstance(cal, BoundMethod):
                result = call_shkfn(cal.fn, args, subject=cal.subject, caller_env=env)
            elif isinstance(cal, ShkFn):
                result = call_shkfn(cal, args, subject=recv, caller_env=env)
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

def _eval_args_node(args_node: Any, env: Env) -> List[Any]:
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
                # ignore the name for now; evaluate the value node
                return flatten(node.children[-1])
        return [node]

    if is_tree_node(args_node):
        return [eval_node(n, env) for n in flatten(args_node)]
    if isinstance(args_node, list):  # deliberate: args_node can be pure python list from visitors
        res: List[Any] = []
        for n in args_node:
            res.extend(flatten(n))
        return [eval_node(n, env) for n in res]
    return []

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

def _apply_slice(recv: Any, arms: List[Any], env: Env) -> Any:
    def arm_to_py(node: Any) -> int | None:
        if tree_label(node) == 'emptyexpr':
            return None
        value = eval_node(node, env)
        if isinstance(value, ShkNumber):
            return int(value.value)
        return None
    start, stop, step = map(arm_to_py, arms)
    return slice_value(recv, start, stop, step)

# ---------------- Selectors ----------------

def _apply_index_operation(recv: Any, op: Tree, env: Env) -> Any:
    selectorlist = child_by_label(op, 'selectorlist')
    if selectorlist is None:
        expr_node = _index_expr_from_children(op.children)
        idx_val = eval_node(expr_node, env)
        return index_value(recv, idx_val, env)
    selectors = evaluate_selectorlist(selectorlist, env, eval_node)
    return apply_selectors_to_value(recv, selectors, env)

def _eval_group(n: Tree, env: Env) -> Any:
    child = n.children[0] if n.children else None
    if child is None:
        return ShkNull()
    saved = env.dot
    try:
        return eval_node(child, env)
    finally:
        env.dot = saved

def _eval_ternary(n: Tree, env: Env) -> Any:
    if len(n.children) != 3:
        raise ShakarRuntimeError("Malformed ternary expression")
    cond_node, true_node, false_node = n.children
    cond_val = eval_node(cond_node, env)
    if _is_truthy(cond_val):
        return eval_node(true_node, env)
    return eval_node(false_node, env)

def _eval_rebind_primary(n: Tree, env: Env) -> Any:
    if not n.children:
        raise ShakarRuntimeError("Missing identifier for rebind")
    target = n.children[0]
    if not is_token_node(target) or _token_kind(target) != 'IDENT':
        raise ShakarRuntimeError("Rebind target must be identifier")
    name = target.value
    value = env.get(name)

    def setter(new_value: Any) -> None:
        _assign_ident(name, new_value, env, create=False)
        env.dot = new_value

    env.dot = value
    return _RebindContext(value, setter)

def _eval_optional_expr(node: Any, env: Env) -> Any:
    if node is None:
        return None
    return eval_node(node, env)

def _call_value(cal: Any, args: List[Any], env: Env) -> Any:
    match cal:
        case BoundMethod(fn=fn, subject=subject):
            return call_shkfn(fn, args, subject=subject, caller_env=env)
        case BuiltinMethod(name=name, subject=subject):
            return call_builtin_method(subject, name, args, env)
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
        queue = list(tree_children(params_node))
        while queue:
            node = queue.pop(0)
            ident = _unwrap_ident(node)
            if ident is not None:
                names.append(ident)
                continue
            if is_tree_node(node):
                queue.extend(tree_children(node))
        return names

    def _unwrap_ident(node: Any) -> str | None:
        cur = node
        seen = set()
        while is_tree_node(cur) and cur.children and id(cur) not in seen:
            seen.add(id(cur))
            if len(cur.children) != 1:
                break
            cur = cur.children[0]
        return _ident_token_value(cur)

    def _maybe_method_signature(key_node: Any) -> tuple[str, List[str]] | None:
        if tree_label(key_node) != 'key_expr':
            return None
        target = key_node.children[0] if key_node.children else None
        chain = None
        if is_tree_node(target):
            if tree_label(target) == 'explicit_chain':
                chain = target
            else:
                for ch in tree_children(target):
                    if tree_label(ch) == 'explicit_chain':
                        chain = ch
                        break
        if chain is None or len(chain.children) != 2:
            return None
        head, call_node = chain.children
        name = _expect_ident_token(head, "Object method key")
        if tree_label(call_node) != 'call':
            return None
        args_node = call_node.children[0] if call_node.children else None
        params: List[str] = []
        if is_tree_node(args_node):
            queue = list(args_node.children)
            while queue:
                raw = queue.pop(0)
                raw_label = tree_label(raw)
                if raw_label in {'namedarg', 'kwarg'}:
                    return None
                raw_children = tree_children(raw) if is_tree_node(raw) else None
                if raw_children and raw_label not in {'args', 'arglist', 'arglistnamedmixed', 'argitem', 'arg'}:
                    # allow deeper structures by re-queueing children
                    queue.extend(raw_children)
                    continue
                ident = _unwrap_ident(raw)
                if ident is None:
                    if is_tree_node(raw):
                        queue.extend(tree_children(raw))
                    else:
                        return None
                else:
                    params.append(ident)
        elif args_node is not None:
            ident = _unwrap_ident(args_node)
            if ident is None:
                return None
            params.append(ident)
        return (name, params)

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

    for child in tree_children(n):
        if is_token_node(child):
            continue
        child_label = tree_label(child)
        if child_label == 'object_items':
            for item in tree_children(child):
                if not is_tree_node(item) or tree_label(item) == 'obj_sep':
                    continue
                handle_item(item)
        else:
            if child_label == 'obj_sep':
                continue
            if is_tree_node(child):
                handle_item(child)
    return ShkObject(slots)

def _eval_key(k: Any, env: Env) -> Any:
    label = tree_label(k)
    if label is not None:
        match label:
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

    if is_token_node(k) and _token_kind(k) in ('IDENT','STRING'):
        return k.value.strip('"').strip("'")

    return eval_node(k, env)

def _eval_amp_lambda(n: Tree, env: Env) -> ShkFn:
    if len(n.children) == 1:
        return ShkFn(params=None, body=n.children[0], env=Env(parent=env, dot=None))

    if len(n.children) == 2:
        params_node, body = n.children
        params: List[str] = []
        for p in tree_children(params_node):
            name = _ident_token_value(p)
            if name is None:
                raise ShakarRuntimeError(f"Unsupported param node in amp_lambda: {p}")
            params.append(name)
        return ShkFn(params=params, body=body, env=Env(parent=env, dot=None))

    raise ShakarRuntimeError("amp_lambda malformed")

@contextmanager
def _temporary_subject(env: Env, dot: Any) -> Iterable[None]:
    prev = env.dot
    env.dot = dot
    try:
        yield
    finally:
        env.dot = prev
