
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional
from lark import Tree, Token

from shakar_runtime import (
    Env, ShkNumber, ShkString, ShkBool, ShkNull, ShkArray, ShkObject, Descriptor, ShkFn, BoundMethod,
    ShkSelector, SelectorIndex, SelectorSlice, SelectorPart,
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

def _is_token(node: Any) -> bool:
    return isinstance(node, Token)

def _token_kind(node: Any) -> Optional[str]:
    return node.type if isinstance(node, Token) else None

def _is_tree(node: Any) -> bool:
    return isinstance(node, Tree)

def _tree_label(node: Any) -> Optional[str]:
    return node.data if isinstance(node, Tree) else None

def _is_literal_node(node: Any) -> bool:
    return not isinstance(node, (Tree, Token))

def _child_by_label(node: Any, label: str) -> Any:
    for child in getattr(node, 'children', []):
        if _tree_label(child) == label:
            return child
    return None

def _child_by_labels(node: Any, labels: Iterable[str]) -> Any:
    label_set = set(labels)
    for child in getattr(node, 'children', []):
        if _tree_label(child) in label_set:
            return child
    return None

def _first_child(node: Any, predicate: Callable[[Any], bool]) -> Any:
    for child in getattr(node, 'children', []):
        if predicate(child):
            return child
    return None

def _get_source_segment(node: Any, env: Env) -> Optional[str]:
    source = getattr(env, 'source', None)
    if source is None:
        return None
    meta = getattr(node, "meta", None)
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
    if _is_token(n):
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
        case 'listcomp':
            return _eval_listcomp(n, env)
        case 'setcomp':
            return _eval_setcomp(n, env)
        case 'setliteral':
            return _eval_setliteral(n, env)
        case 'dictcomp':
            return _eval_dictcomp(n, env)
        case 'selectorliteral':
            return _eval_selectorliteral(n, env)
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
    if _tree_label(node) == 'inlinebody':
        for child in getattr(node, 'children', ()):
            if _tree_label(child) == 'stmtlist':
                return _eval_program(child.children, env)
        if not getattr(node, 'children', ()):
            return ShkNull()
        return eval_node(node.children[0], env)
    return eval_node(node, env)

def _eval_indent_block(node: Tree, env: Env) -> Any:
    return _eval_program(node.children, env)

def _eval_oneline_guard(children: List[Any], env: Env) -> Any:
    branches: List[Tree] = []
    else_body: Tree | None = None
    for child in children:
        data = _tree_label(child)
        if data == 'guardbranch':
            branches.append(child)
        elif data == 'inlinebody':
            else_body = child
    outer_dot = env.dot
    try:
        for branch in branches:
            if not _is_tree(branch) or len(branch.children) != 2:
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

def _eval_apply_assign(children: List[Any], env: Env) -> Any:
    lvalue_node = None
    rhs_node = None
    for child in children:
        if _tree_label(child) == 'lvalue':
            lvalue_node = child
        elif _is_tree(child):
            rhs_node = child
    if lvalue_node is None or rhs_node is None:
        raise ShakarRuntimeError("Malformed apply-assign expression")
    return _apply_assign(lvalue_node, rhs_node, env)

def _eval_destructure(n: Tree, env: Env, create: bool, allow_broadcast: bool) -> Any:
    if len(n.children) != 2:
        raise ShakarRuntimeError("Malformed destructure")
    pattern_list, rhs_node = n.children
    patterns = [c for c in getattr(pattern_list, 'children', []) if _tree_label(c) == 'pattern']
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
    if not _is_tree(node) or _tree_label(node) != 'lvalue':
        raise ShakarRuntimeError("Invalid assignment target")
    if not node.children:
        raise ShakarRuntimeError("Empty lvalue")
    head, *ops = node.children
    if not ops and _is_token(head) and _token_kind(head) == 'IDENT':
        return _assign_ident(head.value, value, env, create=create)
    target = eval_node(head, env)
    if not ops:
        raise ShakarRuntimeError("Malformed lvalue")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    label = _tree_label(final_op)
    match label:
        case 'field' | 'fieldsel':
            name_tok = final_op.children[0]
            assert _is_token(name_tok) and _token_kind(name_tok) == 'IDENT'
            return _set_field(target, name_tok.value, value, env, create=create)
        case 'lv_index':
            return _set_index(target, final_op, value, env)
        case 'fieldfan':
            fieldlist_node = _child_by_label(final_op, 'fieldlist')
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")
            names = [tok.value for tok in fieldlist_node.children if _token_kind(tok) == 'IDENT']
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            vals = _fanout_values(value, len(names))
            for name, val in zip(names, vals):
                _set_field(target, name, val, env, create=create)
            return value
    raise ShakarRuntimeError("Unsupported assignment target")

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
    if not ops:
        raise ShakarRuntimeError("Malformed apply-assign target")
    for op in ops[:-1]:
        target = _apply_op(target, op, env)
    final_op = ops[-1]
    label = _tree_label(final_op)
    match label:
        case 'field' | 'fieldsel':
            name_tok = final_op.children[0]
            assert _token_kind(name_tok) == 'IDENT'
            old_val = _get_field(target, name_tok.value, env)
            rhs_env = Env(parent=env, dot=old_val)
            new_val = eval_node(rhs_node, rhs_env)
            _set_field(target, name_tok.value, new_val, env, create=False)
            return new_val
        case 'lv_index':
            idx_expr = _index_expr_from_children(final_op.children)
            idx_val = eval_node(idx_expr, env)
            old_val = _index(target, idx_val, env)
            rhs_env = Env(parent=env, dot=old_val)
            new_val = eval_node(rhs_node, rhs_env)
            _set_index(target, final_op, new_val, env)
            return new_val
        case 'fieldfan':
            fieldlist_node = _child_by_label(final_op, 'fieldlist')
            if fieldlist_node is None:
                raise ShakarRuntimeError("Malformed field fan-out list")
            names = [tok.value for tok in fieldlist_node.children if _token_kind(tok) == 'IDENT']
            if not names:
                raise ShakarRuntimeError("Empty field fan-out list")
            results: list[Any] = []
            for name in names:
                old_val = _get_field(target, name, env)
                rhs_env = Env(parent=env, dot=old_val)
                new_val = eval_node(rhs_node, rhs_env)
                _set_field(target, name, new_val, env, create=False)
                results.append(new_val)
            return ShkArray(results)
    raise ShakarRuntimeError("Unsupported apply-assign target")

def _evaluate_destructure_rhs(rhs_node: Any, env: Env, target_count: int, allow_broadcast: bool) -> tuple[list[Any], Any]:
    if _tree_label(rhs_node) == 'pack':
        vals = [eval_node(child, env) for child in rhs_node.children]
        result = ShkArray(vals)
    else:
        single = eval_node(rhs_node, env)
        vals = [single]
        result = single
    if len(vals) == 1 and target_count > 1:
        single = vals[0]
        if _is_sequence_value(single):
            items = _sequence_items(single)
            if len(items) == target_count:
                vals = list(items)
                result = ShkArray(vals)
            elif len(items) == 0 and allow_broadcast:
                replicated = _replicate_empty_sequence(single, target_count)
                vals = replicated
                result = ShkArray(vals)
            else:
                raise ShakarRuntimeError("Destructure arity mismatch")
        elif allow_broadcast:
            vals = [single] * target_count
        else:
            raise ShakarRuntimeError("Destructure arity mismatch")
    elif len(vals) != target_count:
        raise ShakarRuntimeError("Destructure arity mismatch")
    return vals, result

def _assign_pattern(pattern: Tree, value: Any, env: Env, create: bool, allow_broadcast: bool) -> None:
    if _tree_label(pattern) != 'pattern' or not getattr(pattern, 'children', None):
        raise ShakarRuntimeError("Malformed pattern")
    target = pattern.children[0]
    if isinstance(target, Token) and target.type == 'IDENT':
        if create:
            env.define(target.value, value)
        else:
            _assign_ident(target.value, value, env, create=False)
        return
    if _tree_label(target) == 'pattern_list':
        subpatterns = [c for c in getattr(target, 'children', []) if _tree_label(c) == 'pattern']
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

def _normalize_object_key(value: Any) -> str:
    match value:
        case ShkString(value=s):
            return s
        case ShkNumber(value=num):
            return str(int(num)) if num.is_integer() else str(num)
        case ShkBool(value=b):
            return 'true' if b else 'false'
        case ShkNull():
            return 'null'
        case _:
            return str(value)

def _infer_implicit_binders(exprs: list[Any], ifclause: Tree | None, env: Env) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def consider(name: str) -> None:
        if name in seen:
            return
        if _name_exists(env, name):
            return
        seen.add(name)
        names.append(name)

    for expr in exprs:
        _collect_free_identifiers(expr, consider)
    if ifclause is not None and ifclause.children:
        guard_expr = ifclause.children[-1]
        _collect_free_identifiers(guard_expr, consider)
    return names

def _collect_free_identifiers(node: Any, callback) -> None:
    skip_nodes = {'field', 'fieldsel', 'fieldfan', 'fieldlist', 'key_ident', 'key_string'}

    def walk(n: Any) -> None:
        if isinstance(n, Token):
            if n.type == 'IDENT':
                callback(n.value)
            return
        if _is_tree(n):
            if _tree_label(n) == 'amp_lambda':
                return
            if _tree_label(n) in skip_nodes:
                return
            for ch in n.children:
                walk(ch)

    walk(node)

def _name_exists(env: Env, name: str) -> bool:
    try:
        env.get(name)
        return True
    except ShakarRuntimeError:
        return False

def _prepare_comprehension(n: Tree, env: Env, head_nodes: list[Any]) -> tuple[Any, list[dict[str, Any]], str, Tree | None]:
    comphead = _child_by_label(n, 'comphead')
    if comphead is None:
        raise ShakarRuntimeError("Malformed comprehension")
    ifclause = _child_by_label(n, 'ifclause')
    iter_expr_node, binders, mode = _parse_comphead(comphead)
    if not binders:
        implicit_names = _infer_implicit_binders(head_nodes, ifclause, env)
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
            _apply_comp_binders(binders, mode, element, iter_env, env)
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
        if not _value_in_list(items, result):
            items.append(result)
    return ShkArray(items)

def _eval_setliteral(n: Tree, env: Env) -> ShkArray:
    items: list[Any] = []
    for child in getattr(n, 'children', []):
        val = eval_node(child, env)
        if not _value_in_list(items, val):
            items.append(val)
    return ShkArray(items)

# ---------------- Selector semantics ----------------

def _eval_selectorliteral(n: Tree, env: Env) -> ShkSelector:
    sellist = _child_by_label(n, 'sellist')
    if sellist is None:
        return ShkSelector([])
    parts: list[SelectorPart] = []
    for item in getattr(sellist, 'children', []):
        label = _tree_label(item)
        if label == 'selitem':
            parts.extend(_selector_parts_from_selitem(item, env))
        else:
            parts.extend(_selector_parts_from_selitem(Tree('selitem', [item]), env))
    return ShkSelector(parts)

def _eval_dictcomp(n: Tree, env: Env) -> ShkObject:
    if len(n.children) < 3:
        raise ShakarRuntimeError("Malformed dict comprehension")
    key_node = n.children[0]
    value_node = n.children[1]
    slots: dict[str, Any] = {}
    for _, iter_env in _iterate_comprehension(n, env, [key_node, value_node]):
        key_val = eval_node(key_node, iter_env)
        value_val = eval_node(value_node, iter_env)
        key_str = _normalize_object_key(key_val)
        slots[key_str] = value_val
    return ShkObject(slots)

def _parse_comphead(node: Tree) -> tuple[Any, list[dict[str, Any]], str]:
    overspec = _child_by_label(node, 'overspec')
    if overspec is None:
        raise ShakarRuntimeError("Malformed comprehension head")
    return _parse_overspec(overspec)

def _parse_overspec(node: Tree) -> tuple[Any, list[dict[str, Any]], str]:
    children = list(node.children)
    binders: list[dict[str, Any]] = []
    if not children:
        raise ShakarRuntimeError("Malformed overspec")
    first = children[0]
    if _tree_label(first) == 'binderlist':
        mode = 'list'
        if len(children) < 2:
            raise ShakarRuntimeError("Binder list requires a source")
        iter_expr_node = children[1]
        for bp in first.children:
            bp_label = _tree_label(bp)
            if bp_label == 'binderpattern' and bp.children:
                pattern_node = bp.children[0]
                pattern_label = _tree_label(pattern_node)
                if pattern_label == 'pattern' and pattern_node.children:
                    child = pattern_node.children[0]
                    if _tree_label(child) == 'pattern_list':
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

def _apply_comp_binders(binders: list[dict[str, Any]], mode: str, element: Any, iter_env: Env, outer_env: Env) -> None:
    if not binders:
        return
    if len(binders) == 1:
        values = [element]
    else:
        seq = _coerce_sequence(element, len(binders))
        if seq is None:
            raise ShakarRuntimeError("Comprehension element arity mismatch")
        values = seq
    for binder, val in zip(binders, values):
        target_env = outer_env if binder.get('hoist') else iter_env
        _assign_pattern(binder['pattern'], val, target_env, create=True, allow_broadcast=False)

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
            return _selector_iter_values(value)
        case _:
            if isinstance(value, list):
                return list(value)
            if isinstance(value, tuple):
                return list(value)
            raise ShakarTypeError(f"Cannot iterate over {type(value).__name__}")

def _value_in_list(seq: list[Any], value: Any) -> bool:
    for existing in seq:
        if _shk_equals(existing, value):
            return True
    return False

# ---------------- Sequence helpers ----------------

def _is_sequence_value(value: Any) -> bool:
    return isinstance(value, (ShkArray, list, tuple))

def _sequence_items(value: Any) -> list[Any]:
    if isinstance(value, ShkArray):
        return list(value.items)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []

def _coerce_sequence(value: Any, expected_len: int | None) -> list[Any] | None:
    if not _is_sequence_value(value):
        return None
    items = _sequence_items(value)
    if expected_len is not None and len(items) != expected_len:
        raise ShakarRuntimeError("Destructure arity mismatch")
    return items

def _fanout_values(value: Any, count: int) -> list[Any]:
    if isinstance(value, ShkArray) and len(value.items) == count:
        return list(value.items)
    if isinstance(value, list) and len(value) == count:
        return list(value)
    return [value] * count

def _replicate_empty_sequence(value: Any, count: int) -> list[Any]:
    if isinstance(value, ShkArray) and len(value.items) == 0:
        return [ShkArray([]) for _ in range(count)]
    if isinstance(value, list) and len(value) == 0:
        return [[] for _ in range(count)]
    if isinstance(value, tuple) and len(value) == 0:
        return [tuple() for _ in range(count)]
    return [value] * count

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
    if _is_tree(node):
        return _tree_label(node) not in {'implicit_chain', 'subject'}
    return True

# ---------------- Arithmetic ----------------

def _normalize_unary_op(op_node: Any, env: Env) -> Any:
    if _tree_label(op_node) == 'unaryprefixop':
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
    label = _tree_label(x)
    if label is not None:
        # e.g. Tree('addop', [Token('PLUS','+')]) or mulop/powop
        if label in ('addop', 'mulop', 'powop') and len(x.children) == 1 and isinstance(x.children[0], Token):
            return x.children[0].value
        if label == 'cmpop':
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
            return _apply_index_operation(recv, op, env)
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
        if _is_tree(node):
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

    if _is_tree(args_node):
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
            if isinstance(idx, ShkSelector):
                cloned = _clone_selector_parts(idx.parts, clamp=True)
                return _apply_selectors_to_array(recv, cloned, env)
            if isinstance(idx, ShkNumber):
                return items[int(idx.value)]
            raise ShakarTypeError("Array index must be a number")
        case ShkString(value=s):
            if isinstance(idx, ShkSelector):
                cloned = _clone_selector_parts(idx.parts, clamp=True)
                return _apply_selectors_to_string(recv, cloned, env)
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
        if not _is_tree(node):
            return node
        tag = _tree_label(node)
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
        if _tree_label(t) == 'emptyexpr':
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

# ---------------- Selectors ----------------

def _apply_index_operation(recv: Any, op: Tree, env: Env) -> Any:
    selectorlist = _child_by_label(op, 'selectorlist')
    if selectorlist is None:
        expr_node = _index_expr_from_children(op.children)
        idx_val = eval_node(expr_node, env)
        return _index(recv, idx_val, env)
    selectors = _evaluate_selectorlist(selectorlist, env)
    return _apply_selectors_to_value(recv, selectors, env)

def _evaluate_selectorlist(node: Tree, env: Env) -> list[SelectorPart]:
    results: list[SelectorPart] = []
    for raw_selector in getattr(node, 'children', []):
        inner = _child_by_labels(raw_selector, {'slicesel', 'indexsel'}) or raw_selector
        label = _tree_label(inner)
        if label == 'slicesel':
            results.append(_selector_slice_from_slicesel(inner, env, clamp=True))
            continue
        if label == 'indexsel':
            expr_node = _first_child(inner, lambda child: not isinstance(child, Token))
            if expr_node is None and inner.children:
                expr_node = inner.children[0]
            value = eval_node(expr_node, env)
            results.extend(_expand_selector_value(value, clamp=True))
            continue
        # fallback: evaluate entire node
        value = eval_node(inner, env)
        results.extend(_expand_selector_value(value, clamp=True))
    return results

def _selector_parts_from_selitem(node: Tree, env: Env) -> list[SelectorPart]:
    inner = _child_by_labels(node, {'sliceitem', 'indexitem'})
    if inner is None:
        return []
    label = _tree_label(inner)
    if label == 'sliceitem':
        slice_part = _selector_slice_from_sliceitem(inner, env)
        return [slice_part]
    if label == 'indexitem':
        selatom = _child_by_label(inner, 'selatom')
        value = _eval_selector_atom(selatom, env)
        return [SelectorIndex(value)]
    return []

def _selector_slice_from_sliceitem(node: Tree, env: Env) -> SelectorSlice:
    children = list(getattr(node, 'children', []))
    index = 0
    start_node = None
    if index < len(children) and _tree_label(children[index]) == 'selatom':
        start_node = children[index]
        index += 1
    stop_node = None
    if index < len(children) and _tree_label(children[index]) == 'seloptstop':
        stop_node = children[index]
        index += 1
    step_node = None
    if index < len(children) and _tree_label(children[index]) == 'selatom':
        step_node = children[index]
    start_val = _coerce_selector_number(_eval_selector_atom(start_node, env), allow_none=True)
    stop_value, exclusive = _eval_seloptstop(stop_node, env)
    stop_val = _coerce_selector_number(stop_value, allow_none=True)
    step_val = _coerce_selector_number(_eval_selector_atom(step_node, env), allow_none=True)
    return SelectorSlice(start=start_val, stop=stop_val, step=step_val, clamp=False, exclusive_stop=exclusive)

def _selector_slice_from_slicesel(node: Tree, env: Env, clamp: bool) -> SelectorSlice:
    children = list(getattr(node, 'children', []))
    start_node = children[0] if len(children) > 0 else None
    stop_node = children[1] if len(children) > 1 else None
    step_node = children[2] if len(children) > 2 else None

    def unwrap(expr_node):
        if _tree_label(expr_node) == 'emptyexpr':
            return None
        return expr_node

    start_val = _coerce_selector_number(_eval_optional_expr(unwrap(start_node), env), allow_none=True)
    stop_val = _coerce_selector_number(_eval_optional_expr(unwrap(stop_node), env), allow_none=True)
    step_val = _coerce_selector_number(_eval_optional_expr(unwrap(step_node), env), allow_none=True)
    return SelectorSlice(start=start_val, stop=stop_val, step=step_val, clamp=clamp, exclusive_stop=True)

def _eval_optional_expr(node: Any, env: Env) -> Any:
    if node is None:
        return None
    return eval_node(node, env)

def _eval_selector_atom(node: Any, env: Env) -> Any:
    if node is None:
        return None
    if not _is_tree(node):
        return eval_node(node, env)
    if not getattr(node, 'children', None):
        return eval_node(node, env)
    child = node.children[0]
    if _tree_label(child) == 'interp':
        expr = next((grand for grand in getattr(child, 'children', []) if _tree_label(grand) == 'expr'), None)
        if expr is None and getattr(child, 'children', None):
            expr = child.children[0]
        if expr is None:
            raise ShakarRuntimeError("Empty interpolation in selector literal")
        return eval_node(expr, env)
    if _is_tree(child):
        return eval_node(child, env)
    return eval_node(child, env)

def _eval_seloptstop(node: Any, env: Env) -> tuple[Any, bool]:
    if node is None:
        return None, False
    selatom = _child_by_label(node, 'selatom')
    value = _eval_selector_atom(selatom, env)
    segment = _get_source_segment(node, env)
    exclusive = False
    if segment is not None:
        exclusive = segment.lstrip().startswith('<')
    return value, exclusive

def _coerce_selector_number(value: Any, allow_none: bool=False) -> Optional[int]:
    if value is None:
        if allow_none:
            return None
        raise ShakarTypeError("Selector expects a numeric bound")
    if isinstance(value, ShkNull):
        if allow_none:
            return None
        raise ShakarTypeError("Selector expects a numeric bound")
    if isinstance(value, ShkNumber):
        num = value.value
    elif isinstance(value, (int, float)):
        num = float(value)
    else:
        raise ShakarTypeError("Selector expects a numeric bound")
    if not float(num).is_integer():
        raise ShakarTypeError("Selector bounds must be integral")
    return int(num)

def _expand_selector_value(value: Any, clamp: bool) -> list[SelectorPart]:
    if isinstance(value, ShkSelector):
        return _clone_selector_parts(value.parts, clamp)
    return [SelectorIndex(value)]

def _clone_selector_parts(parts: Iterable[SelectorPart], clamp: bool) -> list[SelectorPart]:
    cloned: list[SelectorPart] = []
    for part in parts:
        if isinstance(part, SelectorSlice):
            cloned.append(SelectorSlice(
                start=part.start,
                stop=part.stop,
                step=part.step,
                clamp=clamp,
                exclusive_stop=part.exclusive_stop,
            ))
        elif isinstance(part, SelectorIndex):
            cloned.append(SelectorIndex(part.value))
    return cloned

def _apply_selectors_to_value(recv: Any, selectors: list[SelectorPart], env: Env) -> Any:
    if not selectors:
        return ShkNull()
    if isinstance(recv, ShkArray):
        return _apply_selectors_to_array(recv, selectors, env)
    if isinstance(recv, ShkString):
        return _apply_selectors_to_string(recv, selectors, env)
    if len(selectors) == 1 and isinstance(selectors[0], SelectorIndex):
        return _index(recv, selectors[0].value, env)
    raise ShakarTypeError("Complex selectors only supported on arrays or strings")

def _apply_selectors_to_array(arr: ShkArray, selectors: list[SelectorPart], env: Env) -> ShkArray:
    result: list[Any] = []
    items = _sequence_items(arr)
    length = len(items)
    for part in selectors:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            pos = _normalize_index_position(idx, length)
            if pos < 0 or pos >= length:
                raise ShakarRuntimeError("Array index out of bounds")
            result.append(items[pos])
            continue
        slice_obj = _selector_slice_to_slice(part, length)
        result.extend(items[slice_obj])
    return ShkArray(result)

def _apply_selectors_to_string(s: ShkString, selectors: list[SelectorPart], env: Env) -> ShkString:
    pieces: list[str] = []
    length = len(s.value)
    for part in selectors:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            pos = _normalize_index_position(idx, length)
            if pos < 0 or pos >= length:
                raise ShakarRuntimeError("String index out of bounds")
            pieces.append(s.value[pos])
            continue
        slice_obj = _selector_slice_to_slice(part, length)
        pieces.append(s.value[slice_obj])
    return ShkString("".join(pieces))

def _selector_index_to_int(value: Any) -> int:
    if isinstance(value, ShkNumber):
        num = value.value
    elif isinstance(value, (int, float)):
        num = float(value)
    else:
        raise ShakarTypeError("Index selector expects a numeric value")
    if not float(num).is_integer():
        raise ShakarTypeError("Index selector expects an integer value")
    return int(num)

def _normalize_index_position(index: int, length: int) -> int:
    if index < 0:
        return index + length
    return index

def _selector_slice_to_slice(part: SelectorSlice, length: int) -> slice:
    step = part.step if part.step is not None else 1
    if step == 0:
        raise ShakarRuntimeError("Slice step cannot be zero")
    start = part.start
    stop = part.stop
    if part.clamp:
        stop_adj = stop
        if not part.exclusive_stop and stop_adj is not None:
            stop_adj = stop_adj + (1 if step > 0 else -1)
        slice_obj = slice(start, stop_adj, step)
        start_idx, stop_idx, step_idx = slice_obj.indices(length)
        return slice(start_idx, stop_idx, step_idx)
    if start is None:
        start = 0 if step > 0 else length - 1
    if stop is None:
        stop = length if step > 0 else -1
    if not part.exclusive_stop and stop is not None:
        stop = stop + (1 if step > 0 else -1)
    if start < 0:
        start += length
    if stop is not None and stop < 0:
        stop += length
    return slice(start, stop, step)

def _selector_iter_values(selector: ShkSelector) -> list[Any]:
    items: list[Any] = []
    for part in selector.parts:
        if isinstance(part, SelectorIndex):
            idx = _selector_index_to_int(part.value)
            items.append(ShkNumber(float(idx)))
            continue
        items.extend(ShkNumber(float(i)) for i in _iterate_selector_slice(part))
    return items

def _iterate_selector_slice(part: SelectorSlice) -> Iterable[int]:
    step = part.step if part.step is not None else 1
    if step == 0:
        raise ShakarRuntimeError("Selector slice step cannot be zero")
    if part.start is None or part.stop is None:
        raise ShakarRuntimeError("Selector slice requires explicit start and stop when iterated")
    start = part.start
    stop = part.stop
    stop_exclusive = stop if part.exclusive_stop else stop + (1 if step > 0 else -1)
    return range(start, stop_exclusive, step)

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
            if _is_tree(node):
                queue.extend(node.children)
        return names

    def _unwrap_ident(node: Any) -> str | None:
        cur = node
        seen = set()
        while _is_tree(cur) and cur.children and id(cur) not in seen:
            seen.add(id(cur))
            if len(cur.children) != 1:
                break
            cur = cur.children[0]
        if isinstance(cur, Token) and cur.type == 'IDENT':
            return cur.value
        return None

    def _maybe_method_signature(key_node: Any) -> tuple[str, List[str]] | None:
        if _tree_label(key_node) != 'key_expr':
            return None
        target = key_node.children[0] if key_node.children else None
        chain = None
        if _is_tree(target):
            if _tree_label(target) == 'explicit_chain':
                chain = target
            else:
                for ch in getattr(target, 'children', []):
                    if _tree_label(ch) == 'explicit_chain':
                        chain = ch
                        break
        if chain is None or len(chain.children) != 2:
            return None
        head, call_node = chain.children
        if not isinstance(head, Token) or head.type != 'IDENT':
            return None
        if _tree_label(call_node) != 'call':
            return None
        args_node = call_node.children[0] if call_node.children else None
        params: List[str] = []
        if _is_tree(args_node):
            queue = list(args_node.children)
            while queue:
                raw = queue.pop(0)
                raw_label = _tree_label(raw)
                if raw_label in {'namedarg', 'kwarg'}:
                    return None
                if _is_tree(raw) and raw.children and raw_label not in {'args', 'arglist', 'arglistnamedmixed', 'argitem', 'arg'}:
                    # allow deeper structures by re-queueing children
                    queue.extend(raw.children)
                    continue
                ident = _unwrap_ident(raw)
                if ident is None:
                    if _is_tree(raw):
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
        child_label = _tree_label(child)
        if child_label == 'object_items':
            for item in getattr(child, 'children', []):
                if not _is_tree(item) or _tree_label(item) == 'obj_sep':
                    continue
                handle_item(item)
        else:
            if child_label == 'obj_sep':
                continue
            if _is_tree(child):
                handle_item(child)
    return ShkObject(slots)

def _eval_key(k: Any, env: Env) -> Any:
    label = _tree_label(k)
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
