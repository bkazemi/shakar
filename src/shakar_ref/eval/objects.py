from __future__ import annotations

from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple

from ..tree import Node, Tok
from ..token_types import TT

from ..runtime import (
    Descriptor,
    Frame,
    ShkFn,
    ShkObject,
    ShkString,
    ShkValue,
    ShakarRuntimeError,
    ShakarTypeError,
)
from ..types import Builtins
from ..tree import Tree, is_token, is_tree, tree_children, tree_label
from ..utils import normalize_object_key
from .common import (
    expect_ident_token as _expect_ident_token,
    extract_function_signature,
)
from .fn import _inject_contract_assertions
from .helpers import closure_frame, eval_anchor_scoped

EvalFunc = Callable[[Node, Frame], ShkValue]


def _normalize_literal_key(key: ShkValue | str) -> str:
    if isinstance(key, str):
        return key
    return normalize_object_key(key)


def _install_descriptor(
    slots: dict[str, ShkValue],
    name: str,
    getter: Optional[ShkFn] = None,
    setter: Optional[ShkFn] = None,
) -> None:
    existing = slots.get(name)

    if isinstance(existing, Descriptor):
        if getter is not None:
            existing.getter = getter
        if setter is not None:
            existing.setter = setter
        slots[name] = existing
        return

    slots[name] = Descriptor(getter=getter, setter=setter)


def _unwrap_ident(node: Optional[Node]) -> Optional[str]:
    if node is None:
        return None

    cur = node
    seen = set()

    while is_tree(cur) and cur.children and id(cur) not in seen:
        seen.add(id(cur))

        if len(cur.children) != 1:
            break
        cur = cur.children[0]
    return cur.value if isinstance(cur, Tok) and cur.type == TT.IDENT else None


def _maybe_method_signature(
    key_node: Optional[Node],
) -> Optional[Tuple[str, List[str]]]:
    if key_node is None or tree_label(key_node) != "key_expr":
        return None
    if not is_tree(key_node):
        return None

    target = key_node.children[0] if key_node.children else None
    chain: Optional[Tree] = None

    if target is not None and is_tree(target):
        if tree_label(target) == "explicit_chain":
            chain = target
        else:
            chain = next(
                (
                    ch
                    for ch in tree_children(target)
                    if is_tree(ch) and tree_label(ch) == "explicit_chain"
                ),
                None,
            )

    if chain is None or len(chain.children) != 2:
        return None

    head, call_node = chain.children
    name = _expect_ident_token(head, "Object method key")

    if not is_tree(call_node) or tree_label(call_node) != "call":
        return None

    args_node = call_node.children[0] if call_node.children else None
    params: List[str] = []

    if args_node is not None and is_tree(args_node):
        queue = deque(args_node.children)

        while queue:
            raw = queue.popleft()
            raw_label = tree_label(raw)

            if raw_label in {"namedarg", "kwarg"}:
                return None

            raw_children = tree_children(raw) if is_tree(raw) else None
            if raw_children and raw_label in {
                "args",
                "arglist",
                "arglistnamedmixed",
                "argitem",
                "arg",
            }:
                queue.extend(raw_children)
                continue

            ident = _unwrap_ident(raw)
            if ident is not None:
                params.append(ident)
                continue

            return None
    elif args_node is not None:
        ident = _unwrap_ident(args_node)
        if ident is None:
            return None
        params.append(ident)
    return (name, params)


def _handle_obj_field(
    item: Tree,
    slots: dict[str, ShkValue],
    frame: Frame,
    eval_func: EvalFunc,
) -> None:
    key_node, val_node = item.children
    if not is_tree(key_node):
        raise ShakarRuntimeError("Object field missing key")
    method_sig = _maybe_method_signature(key_node)

    if method_sig is not None:
        name, params = method_sig
        slots[name] = ShkFn(
            params=params,
            body=val_node,
            frame=closure_frame(frame),
            name=name,
        )
        return

    key = eval_key(key_node, frame, eval_func)
    val = eval_anchor_scoped(val_node, frame, eval_func)
    slots[_normalize_literal_key(key)] = val


def _handle_obj_field_optional(
    item: Tree,
    slots: dict[str, ShkValue],
    frame: Frame,
    eval_func: EvalFunc,
) -> None:
    # Hygienic sugar: resolve Optional directly from builtins, bypassing scope
    # so user-defined Optional cannot shadow it.
    key_node, val_node = item.children
    if not is_tree(key_node):
        raise ShakarRuntimeError("Object optional field missing key")
    key = eval_key(key_node, frame, eval_func)
    val = eval_anchor_scoped(val_node, frame, eval_func)
    optional_fn = Builtins.stdlib_functions.get("Optional")
    if optional_fn is None:
        raise ShakarRuntimeError("Builtin 'Optional' not found")
    slots[_normalize_literal_key(key)] = optional_fn.fn(frame, None, [val], None)


def _handle_obj_get(item: Tree, slots: dict[str, ShkValue], frame: Frame) -> None:
    name_tok, body = item.children
    if name_tok is None:
        raise ShakarRuntimeError("Getter missing name")

    key = _expect_ident_token(name_tok, "Getter name")
    getter_fn = ShkFn(
        params=None,
        body=body,
        frame=closure_frame(frame),
        name=f"{key}.get",
    )
    _install_descriptor(slots, key, getter=getter_fn)


def _handle_obj_set(item: Tree, slots: dict[str, ShkValue], frame: Frame) -> None:
    name_tok, param_tok, body = item.children
    if name_tok is None or param_tok is None:
        raise ShakarRuntimeError("Setter missing name or parameter")

    key = _expect_ident_token(name_tok, "Setter name")
    param_name = _expect_ident_token(param_tok, "Setter parameter")
    setter_fn = ShkFn(
        params=[param_name],
        body=body,
        frame=closure_frame(frame),
        name=f"{key}.set",
    )
    _install_descriptor(slots, key, setter=setter_fn)


def _handle_obj_method(item: Tree, slots: dict[str, ShkValue], frame: Frame) -> None:
    name_tok, params_node, body = item.children
    if name_tok is None:
        raise ShakarRuntimeError("Method missing name")

    method_name = _expect_ident_token(name_tok, "Method name")
    param_names, varargs, defaults, contracts, spread_contracts = (
        extract_function_signature(params_node, context="method definition")
    )
    final_body = (
        _inject_contract_assertions(body, contracts, spread_contracts)
        if contracts or spread_contracts
        else body
    )
    slots[method_name] = ShkFn(
        params=param_names,
        body=final_body,
        frame=closure_frame(frame),
        vararg_indices=varargs,
        param_defaults=defaults,
        name=method_name,
    )


def _handle_obj_spread(
    item: Tree,
    slots: dict[str, ShkValue],
    frame: Frame,
    eval_func: EvalFunc,
) -> None:
    spread_expr = item.children[0] if item.children else None
    if spread_expr is None:
        raise ShakarRuntimeError("Object spread missing value")

    spread_val = eval_anchor_scoped(spread_expr, frame, eval_func)
    if not isinstance(spread_val, ShkObject):
        raise ShakarTypeError("Object spread expects an object value")

    for key, val in spread_val.slots.items():
        slots[key] = val


def _handle_object_item(
    item: Tree,
    slots: dict[str, ShkValue],
    frame: Frame,
    eval_func: EvalFunc,
) -> None:
    match item.data:
        case "obj_field":
            _handle_obj_field(item, slots, frame, eval_func)
        case "obj_field_optional":
            _handle_obj_field_optional(item, slots, frame, eval_func)
        case "obj_get":
            _handle_obj_get(item, slots, frame)
        case "obj_set":
            _handle_obj_set(item, slots, frame)
        case "obj_method":
            _handle_obj_method(item, slots, frame)
        case "obj_spread":
            _handle_obj_spread(item, slots, frame, eval_func)
        case _:
            raise ShakarRuntimeError(f"Unknown object item {item.data}")


def _iter_object_items(n: Tree) -> Iterator[Tree]:
    for child in tree_children(n):
        if is_token(child):
            continue

        child_label = tree_label(child)
        if child_label == "object_items":
            for item in tree_children(child):
                if not is_tree(item) or tree_label(item) == "obj_sep":
                    continue
                yield item
            continue

        if child_label == "obj_sep":
            continue
        if is_tree(child):
            yield child


def eval_object(n: Tree, frame: Frame, eval_func: EvalFunc) -> ShkObject:
    """Build an object literal, installing descriptors/getters as needed."""
    slots: dict[str, ShkValue] = {}
    for item in _iter_object_items(n):
        _handle_object_item(item, slots, frame, eval_func)
    return ShkObject(slots)


def eval_key(k: Tree, frame: Frame, eval_func: EvalFunc) -> ShkValue | str:
    label = tree_label(k)

    if label is not None:
        match label:
            case "key_ident":
                t = k.children[0]
                if isinstance(t, Tok) and t.type == TT.IDENT:
                    return str(t.value)
                return eval_anchor_scoped(k, frame, eval_func)
            case "key_string":
                t = k.children[0]
                if not isinstance(t, Tok) or t.type != TT.STRING:
                    return eval_anchor_scoped(k, frame, eval_func)
                s = str(t.value)

                if len(s) >= 2 and (
                    (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
                ):
                    s = s[1:-1]
                return s
            case "key_expr":
                v = eval_anchor_scoped(k.children[0], frame, eval_func)
                return v.value if isinstance(v, ShkString) else v

    if is_token(k):
        if k.type == TT.IDENT:
            return str(k.value)

        if k.type == TT.STRING:
            s = str(k.value)
            if len(s) >= 2 and (
                (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
            ):
                return s[1:-1]
            return s

    return eval_anchor_scoped(k, frame, eval_func)
