# Canonical AST Reference

This document describes the canonical AST as consumed by the evaluator. The canonical tree is the **reference parser output after normalization passes**, not the raw syntax tree or the grammar rules.

## Source of truth

- **Parser:** `src/shakar_ref/parser_rd.py`
- **Normalization:** `src/shakar_ref/ast_transforms.py` (notably `Prune`)
- **Runtime expectations:** `src/shakar_ref/evaluator.py`
- **Executable checks:** `tests/test_ast.py` and related parser/runtime pytest coverage under `tests/`

If these disagree, the runtime behavior wins.

## Conventions

- Node names are lowercase (`body`, `rebind_primary`, `fieldfan`, etc.).
- Semantic metadata is carried in `Tree.attrs` and preserved by transforms.
- Attributes are optional; missing attributes should be treated as falsey defaults.

### Body nodes

All block/inline bodies normalize to a **single node name**:

- `body` with `attrs={'inline': True}` for inline bodies
- `body` with `attrs={'inline': False}` for indented blocks

Examples (shape only):

```python
Tree("body", [Tree("stmtlist", [...])], attrs={"inline": True})
Tree("body", [Tok("INDENT", ...), Tree("stmt", [...]), Tok("DEDENT", ...)], attrs={"inline": False})
```

### Rebind primary

`rebind_primary` distinguishes grouped vs simple forms via attrs:

- `=ident` => `Tree("rebind_primary", [Tok("IDENT", ...)], attrs={"grouped": False})`
- `=(lvalue)` => `Tree("rebind_primary", [Tree("lvalue", ...)], attrs={"grouped": True})`

The evaluator relies on `grouped` to determine chain semantics.

### Field fan in assignment

Assignment fan-out (`.{a, b} = ...` or `.{a, b} .= ...`) is normalized so
`fieldfan` always contains a `fieldlist` of `IDENT` tokens (no `valuefan_list`):

```python
Tree("fieldfan", [Tree("fieldlist", [Tok("IDENT", "a"), Tok("IDENT", "b")])])
```

This is enforced during lvalue parsing; runtime field fan logic assumes it.

## Notes

- `grammar.ebnf` documents **syntax**, not the canonical AST.
- `grammar.lark` is legacy and retained for historical reference.
