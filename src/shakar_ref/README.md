# Reference Runtime (`src/shakar_ref/`)

The `src/shakar_ref/` package hosts the core Python implementation of the parser, lowering pass, and evaluator:

- `parser_rd.py` – recursive-descent parser producing AST from source.
- `lexer_rd.py` – lexer with indentation-aware tokenization.
- `ast_transforms.py` – Prune transform that normalizes raw parse trees.
- `lower.py` – AST lowering (hole desugaring, amp-lambda parameter inference).
- `evaluator.py` – walks the canonical AST and executes programs.
- `runtime.py` – value model (`ShkNumber`, `ShkString`, etc.), frames, and stdlib registry.
- `tree.py` – local Tree/Tok implementations (Lark-compatible).
- `eval/` – helper modules for selectors, mutation, destructuring, and channel logic used by the evaluator.
- `runner.py` – convenience entry point (`PYTHONPATH=src python -m shakar_ref.runner program.shk`) that wires parse → prune → lower → eval.

Developer notes:

- **C port friendly**: Language features must be implementable in C — closures, objects, generators, etc. should map to well-understood C patterns (environment structs, tagged unions, state machines). Don't design features that assume Python's runtime model (heavy reflection, metaclasses, eval). Implementation idioms matter less; they'll be rewritten during the port anyway.
- The repo targets Python 3.10+. Use a virtualenv (`python -m venv .venv && source .venv/bin/activate`). When running modules directly, prepend `PYTHONPATH=src` (or install the package in editable mode) so `shakar_ref` resolves.
- The stdlib (`stdlib.py`) registers its functions via `register_stdlib` and is imported defensively so `print()` and future additions are always available.
- Legacy Lark-based parser lives in `lark.old/parse_auto.py` (deprecated, only used by tree validation tests).
