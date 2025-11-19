# Reference Runtime (`src/shakar_ref/`)

The `src/shakar_ref/` package hosts the core Python implementation of the parser, lowering pass, and evaluator:

- `parse_auto.py` – auto-generated parser/pruner glue to turn Lark trees into canonical nodes.
- `lower.py` – AST normalization (hole desugaring, etc.).
- `evaluator.py` – walks the canonical AST and executes programs.
- `runtime.py` – value model (`ShkNumber`, `ShkString`, etc.), frames, and stdlib registry.
- `eval/` – helper modules for selectors, mutation, destructuring, and await logic used by the evaluator.
- `runner.py` – convenience entry point (`PYTHONPATH=src python -m shakar_ref.runner program.shk`) that wires parse → prune → eval.

Developer notes:

- The repo targets Python 3.10+. Use a virtualenv (`python -m venv .venv && source .venv/bin/activate`) so imports like `lark`/`tree_sitter` resolve cleanly. When running modules directly, prepend `PYTHONPATH=src` (or install the package in editable mode) so `shakar_ref` resolves.
- The stdlib (`stdlib.py`) registers its functions via `register_stdlib` and is imported defensively so `print()` and future additions are always available.
