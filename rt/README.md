# Runtime (`rt/`)

This directory hosts the core Python runtime for Shakar:

- `shakar_eval.py` – evaluator that walks the canonical AST and executes programs.
- `shakar_runtime.py` – value model (`ShkNumber`, `ShkString`, etc.), environments, and stdlib registry.
- `shakar_parse_auto.py` – auto‑generated parser/pruner glue to turn Lark trees into canonical nodes.
- `rt/eval/` – helper modules for selectors, mutation, and destructuring logic used by the evaluator.
- `shakar_run.py` – convenience entry point (`python rt/shakar_run.py program.shk`) that wires parse → prune → eval.

Developer notes:

- The repo targets Python 3.10+. Use a virtualenv (`python -m venv .venv && source .venv/bin/activate`) so imports like `lark`/`tree_sitter` resolve cleanly.
- The stdlib (`rt/shakar_stdlib.py`) registers its functions via `register_stdlib` and is imported defensively so `print()` and future additions are always available.
