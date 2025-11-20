# Repository Guidelines

## Project Structure & Module Organization
- `src/shakar_ref/`: Hosts the reference parser, lowering pass, and evaluator modules. Pure utilities live here.
- `grammar.lark`: The authoritative Lark grammar; must stay aligned with tree-sitter variants.
- `tree-sitter-shakar/`: Holds the tree-sitter grammar and bindings.
- `docs/`: Design/conformance notes. Update when syntax/semantics change.
- `test/`: Quick regression snippets.
- `tests/`: Editor/integration scripts.

## Build, Test, and Development Commands
- `python sanity_check_basic.py`: Builds parser, writes `sanity_report.txt`. Fails on grammar regressions. **Run this before/after every refactor.**
- `python sanity_treecheck.py`: Validates parse-tree invariants.
- `PYTHONPATH=src python -m shakar_ref.runner test/testinp5`: Runs the interpreter (accepts path or `-` for stdin).
- `make -C tree-sitter-shakar test`: Runs tree-sitter corpus suite (requires CLI).

## Architecture & Refactoring Philosophy
- **Clarity > Cleverness:** Prefer explicit logic over complex one-liners. Do not use nested ternary operators.
- **State Management:** In parser/evaluator, prefer passing explicit context objects (e.g., `Frame`) over global state or deep recursion arguments.
- **Cleanup:** When refactoring logic, **DELETE** the old implementation. Do not leave commented-out blocks unless they are specific WIP tests.
- **Cognitive Load:** If a function exceeds ~50 lines, propose extracting logical phases into named helper functions.
- **Safety:** Review `sanity_report.txt` diffs to ensure refactoring did not alter parser output.

## Coding Style & Naming
- **Conventions:** Python uses 4-space indent, `snake_case` functions, `CapWords` classes. Grammar rules are `lowercase_with_underscores`; terminals are `UPPERCASE`.
- **Types:** Strict typing. Prefer `Frame | None` over `Optional[Frame]`. No `Any` unless strictly necessary.
- **Imports:** Explicit imports preferred over implicit globals.
- **Readability:** Use blank lines to separate logical phases within functions, and before `return`/`raise` statements.
- **Docstrings:** Keep them tight; no blank line after opening quotes. Code begins immediately after.

## Testing & Commits
- **Coverage:** Extend parser coverage via `sanity_check_basic.py` generators, not ad-hoc loops.
- **Parity:** Mirror key regression snippets in `tree-sitter-shakar/test/corpus`.
- **Commits:** Use Conventional Commits (`feat(rt):`, `fix(grammar):`). Keep grammar/runtime changes in the same commit as their regenerated artifacts.
- **PRs:** Summarize behavioral changes and list validation commands run.
- **Automation guard:** Never create commits unless the user explicitly asks for one. Keep changes staged/uncommitted by default.
