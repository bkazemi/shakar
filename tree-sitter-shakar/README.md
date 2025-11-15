# tree-sitter-shakar

Early Tree-sitter grammar for the Shakar language. This grammar is a hand-port of the current Lark specification and focuses on providing syntax highlighting coverage across the language surface while the language is still evolving.

## Layout

- `grammar.js` – core Tree-sitter rules.
- `queries/highlights.scm` – highlight captures (Neovim, Helix, Zed, etc.).
- `tests/` – future regression corpus (currently empty).
- `package.json` – npm metadata for generating/building the parser.

## Getting Started

```
cd tree-sitter-shakar
npm install
npx tree-sitter generate
npx tree-sitter test
```

Generated artifacts (`src/parser.c`, `src/grammar.json`, `src/node-types.json`, etc.) are created by the commands above. Regenerate them whenever the grammar changes and keep the files under version control so downstream editors can consume the parser without a local build step.

## Alignment with the reference grammar

This grammar trails the reference Lark parser intentionally. The top-level `README.md` (root of the repo) describes the latest runtime features (decorators, comma-chain comparisons, subjectful statements, etc.); when those features land in the runtime you should mirror them here if syntax support is required for highlighting or Tree-sitter-based tooling.

Changes that touch syntax should follow this loop:

1. Update `grammar.js` and any relevant query files.
2. Regenerate the C parser (`npx tree-sitter generate`).
3. Run the corpus (`npx tree-sitter test`).
4. Open a PR that includes both the JS grammar changes and the regenerated artifacts.

The `tests/` directory is intentionally empty right now. When guard/selector/decorator samples stabilize, add them there so `tree-sitter test` guards against regressions.

## Roadmap / Caveats

- Indentation-aware block parsing is not implemented yet; the current grammar treats colon bodies as single inline expressions or explicit `{ … }` blocks. Expect false positives for multi-line blocks until an indentation scanner is added.
- Comprehension/guard coverage matches the current evaluator but will continue to evolve alongside the language grammar.
- Additional queries (`folds.scm`, `indents.scm`) are planned once the grammar stabilizes.
- Decorator syntax exists in the runtime; the Tree-sitter grammar will pick it up once editor tooling needs it. Until then, `@decorator` lines parse as attribute-style modifiers without semantic meaning.

Bug reports and PRs are very welcome while we shake out the edges of the syntax. When filing an issue, please mention whether the discrepancy is in the runtime, Tree-sitter, or both—keeping them in sync is a work-in-progress.
