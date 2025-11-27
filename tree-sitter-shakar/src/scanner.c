// External scanner for Shakar: emits NEWLINE, INDENT, DEDENT.
// Adapted from tree-sitter-python with queued indentation tokens so a
// NEWLINE can be followed by INDENT/DEDENT when the grammar requests it.

#include <tree_sitter/parser.h>
#include <wctype.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

enum TokenType { NEWLINE, INDENT, DEDENT };

typedef struct {
  uint32_t *stack;
  size_t size;
  size_t cap;
  uint32_t pending_indent;
  uint32_t pending_dedent;
  bool pending_newline;
} Scanner;

static inline void push(Scanner *s, uint32_t v) {
  if (s->size == s->cap) {
    s->cap = s->cap ? s->cap * 2 : 8;
    s->stack = realloc(s->stack, s->cap * sizeof(uint32_t));
  }
  s->stack[s->size++] = v;
}

static inline uint32_t top(Scanner *s) { return s->stack[s->size - 1]; }
static inline void pop(Scanner *s) { if (s->size > 1) s->size--; }

static bool emit_pending(Scanner *s, TSLexer *lexer, const bool *valid) {
  if (s->pending_newline) {
    if (valid[NEWLINE]) {
      lexer->result_symbol = NEWLINE;
      s->pending_newline = false;
      return true;
    }
    s->pending_newline = false; // drop if parser does not want it
  }

  if (s->pending_indent) {
    if (valid[INDENT]) {
      push(s, s->pending_indent);
      s->pending_indent = 0;
      lexer->result_symbol = INDENT;
      return true;
    }
  }

  if (s->pending_dedent) {
    if (valid[DEDENT]) {
      pop(s);
      s->pending_dedent--;
      lexer->result_symbol = DEDENT;
      return true;
    }
  }

  return false;
}

void *tree_sitter_shakar_external_scanner_create() {
  Scanner *s = calloc(1, sizeof(Scanner));
  push(s, 0);
  return s;
}

void tree_sitter_shakar_external_scanner_destroy(void *payload) {
  Scanner *s = (Scanner *)payload;
  free(s->stack);
  free(s);
}

unsigned tree_sitter_shakar_external_scanner_serialize(void *payload, char *buffer) {
  Scanner *s = (Scanner *)payload;
  uint32_t *buf = (uint32_t *)buffer;
  size_t count = 0;

  buf[count++] = (uint32_t)s->size;
  memcpy(buf + count, s->stack, s->size * sizeof(uint32_t));
  count += s->size;

  buf[count++] = s->pending_indent;
  buf[count++] = s->pending_dedent;
  buf[count++] = s->pending_newline ? 1u : 0u;

  return (unsigned)(count * sizeof(uint32_t));
}

void tree_sitter_shakar_external_scanner_deserialize(void *payload, const char *buffer, unsigned length) {
  Scanner *s = (Scanner *)payload;
  free(s->stack); s->stack = NULL; s->size = s->cap = 0;
  s->pending_indent = 0;
  s->pending_dedent = 0;
  s->pending_newline = false;

  if (length < sizeof(uint32_t)) {
    push(s, 0);
    return;
  }

  const uint32_t *buf = (const uint32_t *)buffer;
  unsigned total = length / sizeof(uint32_t);
  size_t stack_count = buf[0];
  if (stack_count == 0) stack_count = 1;

  for (size_t i = 0; i < stack_count && 1 + i < total; i++) {
    push(s, buf[1 + i]);
  }

  size_t idx = 1 + stack_count;
  if (idx < total) s->pending_indent = buf[idx++];
  if (idx < total) s->pending_dedent = buf[idx++];
  if (idx < total) s->pending_newline = buf[idx++] != 0;

  if (s->size == 0) push(s, 0);
}

static inline bool is_newline(int32_t c){ return c=='\n' || c=='\r'; }
static inline bool is_space(int32_t c){ return c==' ' || c=='\t'; }

static bool scan(void *payload, TSLexer *lexer, const bool *valid) {
  Scanner *s = (Scanner *)payload;
  if (!(valid[NEWLINE] || valid[INDENT] || valid[DEDENT])) return false;

  if (emit_pending(s, lexer, valid)) return true;

  if (lexer->lookahead == 0) {
    if (s->size > 1) {
      s->pending_dedent = (uint32_t)(s->size - 1);
      return emit_pending(s, lexer, valid);
    }
    return false;
  }

  if (!is_newline(lexer->lookahead)) return false;

  lexer->advance(lexer, true); // consume the newline
  lexer->mark_end(lexer);

  for (;;) {
    uint32_t col = 0;
    while (is_space(lexer->lookahead)) {
      col += lexer->lookahead == '\t' ? 8 : 1;
      lexer->advance(lexer, true);
    }

    if (lexer->lookahead == '#') {
      // Leave the comment to the main lexer; treat it as a non-blank line at the
      // current indentation level so it can be highlighted and counted as a statement.
      s->pending_newline = true;
      s->pending_indent = 0;
      s->pending_dedent = 0;

      uint32_t cur = top(s);
      if (col > cur) {
        s->pending_indent = col;
      } else if (col < cur) {
        size_t dedents = 0;
        while (dedents < s->size && s->stack[s->size - 1 - dedents] > col) dedents++;
        s->pending_dedent = (uint32_t)dedents;
      }

      return emit_pending(s, lexer, valid);
    }

    s->pending_newline = true;
    s->pending_indent = 0;
    s->pending_dedent = 0;

    uint32_t cur = top(s);
    if (col > cur) {
      s->pending_indent = col;
    } else if (col < cur) {
      size_t dedents = 0;
      while (dedents < s->size && s->stack[s->size - 1 - dedents] > col) dedents++;
      s->pending_dedent = (uint32_t)dedents;
    }

    if (emit_pending(s, lexer, valid)) return true;
    return false;
  }
}

bool tree_sitter_shakar_external_scanner_scan(void *payload, TSLexer *lexer, const bool *valid_symbols) {
  return scan(payload, lexer, valid_symbols);
}
