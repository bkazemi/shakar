#include "session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lexer.h"

const char *LSP_TOKEN_TYPES[] = {
    "keyword",   "variable", "number",   "string",           "comment", "operator", "function",
    "decorator", "macro",    "property", "implicit_subject", "type",    "regexp",   "punctuation",
};

const int LSP_TOKEN_TYPE_COUNT = (int)(sizeof(LSP_TOKEN_TYPES) / sizeof(LSP_TOKEN_TYPES[0]));

const char *LSP_TOKEN_MODIFIERS[] = {
    "defaultLibrary",
    "modification",
};

const int LSP_TOKEN_MOD_COUNT = (int)(sizeof(LSP_TOKEN_MODIFIERS) / sizeof(LSP_TOKEN_MODIFIERS[0]));

enum {
    MOD_DEFAULT_LIBRARY = 1 << 0,
    MOD_MODIFICATION = 1 << 1,
};

/* Indices into LSP_TOKEN_TYPES — must match the array order above. */
typedef enum {
    LSP_KW = 0,
    LSP_VAR,
    LSP_NUM,
    LSP_STR,
    LSP_COMM,
    LSP_OP,
    LSP_FN,
    LSP_ATTR,
    LSP_MACRO,
    LSP_PROP,
    LSP_SUBJ,
    LSP_TYPE,
    LSP_RE,
    LSP_PUNC
} LspTokType;

static int map_group(HlGroup group, int *type, int *mods) {
    *mods = 0;
    switch (group) {
    case HL_KEYWORD:
        *type = LSP_KW;
        return 1;
    case HL_BOOLEAN:
        *type = LSP_KW;
        *mods = MOD_DEFAULT_LIBRARY;
        return 1;
    case HL_CONSTANT:
        *type = LSP_VAR;
        *mods = MOD_DEFAULT_LIBRARY;
        return 1;
    case HL_NUMBER:
        *type = LSP_NUM;
        return 1;
    case HL_UNIT:
        *type = LSP_KW;
        *mods = MOD_MODIFICATION;
        return 1;
    case HL_STRING:
        *type = LSP_STR;
        return 1;
    case HL_REGEX:
        *type = LSP_RE;
        return 1;
    case HL_PATH:
        *type = LSP_STR;
        return 1;
    case HL_COMMENT:
        *type = LSP_COMM;
        return 1;
    case HL_OPERATOR:
        *type = LSP_OP;
        return 1;
    case HL_PUNCTUATION:
        *type = LSP_PUNC;
        return 1;
    case HL_IDENTIFIER:
        *type = LSP_VAR;
        return 1;
    case HL_FUNCTION:
        *type = LSP_FN;
        return 1;
    case HL_DECORATOR:
        *type = LSP_ATTR;
        return 1;
    case HL_HOOK:
        *type = LSP_MACRO;
        return 1;
    case HL_PROPERTY:
        *type = LSP_PROP;
        return 1;
    case HL_IMPLICIT_SUBJECT:
        *type = LSP_SUBJ;
        return 1;
    case HL_TYPE:
        *type = LSP_TYPE;
        return 1;
    default:
        return 0;
    }
}

void session_init(Session *s) {
    s->docs = 0;
    s->doc_count = 0;
    s->doc_capacity = 0;
}

static void doc_free(Document *doc) {
    free(doc->uri);
    free(doc->text);
    doc->uri = 0;
    doc->text = 0;
    doc->text_len = 0;
    doc->version = 0;
}

void session_free(Session *s) {
    for (int i = 0; i < s->doc_count; i++) {
        doc_free(&s->docs[i]);
    }
    free(s->docs);
    s->docs = 0;
    s->doc_count = 0;
    s->doc_capacity = 0;
}

static int doc_index(Session *s, const char *uri, int uri_len) {
    for (int i = 0; i < s->doc_count; i++) {
        Document *doc = &s->docs[i];
        if (doc->uri && (int)strlen(doc->uri) == uri_len && memcmp(doc->uri, uri, uri_len) == 0)
            return i;
    }
    return -1;
}

static void doc_set(Document *doc, const char *uri, int uri_len, const char *text, int text_len,
                    int version) {
    free(doc->uri);
    free(doc->text);

    doc->uri = malloc((size_t)uri_len + 1);
    if (doc->uri) {
        memcpy(doc->uri, uri, (size_t)uri_len);
        doc->uri[uri_len] = '\0';
    }

    doc->text = malloc((size_t)text_len + 1);
    if (doc->text) {
        memcpy(doc->text, text, (size_t)text_len);
        doc->text[text_len] = '\0';
        doc->text_len = text_len;
    } else {
        doc->text_len = 0;
    }

    doc->version = version;
}

void session_open(Session *s, const char *uri, int uri_len, const char *text, int text_len,
                  int version) {
    int idx = doc_index(s, uri, uri_len);
    if (idx >= 0) {
        doc_set(&s->docs[idx], uri, uri_len, text, text_len, version);
        return;
    }

    if (s->doc_count >= s->doc_capacity) {
        int       cap = s->doc_capacity ? s->doc_capacity * 2 : 16;
        Document *next = realloc(s->docs, (size_t)cap * sizeof(Document));
        if (!next)
            return;
        s->docs = next;
        s->doc_capacity = cap;
    }

    Document *doc = &s->docs[s->doc_count++];
    memset(doc, 0, sizeof(*doc));
    doc_set(doc, uri, uri_len, text, text_len, version);
}

void session_change(Session *s, const char *uri, int uri_len, const char *text, int text_len,
                    int version) {
    int idx = doc_index(s, uri, uri_len);
    if (idx < 0) {
        session_open(s, uri, uri_len, text, text_len, version);
        return;
    }
    doc_set(&s->docs[idx], uri, uri_len, text, text_len, version);
}

void session_close(Session *s, const char *uri, int uri_len) {
    int idx = doc_index(s, uri, uri_len);
    if (idx < 0)
        return;
    doc_free(&s->docs[idx]);
    int last = s->doc_count - 1;
    if (idx != last) {
        s->docs[idx] = s->docs[last];
    }
    s->doc_count--;
}

Document *session_get(Session *s, const char *uri, int uri_len) {
    int idx = doc_index(s, uri, uri_len);
    if (idx < 0)
        return 0;
    return &s->docs[idx];
}

void lsp_tokens_init(LspTokenBuf *b) {
    b->data = 0;
    b->count = 0;
    b->capacity = 0;
}

void lsp_tokens_free(LspTokenBuf *b) {
    free(b->data);
    b->data = 0;
    b->count = 0;
    b->capacity = 0;
}

static void lsp_tokens_push(LspTokenBuf *b, int v) {
    if (b->count >= b->capacity) {
        int  cap = b->capacity ? b->capacity * 2 : 256;
        int *next = realloc(b->data, (size_t)cap * sizeof(int));
        if (!next)
            return;
        b->data = next;
        b->capacity = cap;
    }
    b->data[b->count++] = v;
}

static int span_cmp(const void *a, const void *b) {
    const HlSpan *sa = (const HlSpan *)a;
    const HlSpan *sb = (const HlSpan *)b;
    if (sa->line != sb->line)
        return (sa->line < sb->line) ? -1 : 1;
    if (sa->col_start != sb->col_start)
        return (sa->col_start < sb->col_start) ? -1 : 1;
    return 0;
}

int session_build_semantic_tokens(Session *s, const char *uri, int uri_len, const LspRange *range,
                                  LspTokenBuf *out) {
    Document *doc = session_get(s, uri, uri_len);
    if (!doc || !doc->text)
        return 0;

    Lexer lex;
    HlBuf hl;
    lexer_init(&lex, doc->text, doc->text_len, 0, 1);
    lexer_tokenize(&lex);

    hlbuf_init(&hl);
    highlight(doc->text, doc->text_len, &lex.tokens, &hl);

    if (hl.count > 1)
        qsort(hl.spans, (size_t)hl.count, sizeof(HlSpan), span_cmp);

    int prev_line = 0;
    int prev_start = 0;

    for (int i = 0; i < hl.count; i++) {
        HlSpan *span = &hl.spans[i];
        int     line = span->line;
        int     start = span->col_start;
        int     end = span->col_end;

        if (range) {
            if (line < range->start_line || line > range->end_line)
                continue;
            if (line == range->start_line && end <= range->start_col)
                continue;
            if (line == range->end_line && start >= range->end_col)
                continue;
            if (line == range->start_line && start < range->start_col)
                start = range->start_col;
            if (line == range->end_line && end > range->end_col)
                end = range->end_col;
            if (end <= start)
                continue;
        }

        int type = 0;
        int mods = 0;
        if (!map_group(span->group, &type, &mods))
            continue;

        int delta_line = line - prev_line;
        int delta_start = delta_line == 0 ? start - prev_start : start;
        int length = end - start;

        lsp_tokens_push(out, delta_line);
        lsp_tokens_push(out, delta_start);
        lsp_tokens_push(out, length);
        lsp_tokens_push(out, type);
        lsp_tokens_push(out, mods);

        prev_line = line;
        prev_start = start;
    }

    hlbuf_free(&hl);
    lexer_free(&lex);
    return 1;
}

int session_lex_diagnostics(Session *s, const char *uri, int uri_len, Diagnostic *out) {
    Document *doc = session_get(s, uri, uri_len);
    if (!doc || !doc->text)
        return 0;

    Lexer lex;
    lexer_init(&lex, doc->text, doc->text_len, 1, 0);
    int r = lexer_tokenize(&lex);
    if (r >= 0) {
        lexer_free(&lex);
        return 0;
    }

    int line = lex.error_line > 0 ? lex.error_line - 1 : 0;
    int col = lex.error_col > 0 ? lex.error_col - 1 : 0;

    out->line = line;
    out->col_start = col;
    out->col_end = col + 1;
    snprintf(out->message, sizeof(out->message), "%s", lex.error_msg);

    lexer_free(&lex);
    return 1;
}
