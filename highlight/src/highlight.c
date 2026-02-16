/*
 * Shakar highlight — lexical + structural (token-stream) highlighting.
 *
 * Mirrors highlight_server.py: _tokens_to_highlights, _classify_identifier,
 * _classify_string, _duration_size_spans, _token_spans.
 */

#include "highlight.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================== */
/* HlBuf                                                                     */
/* ======================================================================== */

void hlbuf_init(HlBuf *b) {
    b->spans = 0;
    b->count = 0;
    b->capacity = 0;
}

void hlbuf_free(HlBuf *b) {
    free(b->spans);
    b->spans = 0;
    b->count = 0;
    b->capacity = 0;
}

static void hlbuf_push(HlBuf *b, int line, int cs, int ce, HlGroup g) {
    if (ce <= cs)
        return;
    if (b->count >= b->capacity) {
        int cap = b->capacity ? b->capacity * 2 : 256;
        b->spans = realloc(b->spans, cap * sizeof(HlSpan));
        b->capacity = cap;
    }
    HlSpan *s = &b->spans[b->count++];
    s->line = line;
    s->col_start = cs;
    s->col_end = ce;
    s->group = g;
}

/* ======================================================================== */
/* Group name table                                                          */
/* ======================================================================== */

static const char *GROUP_NAMES[] = {
    "", /* HL_NONE */
    "keyword", "boolean",  "constant",         "number",      "unit",       "string",   "regex",
    "path",    "comment",  "operator",         "punctuation", "identifier", "function", "decorator",
    "hook",    "property", "implicit_subject", "type",
};

const char *hl_group_name(HlGroup g) {
    if (g >= 0 && g < HL__COUNT)
        return GROUP_NAMES[g];
    return "";
}

/* ======================================================================== */
/* Token => base group mapping                                                */
/* ======================================================================== */

static HlGroup base_group(TT t) {
    switch (t) {
    /* Keywords */
    case TT_IF:
    case TT_ELIF:
    case TT_ELSE:
    case TT_UNLESS:
    case TT_WHILE:
    case TT_FOR:
    case TT_MATCH:
    case TT_IN:
    case TT_BREAK:
    case TT_CONTINUE:
    case TT_RETURN:
    case TT_FN:
    case TT_LET:
    case TT_WAIT:
    case TT_SPAWN:
    case TT_USING:
    case TT_CALL:
    case TT_DEFER:
    case TT_AFTER:
    case TT_THROW:
    case TT_CATCH:
    case TT_TRY:
    case TT_ASSERT:
    case TT_DBG:
    case TT_DECORATOR:
    case TT_GET:
    case TT_SET:
    case TT_HOOK:
    case TT_OVER:
    case TT_BIND:
    case TT_IMPORT:
    case TT_FAN:
    case TT_NOT:
    case TT_AND:
    case TT_OR:
    case TT_IS:
        return HL_KEYWORD;

    case TT_TRUE:
    case TT_FALSE:
        return HL_BOOLEAN;
    case TT_NIL:
        return HL_CONSTANT;

    case TT_NUMBER:
    case TT_DURATION:
    case TT_SIZE:
        return HL_NUMBER;

    case TT_STRING:
    case TT_RAW_STRING:
    case TT_RAW_HASH_STRING:
    case TT_SHELL_STRING:
    case TT_SHELL_BANG_STRING:
    case TT_ENV_STRING:
        return HL_STRING;

    case TT_REGEX:
        return HL_REGEX;
    case TT_PATH_STRING:
        return HL_PATH;
    case TT_COMMENT:
        return HL_COMMENT;

    case TT_IDENT:
        return HL_IDENTIFIER;

    /* Operators */
    case TT_PLUS:
    case TT_MINUS:
    case TT_STAR:
    case TT_SLASH:
    case TT_FLOORDIV:
    case TT_MOD:
    case TT_POW:
    case TT_DEEPMERGE:
    case TT_CARET:
    case TT_EQ:
    case TT_NEQ:
    case TT_LT:
    case TT_LTE:
    case TT_GT:
    case TT_GTE:
    case TT_SEND:
    case TT_RECV:
    case TT_NEG:
    case TT_NULLISH:
    case TT_ASSIGN:
    case TT_WALRUS:
    case TT_APPLYASSIGN:
    case TT_PLUSEQ:
    case TT_MINUSEQ:
    case TT_STAREQ:
    case TT_SLASHEQ:
    case TT_FLOORDIVEQ:
    case TT_MODEQ:
    case TT_POWEQ:
    case TT_INCR:
    case TT_DECR:
    case TT_TILDE:
    case TT_REGEXMATCH:
    case TT_AMP:
    case TT_PIPE:
        return HL_OPERATOR;

    /* Punctuation */
    case TT_LPAR:
    case TT_RPAR:
    case TT_LSQB:
    case TT_RSQB:
    case TT_LBRACE:
    case TT_RBRACE:
    case TT_DOT:
    case TT_COMMA:
    case TT_COLON:
    case TT_SEMI:
    case TT_QMARK:
    case TT_AT:
    case TT_DOLLAR:
    case TT_BACKQUOTE:
        return HL_PUNCTUATION;

    /* Layout / special — no highlight */
    default:
        return HL_NONE;
    }
}

/* ======================================================================== */
/* Structural classification                                                 */
/* ======================================================================== */

static int is_layout(TT t) {
    return t == TT_NEWLINE || t == TT_INDENT || t == TT_DEDENT || t == TT_EOF || t == TT_SEMI;
}

static int is_line_blank_or_comment(const char *src, int start, int end) {
    int i = start;
    while (i < end) {
        char ch = src[i];
        if (ch == ' ' || ch == '\t') {
            i++;
            continue;
        }
        if (ch == '#' || ch == '\r' || ch == '\n')
            return 1;
        return 0;
    }
    return 1;
}

static int line_start_for_pos(const char *src, int pos) {
    int i = pos;
    while (i > 0) {
        char ch = src[i - 1];
        if (ch == '\n' || ch == '\r')
            break;
        i--;
    }
    return i;
}

static int line_indent_width(const char *src, int line_start, int src_len, int *first_non_ws_out) {
    int i = line_start;
    int indent = 0;
    while (i < src_len) {
        char ch = src[i];
        if (ch == ' ')
            indent += 1;
        else if (ch == '\t')
            indent += 8;
        else
            break;
        i++;
    }
    if (first_non_ws_out)
        *first_non_ws_out = i;
    return indent;
}

/* Cache line-local prefix state so callers can avoid rescanning long lines
 * for every token. */
static void line_prefix_info_for_pos(const char *src, int src_len, int pos, int *line_start_out,
                                     int *indent_out, int *first_non_ws_out) {
    int line_start = line_start_for_pos(src, pos);
    int p = line_start;
    int indent = line_indent_width(src, line_start, src_len, &p);

    if (line_start_out)
        *line_start_out = line_start;
    if (indent_out)
        *indent_out = indent;
    if (first_non_ws_out)
        *first_non_ws_out = p;
}

/* Find previous non-blank, non-comment line ending before line_start. */
static int find_prev_nonblank_line(const char *src, int line_start, int *start_out, int *end_out) {
    int i = line_start;
    while (i > 0) {
        int line_end = i;
        if (line_end > 0 && src[line_end - 1] == '\n') {
            line_end--;
            if (line_end > 0 && src[line_end - 1] == '\r')
                line_end--;
        } else if (line_end > 0 && src[line_end - 1] == '\r') {
            line_end--;
        }

        int line_begin = line_end;
        while (line_begin > 0 && src[line_begin - 1] != '\n' && src[line_begin - 1] != '\r')
            line_begin--;
        if (!is_line_blank_or_comment(src, line_begin, line_end)) {
            if (start_out)
                *start_out = line_begin;
            if (end_out)
                *end_out = line_end;
            return 1;
        }
        i = line_begin;
    }
    return 0;
}

static int prev_nonblank_indent(const char *src, int line_start) {
    int line_begin = 0;
    int line_end = 0;
    if (!find_prev_nonblank_line(src, line_start, &line_begin, &line_end))
        return -1;
    return line_indent_width(src, line_begin, line_end, 0);
}

static int prev_nonblank_line_info(const char *src, int line_start, int *indent_out) {
    int line_begin = 0;
    int line_end = 0;
    if (!find_prev_nonblank_line(src, line_start, &line_begin, &line_end))
        return 0;
    if (indent_out)
        *indent_out = line_indent_width(src, line_begin, line_end, 0);
    return 1;
}

static int is_expr_end(TT t) {
    switch (t) {
    case TT_IDENT:
    case TT_NUMBER:
    case TT_DURATION:
    case TT_SIZE:
    case TT_STRING:
    case TT_RAW_STRING:
    case TT_RAW_HASH_STRING:
    case TT_SHELL_STRING:
    case TT_SHELL_BANG_STRING:
    case TT_ENV_STRING:
    case TT_REGEX:
    case TT_PATH_STRING:
    case TT_TRUE:
    case TT_FALSE:
    case TT_NIL:
    case TT_RPAR:
    case TT_RSQB:
    case TT_RBRACE:
    case TT_INCR:
    case TT_DECR:
        return 1;
    default:
        return 0;
    }
}

static int token_text_equals(const char *src, int src_len, const Tok *tok, const char *text,
                             int text_len) {
    if (tok->len != text_len)
        return 0;
    if (tok->start < 0 || tok->len < 0)
        return 0;
    if (tok->start > src_len - text_len)
        return 0;
    return memcmp(src + tok->start, text, (size_t)text_len) == 0;
}

static int is_slot_head_keyword(TT t) {
    switch (t) {
    case TT_WAIT:
    case TT_USING:
    case TT_CALL:
    case TT_FAN:
        return 1;
    default:
        return 0;
    }
}

/* ---- Bracket / layout predicates ---- */

static int is_open_group(TT t) { return t == TT_LPAR || t == TT_LSQB || t == TT_LBRACE; }

static int is_close_group(TT t) { return t == TT_RPAR || t == TT_RSQB || t == TT_RBRACE; }

/* Tokens that are invisible to statement-level scanning. */
static int is_scan_layout(TT t) { return t == TT_COMMENT || t == TT_INDENT || t == TT_DEDENT; }

/* Whether the next significant token stays in the same statement as current.
 * NEWLINE only splits statements outside bracketed grouping.
 * SEMI always splits statements. */
static int next_sig_same_statement(const Tok *toks, int cur_sig_idx, int next_sig_idx,
                                   int group_depth_after_cur) {
    for (int i = cur_sig_idx + 1; i < next_sig_idx; i++) {
        TT t = toks[i].type;
        if (is_scan_layout(t))
            continue;
        if (t == TT_SEMI)
            return 0;
        if (t == TT_NEWLINE && group_depth_after_cur == 0)
            return 0;
    }
    return 1;
}

/* ---- Statement-level depth-tracked walker ---- */

/* Visitor callback for walk_stmt_depth0.  Return non-zero to stop. */
typedef int (*StmtVisitFn)(TT type, int tok_idx, void *ctx);

/* Walk significant tokens within a statement, tracking bracket depth.
 * Scans from toks[start] in direction dir (+1 forward, -1 backward).
 * Direction-aware depth: forward => ( opens, ) closes; backward => ) opens, ( closes.
 * Calls visit (if non-NULL) for each non-bracket, non-layout token at depth 0.
 * Stops at statement boundary (NEWLINE/SEMI at depth 0), array edge,
 * or when visit returns non-zero.
 * If stop_idx is non-NULL, set to the boundary token index (-1 if no boundary hit).
 * Returns the visitor's non-zero result, or 0. */
static int walk_stmt_depth0(const Tok *toks, int tc, int start, int dir, int initial_depth,
                            StmtVisitFn visit, void *ctx, int *stop_idx) {
    int depth = initial_depth;
    if (stop_idx)
        *stop_idx = -1;

    for (int i = start; i >= 0 && i < tc; i += dir) {
        TT t = toks[i].type;
        if (is_scan_layout(t))
            continue;

        if ((dir > 0) ? is_open_group(t) : is_close_group(t)) {
            depth++;
            continue;
        }
        if ((dir > 0) ? is_close_group(t) : is_open_group(t)) {
            if (depth > 0)
                depth--;
            continue;
        }

        if (depth == 0 && (t == TT_NEWLINE || t == TT_SEMI)) {
            if (stop_idx)
                *stop_idx = i;
            return 0;
        }

        if (depth == 0 && visit) {
            int r = visit(t, i, ctx);
            if (r)
                return r;
        }
    }

    return 0;
}

/* ---- Statement scanning helpers (thin wrappers over walk_stmt_depth0) ---- */

/* Find the token index immediately after the previous statement boundary. */
static int stmt_start_idx(const Tok *toks, int cur_sig_idx, int group_depth_before_cur) {
    int boundary = -1;
    walk_stmt_depth0(toks, cur_sig_idx, cur_sig_idx - 1, -1, group_depth_before_cur, NULL, NULL,
                     &boundary);

    return (boundary >= 0) ? boundary + 1 : 0;
}

/* Visitor: stop on first colon. */
static int visit_is_colon(TT type, int idx, void *ctx) {
    (void)idx;
    (void)ctx;
    return type == TT_COLON;
}

/* Is there a statement-level colon after cur_sig_idx within this statement? */
static int stmt_has_next_colon(const Tok *toks, int tc, int cur_sig_idx, int group_depth_after) {
    return walk_stmt_depth0(toks, tc, cur_sig_idx + 1, +1, group_depth_after, visit_is_colon, NULL,
                            NULL);
}

/* Context for scanning previous colons, excluding ternary/catch/selector colons. */
typedef struct {
    int ternary_qmarks;
    int catch_pending_colon;
    int in_selector_literal;
    TT last_depth0_sig;
} PrevColonCtx;

static int visit_prev_colon(TT type, int idx, void *ctx) {
    PrevColonCtx *c = ctx;
    (void)idx;

    /* Selector literals: everything between backquotes is opaque. */
    if (type == TT_BACKQUOTE) {
        c->in_selector_literal = !c->in_selector_literal;
        c->last_depth0_sig = type;
        return 0;
    }
    if (c->in_selector_literal)
        return 0;

    if (type == TT_QMARK) {
        c->ternary_qmarks++;
    } else if (type == TT_CATCH) {
        c->catch_pending_colon = 1;
    } else if (type == TT_AT && c->last_depth0_sig == TT_AT) {
        c->catch_pending_colon = 1;
    } else if (type == TT_COLON) {
        if (c->ternary_qmarks > 0) {
            c->ternary_qmarks--;
        } else if (c->catch_pending_colon) {
            c->catch_pending_colon = 0;
        } else {
            return 1; /* found a statement-level colon */
        }
    }

    c->last_depth0_sig = type;
    return 0;
}

/* Is there a non-expression statement-level colon before cur_sig_idx
 * within this statement?  Ternary, catch, and selector-literal colons
 * are excluded. */
static int stmt_has_prev_colon(const Tok *toks, int cur_sig_idx, int group_depth_before_cur) {
    int start = stmt_start_idx(toks, cur_sig_idx, group_depth_before_cur);
    PrevColonCtx ctx = {0, 0, 0, TT_EOF};

    /* Forward scan from statement start to cur_sig_idx (exclusive).
     * Depth starts at 0 since start is after a statement boundary. */
    return walk_stmt_depth0(toks, cur_sig_idx, start, +1, 0, visit_prev_colon, &ctx, NULL);
}

/* Classify identifier based on prev/next significant token. */
static HlGroup classify_ident(const char *src, int src_len, Tok *tok, Tok *next_sig, TT prev_sig,
                              TT prev_prev_sig, int same_stmt_next) {
    if (prev_sig == TT_QMARK && token_text_equals(src, src_len, tok, "ret", 3))
        return HL_KEYWORD;

    /* Modifier/binder bracket slot: wait[mod], using[name], call[name], fan[mod]. */
    if (prev_sig == TT_LSQB && next_sig && next_sig->type == TT_RSQB) {
        if (is_slot_head_keyword(prev_prev_sig)) {
            return HL_KEYWORD;
        }
    }

    if (prev_sig == TT_FN)
        return HL_FUNCTION;
    if (prev_sig == TT_DECORATOR || prev_sig == TT_AT)
        return HL_DECORATOR;
    if (prev_sig == TT_HOOK)
        return HL_HOOK;
    if (prev_sig == TT_GET || prev_sig == TT_SET)
        return HL_PROPERTY;
    if (prev_sig == TT_DOT) {
        /* Method call only within the same statement: .method() => function, .prop => property */
        if (same_stmt_next && next_sig && next_sig->type == TT_LPAR)
            return HL_FUNCTION;
        return HL_PROPERTY;
    }
    if (prev_sig == TT_TILDE)
        return HL_TYPE;

    /* Contextual keywords: timeout, default */
    if (token_text_equals(src, src_len, tok, "timeout", 7))
        return HL_KEYWORD;
    if (token_text_equals(src, src_len, tok, "default", 7))
        return HL_KEYWORD;

    /* Call-site: ident followed by ( */
    if (next_sig && next_sig->type == TT_LPAR)
        return HL_FUNCTION;

    return HL_IDENTIFIER;
}

/* Classify string based on previous significant token. */
static HlGroup classify_string(TT prev_sig) {
    if (prev_sig == TT_HOOK)
        return HL_HOOK;
    return HL_STRING;
}

/* ======================================================================== */
/* Duration/size sub-span splitting                                          */
/* ======================================================================== */

/* Emit number+unit sub-spans for duration/size tokens.
 * Returns 1 if spans were emitted, 0 if not. */
static int emit_dur_size_spans(const char *src, Tok *tok, HlBuf *out) {
    const char *text = src + tok->start;
    int tlen = tok->len;
    int line = tok->line - 1;
    int base_col = tok->col - 1;

    int pos = 0;
    int emitted = 0;

    while (pos < tlen) {
        /* Skip leading digits (number part) */
        int num_start = pos;
        while (pos < tlen &&
               (isdigit((unsigned char)text[pos]) || text[pos] == '_' || text[pos] == '.' ||
                text[pos] == 'e' || text[pos] == 'E' || text[pos] == '+' || text[pos] == '-')) {
            /* Handle e/E only if preceded by digit context */
            if ((text[pos] == '+' || text[pos] == '-') && pos > 0 && text[pos - 1] != 'e' &&
                text[pos - 1] != 'E')
                break;
            if ((text[pos] == 'e' || text[pos] == 'E') && pos > 0 &&
                !isdigit((unsigned char)text[pos - 1]) && text[pos - 1] != '_')
                break;
            pos++;
        }
        int num_end = pos;

        /* Unit part: lowercase alpha */
        int unit_start = pos;
        while (pos < tlen && text[pos] >= 'a' && text[pos] <= 'z')
            pos++;
        int unit_end = pos;

        if (unit_end > unit_start && num_end > num_start) {
            hlbuf_push(out, line, base_col + num_start, base_col + num_end, HL_NUMBER);
            hlbuf_push(out, line, base_col + unit_start, base_col + unit_end, HL_UNIT);
            emitted = 1;
        } else {
            /* Safety: skip one char to avoid infinite loop */
            if (pos == num_start)
                pos++;
        }
    }

    return emitted;
}

/* ======================================================================== */
/* Token spans (multiline support)                                           */
/* ======================================================================== */

/* Emit highlight spans for a token, handling multiline tokens. */
static void emit_token_spans(const char *src, Tok *tok, HlGroup group, HlBuf *out) {
    const char *text = src + tok->start;
    int tlen = tok->len;
    int start_line = tok->line - 1;
    int start_col = tok->col - 1;

    /* Fast path: single-line token */
    int has_newline = 0;
    for (int i = 0; i < tlen; i++) {
        if (text[i] == '\n' || text[i] == '\r') {
            has_newline = 1;
            break;
        }
    }

    if (!has_newline) {
        hlbuf_push(out, start_line, start_col, start_col + tlen, group);
        return;
    }

    /* Multiline: emit one span per line */
    int line_idx = start_line;
    int line_start = 0;
    for (int i = 0; i <= tlen; i++) {
        int at_end = (i == tlen);
        int at_nl = !at_end && (text[i] == '\n' || text[i] == '\r');

        if (at_end || at_nl) {
            int line_len = i - line_start;
            /* Strip trailing \r */
            if (line_len > 0 && text[line_start + line_len - 1] == '\r')
                line_len--;

            if (line_len > 0) {
                int cs = (line_idx == start_line) ? start_col : 0;
                int ce = cs + line_len;
                hlbuf_push(out, line_idx, cs, ce, group);
            }

            if (at_nl) {
                if (text[i] == '\r' && i + 1 < tlen && text[i + 1] == '\n')
                    i++;
                line_idx++;
                line_start = i + 1;
            }
        }
    }
}

/* ======================================================================== */
/* DiagBuf                                                                   */
/* ======================================================================== */

void diagbuf_init(DiagBuf *b) {
    b->diags = 0;
    b->count = 0;
    b->capacity = 0;
}

void diagbuf_free(DiagBuf *b) {
    free(b->diags);
    b->diags = 0;
    b->count = 0;
    b->capacity = 0;
}

static void diagbuf_push(DiagBuf *b, int line, int cs, int ce, int sev, const char *msg) {
    if (!b)
        return;
    if (b->count >= b->capacity) {
        int cap = b->capacity ? b->capacity * 2 : 16;
        b->diags = realloc(b->diags, (size_t)cap * sizeof(HlDiag));
        b->capacity = cap;
    }
    HlDiag *d = &b->diags[b->count++];
    d->line = line;
    d->col_start = cs;
    d->col_end = ce;
    d->severity = sev;
    snprintf(d->message, sizeof(d->message), "%s", msg);
}

/* ======================================================================== */
/* Structural highlight pass                                                 */
/* ======================================================================== */

/* Compare spans by document position for sorted lookup. */
static int span_pos_cmp(const void *a, const void *b) {
    const HlSpan *sa = (const HlSpan *)a;
    const HlSpan *sb = (const HlSpan *)b;
    if (sa->line != sb->line)
        return (sa->line < sb->line) ? -1 : 1;
    if (sa->col_start != sb->col_start)
        return (sa->col_start < sb->col_start) ? -1 : 1;
    return 0;
}

/* Override the group of a span matching (line, col_start).
 * Uses cursor to avoid rescanning from the start. */
static void override_span(HlBuf *hl, int line, int col_start, HlGroup group, int *cursor) {
    for (int i = *cursor; i < hl->count; i++) {
        HlSpan *s = &hl->spans[i];
        if (s->line > line || (s->line == line && s->col_start > col_start))
            break;
        if (s->line == line && s->col_start == col_start) {
            s->group = group;
            *cursor = i;
            return;
        }
    }
}

/* Is this a block-header keyword that expects a colon? */
static int is_block_header(TT t) {
    switch (t) {
    case TT_IF:
    case TT_ELIF:
    case TT_ELSE:
    case TT_WHILE:
    case TT_FOR:
    case TT_MATCH:
    case TT_FN:
    case TT_UNLESS:
    case TT_CATCH:
    case TT_TRY:
    case TT_DECORATOR:
        return 1;
    default:
        return 0;
    }
}

void structural_highlight(const char *src, int src_len, TokBuf *tokens, HlBuf *hl, DiagBuf *diags) {
    int tc = tokens->count;
    Tok *toks = tokens->toks;

    /* Sort spans for cursor-based override lookup. */
    if (hl->count > 1)
        qsort(hl->spans, (size_t)hl->count, sizeof(HlSpan), span_pos_cmp);

    /* State */
    int at_stmt_start = 1;

    int bracket_depth = 0;
    int continuation_indent = -1; /* explicit dot-continuation block indent, -1 when inactive */
    int cursor = 0;               /* span search cursor */
    int cached_line = -1;
    int cached_line_start = 0;
    int cached_indent = -1;
    int cached_first_non_ws = 0;

    /* Bracket matching stack */
    typedef struct {
        TT type;
        int line;
        int col;
    } BracketEntry;

    BracketEntry bstack[256];
    int bcount = 0;

    /* Block header colon tracking */
    int need_colon = 0;
    int nc_line = -1;
    int nc_col = -1;

    TT prev_sig = TT_EOF;

    for (int i = 0; i < tc; i++) {
        Tok *tok = &toks[i];
        TT t = tok->type;
        int line = tok->line - 1;
        int col = tok->col - 1;
        int line_start;
        int at_line_start;
        int cur_indent;

        if (line != cached_line) {
            cached_line = line;
            line_prefix_info_for_pos(src, src_len, tok->start, &cached_line_start, &cached_indent,
                                     &cached_first_non_ws);
        }
        line_start = cached_line_start;
        at_line_start = tok->start <= cached_first_non_ws;
        cur_indent = at_line_start ? cached_indent : -1;

        /* ---- Layout tokens: update statement boundary ---- */
        if (t == TT_NEWLINE || t == TT_INDENT || t == TT_DEDENT) {
            if (bracket_depth == 0) {
                /* Missing colon check: block keyword seen, then NEWLINE=>INDENT
                 * without an intervening colon. */
                if (need_colon && t == TT_NEWLINE && diags) {
                    int j = i + 1;
                    /* Skip consecutive NEWLINEs */
                    while (j < tc && toks[j].type == TT_NEWLINE)
                        j++;
                    if (j < tc && toks[j].type == TT_INDENT)
                        diagbuf_push(diags, nc_line, nc_col, nc_col + 1, 2,
                                     "missing colon after block header");
                    need_colon = 0;
                }
                at_stmt_start = 1;
            }
            continue;
        }
        if (t == TT_COMMENT || t == TT_EOF)
            continue;

        /* ---- Semicolon: statement boundary ---- */
        if (t == TT_SEMI) {
            at_stmt_start = 1;

            continuation_indent = -1;
            need_colon = 0;
            prev_sig = t;
            continue;
        }

        /* ---- Colon at depth 0: clears block-header expectation ---- */
        if (t == TT_COLON && bracket_depth == 0) {
            need_colon = 0;
            at_stmt_start = 0;

            continuation_indent = -1;
            prev_sig = t;
            continue;
        }

        /* ---- Bracket tracking + diagnostics ---- */
        if (t == TT_LPAR || t == TT_LSQB || t == TT_LBRACE) {
            /* Bracket-led top-level statements must break explicit dot-continuation chains. */
            if (bracket_depth == 0 && at_stmt_start)
                continuation_indent = -1;

            bracket_depth++;
            if (bcount < 256) {
                bstack[bcount].type = t;
                bstack[bcount].line = line;
                bstack[bcount].col = col;
                bcount++;
            }
            at_stmt_start = 0;
            prev_sig = t;
            continue;
        }
        if (t == TT_RPAR || t == TT_RSQB || t == TT_RBRACE) {
            TT expected = (t == TT_RPAR) ? TT_LPAR : (t == TT_RSQB) ? TT_LSQB : TT_LBRACE;
            if (bcount > 0 && bstack[bcount - 1].type == expected) {
                bcount--;
                if (bracket_depth > 0)
                    bracket_depth--;
            } else {
                if (diags)
                    diagbuf_push(diags, line, col, col + 1, 1,
                                 bcount > 0 ? "mismatched closing bracket"
                                            : "unexpected closing bracket");
                if (bracket_depth > 0)
                    bracket_depth--;
            }
            at_stmt_start = 0;
            prev_sig = t;
            continue;
        }

        /* ---- Implicit-subject chain detection (depth 0 only) ---- */
        if (bracket_depth == 0) {
            /* continuation_indent tracks explicit line-start dot continuation blocks.
             * It activates when a line-start dot follows an expression-ending previous line
             * at greater indent, and deactivates when statement-start indent no longer matches. */
            int explicit_continuation_dot = 0;

            if (at_stmt_start && at_line_start && continuation_indent >= 0 &&
                cur_indent != continuation_indent)
                continuation_indent = -1;

            /* Starting dot at statement start => begin chain */
            if (at_stmt_start && t == TT_DOT) {
                if (at_line_start) {
                    if (continuation_indent >= 0 && cur_indent == continuation_indent) {
                        explicit_continuation_dot = 1;
                    } else if (is_expr_end(prev_sig)) {
                        int prev_indent = -1;
                        if (prev_nonblank_line_info(src, line_start, &prev_indent) &&
                            cur_indent > prev_indent) {
                            explicit_continuation_dot = 1;
                            continuation_indent = cur_indent;
                        }
                    }
                }

                if (explicit_continuation_dot) {
                    /* Dot-chain continuation keeps explicit receiver semantics. */
                    override_span(hl, line, col, HL_PUNCTUATION, &cursor);
                    at_stmt_start = 0;

                    prev_sig = t;
                    continue;
                }

                continuation_indent = -1;
                /* Only the leading dot is implicit subject; subsequent
                 * dots and idents keep their lexical classification
                 * (punctuation / property / function). */
                override_span(hl, line, col, HL_IMPLICIT_SUBJECT, &cursor);
                at_stmt_start = 0;
                prev_sig = t;
                continue;
            }
        }

        /* ---- Block header keyword detection ---- */
        if (bracket_depth == 0 && is_block_header(t)) {
            need_colon = 1;
            nc_line = line;
            nc_col = col;
        }

        if (at_stmt_start)
            continuation_indent = -1;
        at_stmt_start = 0;
        prev_sig = t;
    }

    /* Unclosed bracket diagnostics */
    if (diags) {
        for (int i = 0; i < bcount; i++)
            diagbuf_push(diags, bstack[i].line, bstack[i].col, bstack[i].col + 1, 1,
                         "unclosed bracket");
    }
}

/* ======================================================================== */
/* Line-prefix cache                                                         */
/* ======================================================================== */

/* Cached line-prefix state to avoid rescanning the same line for every token. */
typedef struct {
    int line;
    int line_start;
    int indent;
    int first_non_ws;
} LinePrefixCache;

static void lpc_update(LinePrefixCache *c, const char *src, int src_len, const Tok *tok) {
    int line = tok->line - 1;
    if (line != c->line) {
        c->line = line;
        line_prefix_info_for_pos(src, src_len, tok->start, &c->line_start, &c->indent,
                                 &c->first_non_ws);
    }
}

static int lpc_at_line_start(const LinePrefixCache *c, const Tok *tok) {
    return tok->start <= c->first_non_ws;
}

/* ======================================================================== */
/* Guard-pipe classification                                                 */
/* ======================================================================== */

/* Classify whether a depth-0 pipe token is a guard-branch keyword.
 * Guard pipes: `cond: body | cond: body |: else`. */
static int is_guard_pipe(const char *src, int src_len, const Tok *toks, int tc, Tok *tok,
                         int cur_sig_idx, int group_depth, int group_depth_after, TT next_sig_type,
                         LinePrefixCache *lpc) {
    /* |: is always a guard else */
    if (next_sig_type == TT_COLON)
        return 1;

    /* No following colon in this statement => not a guard pipe */
    if (!stmt_has_next_colon(toks, tc, cur_sig_idx, group_depth_after))
        return 0;

    lpc_update(lpc, src, src_len, tok);

    /* Multi-line guard branch: pipe at line start with following colon */
    if (lpc_at_line_start(lpc, tok))
        return 1;

    /* Inline guard continuation: prior colon exists in same statement */
    return stmt_has_prev_colon(toks, cur_sig_idx, group_depth);
}

/* ======================================================================== */
/* Main highlight function                                                   */
/* ======================================================================== */

void highlight(const char *src, int src_len, TokBuf *tokens, HlBuf *out) {
    /* Build significant-token index for prev/next lookups */
    int tc = tokens->count;
    Tok *toks = tokens->toks;

    /* We iterate only over significant (non-layout) tokens. */
    TT prev_sig = TT_EOF;
    int group_depth = 0;
    LinePrefixCache lpc = {-1, 0, -1, 0};

    /* Pre-scan: build array of sig token indices for next-lookahead. */
    int *sig_idx = malloc(tc * sizeof(int));
    int sig_count = 0;
    /* Emit comment spans first (comments are transparent to structural logic). */
    for (int i = 0; i < tc; i++) {
        if (toks[i].type == TT_COMMENT)
            emit_token_spans(src, &toks[i], HL_COMMENT, out);
    }

    for (int i = 0; i < tc; i++) {
        TT t = toks[i].type;
        if (t == TT_COMMENT || is_layout(t))
            continue;
        sig_idx[sig_count] = i;
        sig_count++;
    }

    for (int si = 0; si < sig_count; si++) {
        int cur_sig_idx = sig_idx[si];
        Tok *tok = &toks[cur_sig_idx];
        int next_sig_idx = (si + 1 < sig_count) ? sig_idx[si + 1] : -1;
        Tok *next_sig_tok = (next_sig_idx >= 0) ? &toks[next_sig_idx] : 0;
        TT prev_prev_sig = (si >= 2) ? toks[sig_idx[si - 2]].type : TT_EOF;
        TT next_sig_type = next_sig_tok ? next_sig_tok->type : TT_EOF;
        int group_depth_after_tok = group_depth;
        int same_stmt_next = 0;

        if (is_open_group(tok->type))
            group_depth_after_tok++;
        else if (is_close_group(tok->type) && group_depth_after_tok > 0)
            group_depth_after_tok--;
        if (next_sig_tok) {
            same_stmt_next =
                next_sig_same_statement(toks, cur_sig_idx, next_sig_idx, group_depth_after_tok);
        }

        HlGroup group = base_group(tok->type);
        if (group == HL_NONE) {
            prev_sig = tok->type;
            group_depth = group_depth_after_tok;
            continue;
        }

        int subject_override = 0;
        if (tok->type == TT_DOT) {
            lpc_update(&lpc, src, src_len, tok);
            int line_start = lpc.line_start;
            int at_line_start = lpc_at_line_start(&lpc, tok);
            int cur_indent = lpc.indent;

            if (!at_line_start) {
                if (!is_expr_end(prev_sig))
                    subject_override = 1;
            } else {
                if (!is_expr_end(prev_sig)) {
                    subject_override = 1;
                } else {
                    int prev_indent = prev_nonblank_indent(src, line_start);
                    if (prev_indent < 0 || cur_indent <= prev_indent)
                        subject_override = 1;
                }
            }
        }

        /* Structural classification */
        if (subject_override) {
            group = HL_IMPLICIT_SUBJECT;
        } else if (tok->type == TT_QMARK && next_sig_tok && next_sig_tok->type == TT_IDENT &&
                   token_text_equals(src, src_len, next_sig_tok, "ret", 3)) {
            group = HL_KEYWORD;
        } else if (tok->type == TT_IDENT) {
            group = classify_ident(src, src_len, tok, next_sig_tok, prev_sig, prev_prev_sig,
                                   same_stmt_next);
        } else if (tok->type == TT_STRING || tok->type == TT_RAW_STRING ||
                   tok->type == TT_RAW_HASH_STRING || tok->type == TT_SHELL_STRING ||
                   tok->type == TT_SHELL_BANG_STRING || tok->type == TT_ENV_STRING) {
            group = classify_string(prev_sig);
        } else if (tok->type == TT_PIPE && group_depth == 0) {
            if (is_guard_pipe(src, src_len, toks, tc, tok, cur_sig_idx, group_depth,
                              group_depth_after_tok, next_sig_type, &lpc))
                group = HL_KEYWORD;
        } else if (tok->type == TT_COLON && prev_sig == TT_PIPE && group_depth == 0) {
            group = HL_KEYWORD; /* guard else: |: */
        }

        /* Duration/size sub-spans */
        if (tok->type == TT_DURATION || tok->type == TT_SIZE) {
            if (emit_dur_size_spans(src, tok, out)) {
                prev_sig = tok->type;
                continue;
            }
        }

        emit_token_spans(src, tok, group, out);
        /* Standalone implicit-subject dot (next sig token on different line):
         * preserve prev_sig to prevent the next identifier from being
         * incorrectly classified as a property access. */
        int standalone_subject =
            subject_override && (!next_sig_tok || next_sig_tok->line != tok->line);
        if (tok->type != TT_COMMENT && !standalone_subject)
            prev_sig = tok->type;
        group_depth = group_depth_after_tok;
    }

    free(sig_idx);

    /* Structural pass: fanpath chain overrides, diagnostics. */
    structural_highlight(src, src_len, tokens, out, 0);
}
