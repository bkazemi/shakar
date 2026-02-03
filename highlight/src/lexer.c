/*
 * Shakar lexer — C port of lexer_rd.py
 *
 * Zero-copy: token values reference the source buffer via (start, len).
 * Single-pass tokenization with indentation tracking.
 */

#include "lexer.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================== */
/* Helpers                                                                   */
/* ======================================================================== */

static inline char pk(Lexer *L, int off) {
    int i = L->pos + off;
    return (i < L->src_len) ? L->src[i] : '\0';
}

static inline char pk0(Lexer *L) {
    return pk(L, 0);
}

static inline void adv(Lexer *L, int n) {
    for (int i = 0; i < n; i++) {
        L->pos++;
        L->col++;
    }
}

/* Advance inside multiline literal — tracks newlines. */
static void adv_lit(Lexer *L) {
    char ch = pk0(L);
    if (ch == '\r') {
        if (pk(L, 1) == '\n') {
            L->pos += 2;
        } else {
            L->pos += 1;
        }
        L->line++;
        L->col = 1;
    } else if (ch == '\n') {
        L->pos++;
        L->line++;
        L->col = 1;
    } else {
        L->pos++;
        L->col++;
    }
}

static void lexer_error(Lexer *L, const char *msg) {
    if (!L->has_error) {
        L->has_error = 1;
        L->error_line = L->line;
        L->error_col = L->col;
        L->error_pos = L->pos;
        snprintf(L->error_msg, sizeof(L->error_msg), "%s", msg);
    }
}

static void lexer_error_at(Lexer *L, int line, int col, int pos, const char *msg) {
    if (!L->has_error) {
        L->has_error = 1;
        L->error_line = line;
        L->error_col = col;
        L->error_pos = pos;
        snprintf(L->error_msg, sizeof(L->error_msg), "%s", msg);
    }
}

/* Ensure capacity in token buffer. */
static void tokbuf_grow(TokBuf *b) {
    if (b->count >= b->capacity) {
        int cap = b->capacity ? b->capacity * 2 : 256;
        b->toks = realloc(b->toks, cap * sizeof(Tok));
        b->capacity = cap;
    }
}

static void emit(Lexer *L, TT type, int start, int len, int line, int col) {
    TokBuf *b = &L->tokens;
    tokbuf_grow(b);
    Tok *t = &b->toks[b->count++];
    t->type = type;
    t->start = start;
    t->len = len;
    t->line = line;
    t->col = col;

    if (type == TT_NEWLINE) {
        L->indent_after_colon = L->line_ended_with_colon;
        L->line_ended_with_colon = 0;
    } else if (type != TT_INDENT && type != TT_DEDENT && type != TT_EOF) {
        L->line_ended_with_colon = (type == TT_COLON);
    }
}

/* Check if source matches string at pos. */
static int starts_with(Lexer *L, const char *s, int off) {
    int p = L->pos + off;
    while (*s) {
        if (p >= L->src_len || L->src[p] != *s)
            return 0;
        s++;
        p++;
    }
    return 1;
}

static int starts_with0(Lexer *L, const char *s) {
    return starts_with(L, s, 0);
}

/* ======================================================================== */
/* Keyword lookup                                                            */
/* ======================================================================== */

typedef struct {
    const char *word;
    TT          type;
} KWEntry;

/* Sorted by word for binary search. */
static const KWEntry KEYWORDS[] = {
    {"after", TT_AFTER},   {"all", TT_ALL},
    {"and", TT_AND},       {"any", TT_ANY},
    {"assert", TT_ASSERT}, {"bind", TT_BIND},
    {"break", TT_BREAK},   {"call", TT_CALL},
    {"catch", TT_CATCH},   {"continue", TT_CONTINUE},
    {"dbg", TT_DBG},       {"decorator", TT_DECORATOR},
    {"defer", TT_DEFER},   {"elif", TT_ELIF},
    {"else", TT_ELSE},     {"false", TT_FALSE},
    {"fan", TT_FAN},       {"fn", TT_FN},
    {"for", TT_FOR},       {"get", TT_GET},
    {"group", TT_GROUP},   {"hook", TT_HOOK},
    {"if", TT_IF},         {"import", TT_IMPORT},
    {"in", TT_IN},         {"is", TT_IS},
    {"let", TT_LET},       {"match", TT_MATCH},
    {"nil", TT_NIL},       {"not", TT_NOT},
    {"or", TT_OR},         {"over", TT_OVER},
    {"return", TT_RETURN}, {"set", TT_SET},
    {"spawn", TT_SPAWN},   {"throw", TT_THROW},
    {"true", TT_TRUE},     {"unless", TT_UNLESS},
    {"using", TT_USING},   {"wait", TT_WAIT},
    {"while", TT_WHILE},
};

#define KW_COUNT (sizeof(KEYWORDS) / sizeof(KEYWORDS[0]))

static TT keyword_lookup(const char *word, int len) {
    int lo = 0, hi = (int)KW_COUNT - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strncmp(word, KEYWORDS[mid].word, len);
        if (cmp == 0) {
            /* Check exact length match. */
            if (KEYWORDS[mid].word[len] == '\0')
                return KEYWORDS[mid].type;
            cmp = -1; /* word is shorter */
        }
        if (cmp < 0)
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    return TT_IDENT;
}

/* ======================================================================== */
/* Operator table                                                            */
/* ======================================================================== */

typedef struct {
    const char *op;
    int         len;
    TT          type;
} OpEntry;

static const OpEntry OPERATORS[] = {
    /* Three-character */
    {"//=", 3, TT_FLOORDIVEQ},
    {"**=", 3, TT_POWEQ},
    {"...", 3, TT_SPREAD},
    /* Two-character */
    {"~~", 2, TT_REGEXMATCH},
    {"==", 2, TT_EQ},
    {"!=", 2, TT_NEQ},
    {"->", 2, TT_SEND},
    {"<-", 2, TT_RECV},
    {"<=", 2, TT_LTE},
    {">=", 2, TT_GTE},
    {"&&", 2, TT_AND},
    {"||", 2, TT_OR},
    {":=", 2, TT_WALRUS},
    {".=", 2, TT_APPLYASSIGN},
    {"+=", 2, TT_PLUSEQ},
    {"-=", 2, TT_MINUSEQ},
    {"*=", 2, TT_STAREQ},
    {"/=", 2, TT_SLASHEQ},
    {"//", 2, TT_FLOORDIV},
    {"**", 2, TT_POW},
    {"%=", 2, TT_MODEQ},
    {"++", 2, TT_INCR},
    {"--", 2, TT_DECR},
    {"??", 2, TT_NULLISH},
    {"+>", 2, TT_DEEPMERGE},
    /* Single-character */
    {"+", 1, TT_PLUS},
    {"-", 1, TT_MINUS},
    {"*", 1, TT_STAR},
    {"/", 1, TT_SLASH},
    {"%", 1, TT_MOD},
    {"^", 1, TT_CARET},
    {"<", 1, TT_LT},
    {">", 1, TT_GT},
    {"!", 1, TT_NEG},
    {"=", 1, TT_ASSIGN},
    {"(", 1, TT_LPAR},
    {")", 1, TT_RPAR},
    {"[", 1, TT_LSQB},
    {"]", 1, TT_RSQB},
    {"{", 1, TT_LBRACE},
    {"}", 1, TT_RBRACE},
    {".", 1, TT_DOT},
    {",", 1, TT_COMMA},
    {":", 1, TT_COLON},
    {";", 1, TT_SEMI},
    {"?", 1, TT_QMARK},
    {"@", 1, TT_AT},
    {"$", 1, TT_DOLLAR},
    {"`", 1, TT_BACKQUOTE},
    {"&", 1, TT_AMP},
    {"|", 1, TT_PIPE},
    {"~", 1, TT_TILDE},
};

#define OP_COUNT (sizeof(OPERATORS) / sizeof(OPERATORS[0]))

/* ======================================================================== */
/* Quoted content scanner                                                    */
/* ======================================================================== */

/* Scan content between quotes. Returns 0 on success, -1 on error. */
static int scan_quoted(Lexer *L, char quote, int allow_esc, int allow_nl) {
    while (L->pos < L->src_len && pk0(L) != quote) {
        char ch = pk0(L);

        if (ch == '\n' || ch == '\r') {
            if (!allow_nl) {
                lexer_error(L, "Newline in string literal");
                return -1;
            }
            adv_lit(L);
            continue;
        }

        if (allow_esc && ch == '\\') {
            if (allow_nl)
                adv_lit(L);
            else
                adv(L, 1);
            if (L->pos < L->src_len) {
                if ((pk0(L) == '\n' || pk0(L) == '\r') && !allow_nl) {
                    lexer_error(L, "Newline in string literal");
                    return -1;
                }
                if (allow_nl)
                    adv_lit(L);
                else
                    adv(L, 1);
            }
            continue;
        }

        if (allow_nl)
            adv_lit(L);
        else
            adv(L, 1);
    }
    return 0;
}

/* ======================================================================== */
/* Number scanning                                                           */
/* ======================================================================== */

/* Scan decimal digits with underscore separators.
 * Returns 1 if at least one digit seen. */
static int scan_digits_us(Lexer *L) {
    int saw = 0, prev_us = 0;
    while (1) {
        char ch = pk0(L);
        if (ch == '_') {
            if (!saw || prev_us) {
                lexer_error(L, "Invalid underscore in number literal");
                return -1;
            }
            prev_us = 1;
            adv(L, 1);
            continue;
        }
        if (ch >= '0' && ch <= '9') {
            saw = 1;
            prev_us = 0;
            adv(L, 1);
            continue;
        }
        break;
    }
    if (prev_us) {
        lexer_error(L, "Trailing underscore in number literal");
        return -1;
    }
    return saw;
}

static int is_valid_prefixed_digit(char ch, int base) {
    if (ch >= '0' && ch <= '9')
        return (ch - '0') < base;
    if (base == 16) {
        if ((ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'))
            return 1;
    }
    return 0;
}

/* Duration units — longest first for matching. */
static const char *DUR_UNITS[] = {"nsec", "usec", "msec", "sec", "min", "hr", "day", "wk"};
#define DUR_UNIT_COUNT 8

/* Size units — longest first for matching. */
static const char *SIZE_UNITS[] = {"tib", "gib", "mib", "kib", "tb", "gb", "mb", "kb", "b"};
#define SIZE_UNIT_COUNT 9

/* Try to match a unit from the unit list at current position.
 * Returns unit length on match (and advances), 0 on no match. */
static int match_unit(Lexer *L, const char **units, int count) {
    for (int i = 0; i < count; i++) {
        int ulen = strlen(units[i]);
        if (starts_with0(L, units[i])) {
            adv(L, ulen);
            return ulen;
        }
    }
    return 0;
}

/* Scan base-prefixed integer (0b, 0o, 0x).
 * Returns 1 if matched and emitted, 0 if not a prefix, -1 on error. */
static int scan_prefixed_int(Lexer *L, int sline, int scol, int spos) {
    if (pk0(L) != '0')
        return 0;
    char pfx = pk(L, 1);
    if (pfx == 'B' || pfx == 'O' || pfx == 'X') {
        /* Point at the invalid prefix character, not the leading zero. */
        lexer_error_at(L, sline, scol + 1, spos + 1, "Uppercase base prefixes are not allowed");
        return -1;
    }
    int base;
    if (pfx == 'b')
        base = 2;
    else if (pfx == 'o')
        base = 8;
    else if (pfx == 'x')
        base = 16;
    else
        return 0;

    adv(L, 2);

    int saw = 0, prev_us = 0;
    while (1) {
        char ch = pk0(L);
        if (ch == '_') {
            if (!saw || prev_us) {
                lexer_error(L, "Invalid underscore in base-prefixed integer");
                return -1;
            }
            prev_us = 1;
            adv(L, 1);
            continue;
        }
        if (is_valid_prefixed_digit(ch, base)) {
            saw = 1;
            prev_us = 0;
            adv(L, 1);
            continue;
        }
        break;
    }

    if (!saw) {
        lexer_error(L, "Incomplete base-prefixed integer");
        return -1;
    }
    if (prev_us) {
        lexer_error(L, "Trailing underscore in base-prefixed integer");
        return -1;
    }

    char nx = pk0(L);
    if (nx == '.' && pk(L, 1) >= '0' && pk(L, 1) <= '9') {
        lexer_error(L, "Base-prefixed integers cannot be floats");
        return -1;
    }
    if (isalnum((unsigned char)nx) || nx == '_') {
        lexer_error(L, "Base-prefixed integers cannot have unit suffixes");
        return -1;
    }

    emit(L, TT_NUMBER, spos, L->pos - spos, sline, scol);
    return 1;
}

static void scan_number(Lexer *L) {
    int sline = L->line, scol = L->col, spos = L->pos;

    /* Try base-prefixed */
    int r = scan_prefixed_int(L, sline, scol, spos);
    if (r != 0)
        return; /* emitted or error */

    /* Decimal digits */
    scan_digits_us(L);
    if (L->has_error)
        return;

    /* Fractional part */
    if (pk0(L) == '.') {
        char nx = pk(L, 1);
        if (nx == 'e' || nx == 'E') {
            lexer_error(L, "Fractional digits required after decimal point");
            return;
        }
        if ((nx >= '0' && nx <= '9') || nx == '_') {
            adv(L, 1);
            if (pk0(L) == '_') {
                lexer_error(L, "Leading underscore in fractional part");
                return;
            }
            scan_digits_us(L);
            if (L->has_error)
                return;
        }
    }

    /* Exponent */
    if (pk0(L) == 'e' || pk0(L) == 'E') {
        adv(L, 1);
        if (pk0(L) == '+' || pk0(L) == '-')
            adv(L, 1);
        if (pk0(L) == '_') {
            lexer_error(L, "Leading underscore in exponent");
            return;
        }
        if (scan_digits_us(L) <= 0 && !L->has_error) {
            lexer_error(L, "Missing digits in exponent");
            return;
        }
        if (L->has_error)
            return;
    }

    /* Trailing dot check */
    if (pk0(L) == '.' && !isalpha((unsigned char)pk(L, 1)) && pk(L, 1) != '_') {
        lexer_error(L, "Trailing dot after number literal");
        return;
    }

    /* Check for duration/size unit */
    int unit_len = match_unit(L, DUR_UNITS, DUR_UNIT_COUNT);
    TT  lit_type = TT_DURATION;
    if (!unit_len) {
        unit_len = match_unit(L, SIZE_UNITS, SIZE_UNIT_COUNT);
        lit_type = TT_SIZE;
    }

    if (!unit_len) {
        emit(L, TT_NUMBER, spos, L->pos - spos, sline, scol);
        return;
    }

    /* Reject trailing alpha after unit */
    if (isalpha((unsigned char)pk0(L)) || pk0(L) == '_') {
        lexer_error(L, "Invalid duration/size literal");
        return;
    }

    /* Compound literal: more digit+unit pairs */
    const char **unit_list = (lit_type == TT_DURATION) ? DUR_UNITS : SIZE_UNITS;
    int          unit_count = (lit_type == TT_DURATION) ? DUR_UNIT_COUNT : SIZE_UNIT_COUNT;

    while (pk0(L) >= '0' && pk0(L) <= '9') {
        scan_digits_us(L);
        if (L->has_error)
            return;
        if (!match_unit(L, unit_list, unit_count)) {
            lexer_error(L, "Expected unit in compound literal");
            return;
        }
    }

    /* For highlighting we don't need to compute the numeric value —
     * just emit the full literal text as the token value. */
    emit(L, lit_type, spos, L->pos - spos, sline, scol);
}

/* ======================================================================== */
/* String scanners                                                           */
/* ======================================================================== */

static void scan_string(Lexer *L) {
    int  sline = L->line, scol = L->col, spos = L->pos;
    char quote = pk0(L);
    adv(L, 1);

    if (scan_quoted(L, quote, 1, 1) < 0)
        return;
    if (L->pos >= L->src_len) {
        /* Anchor unterminated string errors at the opening quote. */
        lexer_error_at(L, sline, scol, spos, "Unterminated string");
        return;
    }
    adv(L, 1); /* closing quote */
    emit(L, TT_STRING, spos, L->pos - spos, sline, scol);
}

static void scan_raw_string(Lexer *L) {
    /* 'raw' already consumed */
    int sline = L->line, scol = L->col - 3, spos = L->pos - 3;

    if (pk0(L) == '#') {
        /* raw#"..."# */
        adv(L, 1);
        char quote = pk0(L);
        adv(L, 1);
        while (L->pos < L->src_len) {
            if (pk0(L) == quote && pk(L, 1) == '#') {
                adv(L, 2);
                emit(L, TT_RAW_HASH_STRING, spos, L->pos - spos, sline, scol);
                return;
            }
            adv_lit(L);
        }
        lexer_error_at(L, sline, scol, spos, "Unterminated hash raw string");
        return;
    }

    char quote = pk0(L);
    adv(L, 1);
    if (scan_quoted(L, quote, 0, 1) < 0)
        return;
    if (L->pos >= L->src_len) {
        lexer_error_at(L, sline, scol, spos, "Unterminated raw string");
        return;
    }
    adv(L, 1);
    emit(L, TT_RAW_STRING, spos, L->pos - spos, sline, scol);
}

static void scan_shell_string(Lexer *L, int prefix_len, int allow_esc) {
    int  sline = L->line, scol = L->col - prefix_len, spos = L->pos - prefix_len;
    char quote = pk0(L);
    adv(L, 1);
    if (scan_quoted(L, quote, allow_esc, 1) < 0)
        return;
    if (L->pos >= L->src_len) {
        lexer_error_at(L, sline, scol, spos, "Unterminated shell string");
        return;
    }
    adv(L, 1);
    emit(L, TT_SHELL_STRING, spos, L->pos - spos, sline, scol);
}

static void scan_shell_bang_string(Lexer *L, int prefix_len, int allow_esc) {
    int  sline = L->line, scol = L->col - prefix_len, spos = L->pos - prefix_len;
    char quote = pk0(L);
    adv(L, 1);
    if (scan_quoted(L, quote, allow_esc, 1) < 0)
        return;
    if (L->pos >= L->src_len) {
        lexer_error_at(L, sline, scol, spos, "Unterminated shell string");
        return;
    }
    adv(L, 1);
    emit(L, TT_SHELL_BANG_STRING, spos, L->pos - spos, sline, scol);
}

static void scan_path_string(Lexer *L) {
    /* 'p' already consumed */
    int  sline = L->line, scol = L->col - 1, spos = L->pos - 1;
    char quote = pk0(L);
    adv(L, 1);
    if (scan_quoted(L, quote, 1, 0) < 0)
        return;
    if (L->pos >= L->src_len) {
        lexer_error_at(L, sline, scol, spos, "Unterminated path string");
        return;
    }
    adv(L, 1);
    emit(L, TT_PATH_STRING, spos, L->pos - spos, sline, scol);
}

static void scan_env_string(Lexer *L) {
    /* 'env' already consumed */
    int  sline = L->line, scol = L->col - 3, spos = L->pos - 3;
    char quote = pk0(L);
    adv(L, 1);
    if (scan_quoted(L, quote, 1, 0) < 0)
        return;
    if (L->pos >= L->src_len) {
        lexer_error_at(L, sline, scol, spos, "Unterminated env string");
        return;
    }
    adv(L, 1);
    emit(L, TT_ENV_STRING, spos, L->pos - spos, sline, scol);
}

static void scan_regex_literal(Lexer *L) {
    int sline = L->line, scol = L->col, spos = L->pos;
    adv(L, 1); /* consume 'r' */
    char quote = pk0(L);
    adv(L, 1);

    if (scan_quoted(L, quote, 1, 1) < 0)
        return;
    if (L->pos >= L->src_len) {
        lexer_error_at(L, sline, scol, spos, "Unterminated regex literal");
        return;
    }
    adv(L, 1); /* closing quote */

    /* Optional /flags */
    if (pk0(L) == '/') {
        adv(L, 1);
        char ch = pk0(L);
        if (ch != 'i' && ch != 'm' && ch != 's' && ch != 'x' && ch != 'f') {
            lexer_error(L, "Unknown regex flag");
            return;
        }
        while (pk0(L) == 'i' || pk0(L) == 'm' || pk0(L) == 's' || pk0(L) == 'x' || pk0(L) == 'f') {
            adv(L, 1);
        }
        if (isalnum((unsigned char)pk0(L)) || pk0(L) == '_') {
            lexer_error(L, "Unknown regex flag");
            return;
        }
    }

    emit(L, TT_REGEX, spos, L->pos - spos, sline, scol);
}

/* ======================================================================== */
/* Identifier / operator                                                     */
/* ======================================================================== */

static void scan_identifier(Lexer *L) {
    int sline = L->line, scol = L->col, spos = L->pos;
    while (isalnum((unsigned char)pk0(L)) || pk0(L) == '_')
        adv(L, 1);

    int len = L->pos - spos;
    TT  type = keyword_lookup(L->src + spos, len);
    emit(L, type, spos, len, sline, scol);
}

static void scan_operator(Lexer *L) {
    int sline = L->line, scol = L->col;

    for (int i = 0; i < (int)OP_COUNT; i++) {
        if (starts_with0(L, OPERATORS[i].op)) {
            int olen = OPERATORS[i].len;
            TT  type = OPERATORS[i].type;
            int spos = L->pos;
            adv(L, olen);
            emit(L, type, spos, olen, sline, scol);

            if (type == TT_LPAR || type == TT_LSQB || type == TT_LBRACE)
                L->group_depth++;
            else if (type == TT_RPAR || type == TT_RSQB || type == TT_RBRACE) {
                if (L->group_depth > 0)
                    L->group_depth--;
            }
            return;
        }
    }

    char msg[64];
    snprintf(msg, sizeof(msg), "Unexpected character '%c' at line %d, col %d", pk0(L), L->line,
             L->col);
    lexer_error(L, msg);
}

/* ======================================================================== */
/* Indentation                                                               */
/* ======================================================================== */

static void handle_indentation(Lexer *L) {
    int indent = 0;
    while (pk0(L) == ' ' || pk0(L) == '\t') {
        if (pk0(L) == ' ')
            indent += 1;
        else
            indent += 8;
        adv(L, 1);
    }

    L->at_line_start = 0;

    /* Implicit indent after colon inside group at base level */
    if (L->indent_after_colon && L->group_depth > 0 && L->indent_count == 1 &&
        L->prev_line_indent > 0 && L->prev_line_indent < indent) {
        if (L->indent_count < SHK_MAX_INDENT_DEPTH - 1) {
            L->indent_stack[L->indent_count++] = L->prev_line_indent;
        }
    }

    /* Skip blank lines */
    char ch = pk0(L);
    if (ch == '\n' || ch == '\r' || ch == '#')
        return;

    L->prev_line_indent = indent;

    if (L->group_depth > 0 && !L->indent_after_colon && L->indent_count == 1)
        return;

    L->indent_after_colon = 0;

    int current = L->indent_stack[L->indent_count - 1];

    if (indent > current) {
        if (L->indent_count < SHK_MAX_INDENT_DEPTH - 1) {
            L->indent_stack[L->indent_count++] = indent;
        }
        emit(L, TT_INDENT, L->pos, 0, L->line, L->col);
    } else if (indent < current) {
        while (L->indent_count > 1 && L->indent_stack[L->indent_count - 1] > indent) {
            L->indent_count--;
            emit(L, TT_DEDENT, L->pos, 0, L->line, L->col);
        }
        if (L->indent_stack[L->indent_count - 1] != indent) {
            if (!(L->group_depth > 0 && L->indent_count == 1)) {
                lexer_error(L, "Indentation mismatch");
            }
        }
    }
}

/* ======================================================================== */
/* Main scan_token                                                           */
/* ======================================================================== */

/* Check prefix match for string-prefix keywords (sh, env, p, r, raw, etc.).
 * Returns 1 if prefix matches and next char is quote. */
static int match_str_prefix(Lexer *L, const char *pfx) {
    int n = strlen(pfx);
    for (int i = 0; i < n; i++) {
        if (pk(L, i) != pfx[i])
            return 0;
    }
    char nx = pk(L, n);
    return (nx == '"' || nx == '\'');
}

static void scan_token(Lexer *L) {
    if (L->at_line_start && L->track_indent) {
        handle_indentation(L);
        return;
    }

    /* Skip whitespace */
    if (pk0(L) == ' ' || pk0(L) == '\t') {
        while (pk0(L) == ' ' || pk0(L) == '\t')
            adv(L, 1);
        return;
    }

    /* Comments */
    if (pk0(L) == '#') {
        int sline = L->line, scol = L->col, spos = L->pos;
        while (pk0(L) != '\n' && pk0(L) != '\r' && pk0(L) != '\0')
            adv(L, 1);
        if (L->emit_comments)
            emit(L, TT_COMMENT, spos, L->pos - spos, sline, scol);
        return;
    }

    /* Newlines */
    if (pk0(L) == '\n' || pk0(L) == '\r') {
        int sline = L->line, scol = L->col, spos = L->pos;
        if (pk0(L) == '\r' && pk(L, 1) == '\n')
            adv(L, 2);
        else
            adv(L, 1);
        emit(L, TT_NEWLINE, spos, L->pos - spos, sline, scol);
        L->line++;
        L->col = 1;
        L->at_line_start = 1;
        /* Fix: adv already moved col, but newline resets to 1 */
        /* We advanced past the newline chars, line was tracked by emit,
         * but we need to manually fix line/col since adv doesn't handle newlines. */
        /* Actually, adv increments col. We need to override. */
        L->line = sline + 1;
        L->col = 1;
        return;
    }

    /* String literals */
    if (pk0(L) == '"' || pk0(L) == '\'') {
        scan_string(L);
        return;
    }

    /* raw strings: raw"..." or raw#"..."# */
    if (pk0(L) == 'r' && pk(L, 1) == 'a' && pk(L, 2) == 'w') {
        char c3 = pk(L, 3);
        if (c3 == '"' || c3 == '\'' || c3 == '#') {
            adv(L, 3);
            scan_raw_string(L);
            return;
        }
    }

    /* sh_raw! strings */
    if (starts_with0(L, "sh_raw!") && (pk(L, 7) == '"' || pk(L, 7) == '\'')) {
        adv(L, 7);
        scan_shell_bang_string(L, 7, 0);
        return;
    }

    /* sh_raw strings */
    if (starts_with0(L, "sh_raw") && (pk(L, 6) == '"' || pk(L, 6) == '\'')) {
        adv(L, 6);
        scan_shell_string(L, 6, 0);
        return;
    }

    /* sh! strings */
    if (pk0(L) == 's' && pk(L, 1) == 'h' && pk(L, 2) == '!' &&
        (pk(L, 3) == '"' || pk(L, 3) == '\'')) {
        adv(L, 3);
        scan_shell_bang_string(L, 3, 1);
        return;
    }

    /* sh strings */
    if (match_str_prefix(L, "sh")) {
        adv(L, 2);
        scan_shell_string(L, 2, 1);
        return;
    }

    /* env strings */
    if (match_str_prefix(L, "env")) {
        adv(L, 3);
        scan_env_string(L);
        return;
    }

    /* path strings: p"..." */
    if (pk0(L) == 'p' && (pk(L, 1) == '"' || pk(L, 1) == '\'')) {
        adv(L, 1);
        scan_path_string(L);
        return;
    }

    /* regex: r"..." */
    if (pk0(L) == 'r' && (pk(L, 1) == '"' || pk(L, 1) == '\'')) {
        scan_regex_literal(L);
        return;
    }

    /* Numbers */
    if (pk0(L) >= '0' && pk0(L) <= '9') {
        scan_number(L);
        return;
    }

    /* Identifiers / keywords */
    if (isalpha((unsigned char)pk0(L)) || pk0(L) == '_') {
        scan_identifier(L);
        return;
    }

    /* Operators */
    scan_operator(L);
}

/* ======================================================================== */
/* Public API                                                                */
/* ======================================================================== */

void lexer_init(Lexer *L, const char *src, int len, int track_indent, int emit_comments) {
    memset(L, 0, sizeof(*L));
    L->src = src;
    L->src_len = len;
    L->pos = 0;
    L->line = 1;
    L->col = 1;
    L->track_indent = track_indent;
    L->emit_comments = emit_comments;
    L->indent_stack[0] = 0;
    L->indent_count = 1;
    L->at_line_start = 1;
    L->group_depth = 0;
    L->line_ended_with_colon = 0;
    L->indent_after_colon = 0;
    L->prev_line_indent = -1;
    L->has_error = 0;
    L->error_line = 0;
    L->error_col = 0;
    L->error_pos = 0;
    tokbuf_init(&L->tokens);
}

int lexer_tokenize(Lexer *L) {
    while (L->pos < L->src_len && !L->has_error) {
        scan_token(L);
    }

    if (L->has_error)
        return -1;

    /* Remaining DEDENTs */
    if (L->track_indent) {
        while (L->indent_count > 1) {
            L->indent_count--;
            emit(L, TT_DEDENT, L->pos, 0, L->line, L->col);
        }
    }

    emit(L, TT_EOF, L->pos, 0, L->line, L->col);
    return L->tokens.count;
}

void lexer_free(Lexer *L) {
    free(L->tokens.toks);
    L->tokens.toks = 0;
    L->tokens.count = 0;
    L->tokens.capacity = 0;
}
