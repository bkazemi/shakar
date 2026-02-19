#ifndef SHK_TOKEN_TYPES_H
#define SHK_TOKEN_TYPES_H

/* Token type enum — mirrors token_types.py TT enum.
 * Order matters: values are used as indices in highlight group lookup. */
typedef enum {
    /* Literals */
    TT_NUMBER = 0,
    TT_DURATION,
    TT_SIZE,
    TT_STRING,
    TT_RAW_STRING,
    TT_RAW_HASH_STRING,
    TT_SHELL_STRING,
    TT_SHELL_BANG_STRING,
    TT_PATH_STRING,
    TT_ENV_STRING,
    TT_REGEX,
    TT_IDENT,

    /* Keywords */
    TT_IF,
    TT_ELIF,
    TT_ELSE,
    TT_UNLESS,
    TT_WHILE,
    TT_FOR,
    TT_MATCH,
    TT_IN,
    TT_BREAK,
    TT_CONTINUE,
    TT_RETURN,
    TT_FN,
    TT_LET,
    TT_ONCE,
    TT_WAIT,
    TT_SPAWN,
    TT_USING,
    TT_CALL,
    TT_DEFER,
    TT_AFTER,
    TT_THROW,
    TT_CATCH,
    TT_TRY,
    TT_ASSERT,
    TT_DBG,
    TT_DECORATOR,
    TT_GET,
    TT_SET,
    TT_HOOK,
    TT_OVER,
    TT_BIND,
    TT_IMPORT,
    TT_FAN,

    /* Literal keywords */
    TT_TRUE,
    TT_FALSE,
    TT_NIL,

    /* Operators */
    TT_PLUS,
    TT_MINUS,
    TT_STAR,
    TT_SLASH,
    TT_FLOORDIV,
    TT_MOD,
    TT_POW,
    TT_DEEPMERGE,
    TT_CARET,

    /* Comparison */
    TT_EQ,
    TT_NEQ,
    TT_LT,
    TT_LTE,
    TT_GT,
    TT_GTE,
    TT_SEND,
    TT_RECV,
    TT_IS,
    TT_NOT,

    /* Logical */
    TT_AND,
    TT_OR,
    TT_NEG,

    /* Structural match */
    TT_TILDE,
    TT_REGEXMATCH,

    /* Assignment */
    TT_ASSIGN,
    TT_WALRUS,
    TT_APPLYASSIGN,
    TT_PLUSEQ,
    TT_MINUSEQ,
    TT_STAREQ,
    TT_SLASHEQ,
    TT_FLOORDIVEQ,
    TT_MODEQ,
    TT_POWEQ,

    /* Punctuation */
    TT_LPAR,
    TT_RPAR,
    TT_LSQB,
    TT_RSQB,
    TT_LBRACE,
    TT_RBRACE,
    TT_DOT,
    TT_COMMA,
    TT_COLON,
    TT_SEMI,
    TT_QMARK,
    TT_NULLISH,
    TT_AT,
    TT_DOLLAR,
    TT_BACKQUOTE,

    /* Increment/Decrement */
    TT_INCR,
    TT_DECR,

    /* Lambda */
    TT_AMP,

    /* Pipe */
    TT_PIPE,

    /* Spread */
    TT_SPREAD,

    /* Special */
    TT_NEWLINE,
    TT_INDENT,
    TT_DEDENT,
    TT_EOF,
    TT_COMMENT,

    TT__COUNT /* sentinel — total number of token types */
} TT;

/* Token: references source buffer via offset + length (zero-copy). */
typedef struct {
    TT type;
    int start; /* byte offset into source */
    int len;   /* byte length of token value */
    int line;  /* 1-based line */
    int col;   /* 1-based column */
} Tok;

/* Growable token buffer. */
typedef struct {
    Tok *toks;
    int count;
    int capacity;
} TokBuf;

static inline void tokbuf_init(TokBuf *b) {
    b->toks = 0;
    b->count = 0;
    b->capacity = 0;
}

#endif /* SHK_TOKEN_TYPES_H */
