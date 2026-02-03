#include "protocol.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int strbuf_ensure(StrBuf *b, int extra) {
    if (b->len + extra + 1 <= b->cap)
        return 1;
    int cap = b->cap ? b->cap * 2 : 256;
    while (cap < b->len + extra + 1)
        cap *= 2;
    char *next = realloc(b->data, (size_t)cap);
    if (!next)
        return 0;
    b->data = next;
    b->cap = cap;
    return 1;
}

void strbuf_init(StrBuf *b) {
    b->data = 0;
    b->len = 0;
    b->cap = 0;
}

void strbuf_free(StrBuf *b) {
    free(b->data);
    b->data = 0;
    b->len = 0;
    b->cap = 0;
}

int strbuf_append_n(StrBuf *b, const char *s, int len) {
    if (len <= 0)
        return 1;
    if (!strbuf_ensure(b, len))
        return 0;
    memcpy(b->data + b->len, s, (size_t)len);
    b->len += len;
    b->data[b->len] = '\0';
    return 1;
}

int strbuf_append(StrBuf *b, const char *s) {
    return strbuf_append_n(b, s, (int)strlen(s));
}

int strbuf_appendf(StrBuf *b, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char tmp[256];
    int  n = vsnprintf(tmp, sizeof(tmp), fmt, args);
    va_end(args);

    if (n < 0)
        return 0;
    if (n < (int)sizeof(tmp))
        return strbuf_append_n(b, tmp, n);

    char *buf = malloc((size_t)n + 1);
    if (!buf)
        return 0;
    va_start(args, fmt);
    vsnprintf(buf, (size_t)n + 1, fmt, args);
    va_end(args);

    int ok = strbuf_append_n(b, buf, n);
    free(buf);
    return ok;
}

int strbuf_append_json_string(StrBuf *b, const char *s, int len) {
    if (!strbuf_append_n(b, "\"", 1))
        return 0;

    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':
            if (!strbuf_append(b, "\\\""))
                return 0;
            break;
        case '\\':
            if (!strbuf_append(b, "\\\\"))
                return 0;
            break;
        case '\b':
            if (!strbuf_append(b, "\\b"))
                return 0;
            break;
        case '\f':
            if (!strbuf_append(b, "\\f"))
                return 0;
            break;
        case '\n':
            if (!strbuf_append(b, "\\n"))
                return 0;
            break;
        case '\r':
            if (!strbuf_append(b, "\\r"))
                return 0;
            break;
        case '\t':
            if (!strbuf_append(b, "\\t"))
                return 0;
            break;
        default:
            if (c < 0x20) {
                if (!strbuf_appendf(b, "\\u%04x", c))
                    return 0;
            } else {
                if (!strbuf_append_n(b, (const char *)&c, 1))
                    return 0;
            }
            break;
        }
    }

    return strbuf_append_n(b, "\"", 1);
}

static int token_streq(const char *json, const jsmntok_t *tok, const char *s) {
    int len = tok->end - tok->start;
    return (int)strlen(s) == len && strncmp(json + tok->start, s, (size_t)len) == 0;
}

static int json_tok_skip(jsmntok_t *tokens, int idx) {
    int i = idx + 1;
    if (tokens[idx].type == JSMN_OBJECT) {
        for (int j = 0; j < tokens[idx].size; j++) {
            i = json_tok_skip(tokens, i);
            i = json_tok_skip(tokens, i);
        }
        return i;
    }
    if (tokens[idx].type == JSMN_ARRAY) {
        for (int j = 0; j < tokens[idx].size; j++) {
            i = json_tok_skip(tokens, i);
        }
        return i;
    }
    return i;
}

int protocol_find_key(const ProtocolMessage *msg, int obj_idx, const char *key) {
    if (obj_idx < 0 || obj_idx >= msg->token_count)
        return -1;
    jsmntok_t *tok = &msg->tokens[obj_idx];
    if (tok->type != JSMN_OBJECT)
        return -1;

    int end = json_tok_skip(msg->tokens, obj_idx);
    int i = obj_idx + 1;
    while (i < end) {
        int key_idx = i;
        int val_idx = json_tok_skip(msg->tokens, key_idx);
        if (msg->tokens[key_idx].type == JSMN_STRING &&
            token_streq(msg->json, &msg->tokens[key_idx], key)) {
            return val_idx;
        }
        i = json_tok_skip(msg->tokens, val_idx);
    }
    return -1;
}

int protocol_array_nth(const ProtocolMessage *msg, int arr_idx, int n) {
    if (arr_idx < 0 || arr_idx >= msg->token_count)
        return -1;
    jsmntok_t *tok = &msg->tokens[arr_idx];
    if (tok->type != JSMN_ARRAY || n < 0 || n >= tok->size)
        return -1;

    int i = arr_idx + 1;
    for (int idx = 0; idx < n; idx++) {
        i = json_tok_skip(msg->tokens, i);
    }
    return i;
}

int protocol_method_is(const ProtocolMessage *msg, const char *method) {
    if (msg->method < 0)
        return 0;
    return token_streq(msg->json, &msg->tokens[msg->method], method);
}

int protocol_token_is_null(const ProtocolMessage *msg, int tok_idx) {
    if (tok_idx < 0 || tok_idx >= msg->token_count)
        return 0;
    jsmntok_t *tok = &msg->tokens[tok_idx];
    if (tok->type != JSMN_PRIMITIVE)
        return 0;
    return token_streq(msg->json, tok, "null");
}

int protocol_token_int(const ProtocolMessage *msg, int tok_idx, int *out_value) {
    if (tok_idx < 0 || tok_idx >= msg->token_count)
        return 0;
    jsmntok_t *tok = &msg->tokens[tok_idx];
    if (tok->type != JSMN_PRIMITIVE)
        return 0;

    const char *s = msg->json + tok->start;
    int         len = tok->end - tok->start;
    char        tmp[32];
    if (len <= 0 || len >= (int)sizeof(tmp))
        return 0;
    memcpy(tmp, s, (size_t)len);
    tmp[len] = '\0';

    char *end = 0;
    long  v = strtol(tmp, &end, 10);
    if (!end || *end != '\0')
        return 0;
    *out_value = (int)v;
    return 1;
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F')
        return 10 + (c - 'A');
    return -1;
}

static int utf8_append(StrBuf *b, unsigned int cp) {
    char out[4];
    int  n = 0;
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        n = 1;
    } else if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    } else if (cp <= 0x10FFFF) {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        n = 4;
    } else {
        return 0;
    }
    return strbuf_append_n(b, out, n);
}

static int json_unescape_string(const char *src, int len, StrBuf *out) {
    for (int i = 0; i < len; i++) {
        char c = src[i];
        if (c != '\\') {
            if (!strbuf_append_n(out, &c, 1))
                return 0;
            continue;
        }

        i++;
        if (i >= len)
            return 0;
        char esc = src[i];
        switch (esc) {
        case '"':
        case '\\':
        case '/':
            if (!strbuf_append_n(out, &esc, 1))
                return 0;
            break;
        case 'b': {
            char ch = '\b';
            if (!strbuf_append_n(out, &ch, 1))
                return 0;
            break;
        }
        case 'f': {
            char ch = '\f';
            if (!strbuf_append_n(out, &ch, 1))
                return 0;
            break;
        }
        case 'n': {
            char ch = '\n';
            if (!strbuf_append_n(out, &ch, 1))
                return 0;
            break;
        }
        case 'r': {
            char ch = '\r';
            if (!strbuf_append_n(out, &ch, 1))
                return 0;
            break;
        }
        case 't': {
            char ch = '\t';
            if (!strbuf_append_n(out, &ch, 1))
                return 0;
            break;
        }
        case 'u': {
            if (i + 4 >= len)
                return 0;
            int h1 = hex_val(src[i + 1]);
            int h2 = hex_val(src[i + 2]);
            int h3 = hex_val(src[i + 3]);
            int h4 = hex_val(src[i + 4]);
            if (h1 < 0 || h2 < 0 || h3 < 0 || h4 < 0)
                return 0;
            unsigned int cp = (unsigned int)((h1 << 12) | (h2 << 8) | (h3 << 4) | h4);
            i += 4;

            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (i + 6 < len && src[i + 1] == '\\' && src[i + 2] == 'u') {
                    int l1 = hex_val(src[i + 3]);
                    int l2 = hex_val(src[i + 4]);
                    int l3 = hex_val(src[i + 5]);
                    int l4 = hex_val(src[i + 6]);
                    if (l1 >= 0 && l2 >= 0 && l3 >= 0 && l4 >= 0) {
                        unsigned int low = (unsigned int)((l1 << 12) | (l2 << 8) | (l3 << 4) | l4);
                        if (low >= 0xDC00 && low <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                            i += 6;
                        }
                    }
                }
            }

            if (!utf8_append(out, cp))
                return 0;
            break;
        }
        default:
            return 0;
        }
    }
    return 1;
}

int protocol_token_string_dup(const ProtocolMessage *msg, int tok_idx, char **out, int *out_len) {
    if (tok_idx < 0 || tok_idx >= msg->token_count)
        return 0;
    jsmntok_t *tok = &msg->tokens[tok_idx];
    if (tok->type != JSMN_STRING)
        return 0;

    const char *src = msg->json + tok->start;
    int         len = tok->end - tok->start;

    StrBuf buf;
    strbuf_init(&buf);
    if (!json_unescape_string(src, len, &buf)) {
        strbuf_free(&buf);
        return 0;
    }
    if (!strbuf_ensure(&buf, 0)) {
        strbuf_free(&buf);
        return 0;
    }
    buf.data[buf.len] = '\0';

    *out = buf.data;
    if (out_len)
        *out_len = buf.len;
    return 1;
}

int protocol_parse_message(const char *json, int len, ProtocolMessage *msg, char *err,
                           int err_len) {
    memset(msg, 0, sizeof(*msg));
    msg->json = json;
    msg->len = len;
    msg->tokens = 0;
    msg->token_count = 0;
    msg->root = -1;
    msg->method = -1;
    msg->id = -1;
    msg->params = -1;

    jsmn_parser parser;
    jsmn_init(&parser);

    int tokcap = 256;
    for (;;) {
        jsmntok_t *tokens = realloc(msg->tokens, (size_t)tokcap * sizeof(jsmntok_t));
        if (!tokens) {
            snprintf(err, (size_t)err_len, "out_of_memory");
            return 0;
        }
        msg->tokens = tokens;
        jsmn_init(&parser);
        int rc = jsmn_parse(&parser, json, (size_t)len, msg->tokens, (unsigned int)tokcap);
        if (rc == JSMN_ERROR_NOMEM) {
            tokcap *= 2;
            continue;
        }
        if (rc < 0) {
            snprintf(err, (size_t)err_len, "json_parse_error");
            return 0;
        }
        msg->token_count = rc;
        break;
    }

    if (msg->token_count <= 0 || msg->tokens[0].type != JSMN_OBJECT) {
        snprintf(err, (size_t)err_len, "json_root_not_object");
        return 0;
    }

    msg->root = 0;
    msg->method = protocol_find_key(msg, msg->root, "method");
    msg->id = protocol_find_key(msg, msg->root, "id");
    msg->params = protocol_find_key(msg, msg->root, "params");
    return 1;
}

void protocol_message_free(ProtocolMessage *msg) {
    free(msg->tokens);
    msg->tokens = 0;
    msg->token_count = 0;
}

static int append_token_json(StrBuf *b, const ProtocolMessage *msg, int tok_idx) {
    if (tok_idx < 0 || tok_idx >= msg->token_count)
        return 0;
    jsmntok_t *tok = &msg->tokens[tok_idx];
    if (tok->type == JSMN_STRING) {
        if (!strbuf_append_n(b, "\"", 1))
            return 0;
        if (!strbuf_append_n(b, msg->json + tok->start, tok->end - tok->start))
            return 0;
        return strbuf_append_n(b, "\"", 1);
    }
    return strbuf_append_n(b, msg->json + tok->start, tok->end - tok->start);
}

int protocol_append_id(StrBuf *b, const ProtocolMessage *msg) {
    if (msg->id < 0)
        return 0;
    return append_token_json(b, msg, msg->id);
}

void protocol_write_message(FILE *out, const char *json, int len) {
    fprintf(out, "Content-Length: %d\r\n\r\n", len);
    fwrite(json, 1, (size_t)len, out);
    fflush(out);
}
