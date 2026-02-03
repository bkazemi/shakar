#ifndef SHK_LSP_PROTOCOL_H
#define SHK_LSP_PROTOCOL_H

#include <stdio.h>

#define JSMN_HEADER
#include "jsmn.h"

typedef struct {
    const char *json;
    int         len;
    jsmntok_t  *tokens;
    int         token_count;
    int         root;
    int         method;
    int         id;
    int         params;
} ProtocolMessage;

typedef struct {
    char *data;
    int   len;
    int   cap;
} StrBuf;

void strbuf_init(StrBuf *b);
void strbuf_free(StrBuf *b);
int  strbuf_append(StrBuf *b, const char *s);
int  strbuf_append_n(StrBuf *b, const char *s, int len);
int  strbuf_appendf(StrBuf *b, const char *fmt, ...);
int  strbuf_append_json_string(StrBuf *b, const char *s, int len);

int protocol_parse_message(const char *json, int len, ProtocolMessage *msg, char *err, int err_len);
void protocol_message_free(ProtocolMessage *msg);

int protocol_method_is(const ProtocolMessage *msg, const char *method);
int protocol_find_key(const ProtocolMessage *msg, int obj_idx, const char *key);
int protocol_array_nth(const ProtocolMessage *msg, int arr_idx, int n);
int protocol_token_is_null(const ProtocolMessage *msg, int tok_idx);
int protocol_token_int(const ProtocolMessage *msg, int tok_idx, int *out_value);
int protocol_token_string_dup(const ProtocolMessage *msg, int tok_idx, char **out, int *out_len);

void protocol_write_message(FILE *out, const char *json, int len);
int  protocol_append_id(StrBuf *b, const ProtocolMessage *msg);

#endif /* SHK_LSP_PROTOCOL_H */
