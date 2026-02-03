CC = cc
CFLAGS = -O2 -Wall -Wextra -Wno-unused-parameter -std=c99
INCLUDES = -Isrc -I../highlight/src

SOURCES = src/main.c src/protocol.c src/session.c src/jsmn.c \
          ../highlight/src/lexer.c ../highlight/src/highlight.c

TARGET = shakar-lsp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) src/protocol.h src/session.h src/jsmn.h \
           ../highlight/src/lexer.h ../highlight/src/highlight.h ../highlight/src/token_types.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SOURCES)

clean:
	rm -f $(TARGET)
