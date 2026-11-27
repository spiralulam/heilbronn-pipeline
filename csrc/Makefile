# heil: the C search engine.   make            -> ./heil
#                               make ARCH=-march=native   (x86 / gcc)
CC     ?= cc
ARCH   ?= -mcpu=native
CFLAGS ?= -O3 $(ARCH) -Wall -Wextra
LDLIBS  = -lm
SRC     = main.c common.c free.c sym.c reloc.c

heil: $(SRC) heil.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

clean:
	rm -f heil

.PHONY: clean
