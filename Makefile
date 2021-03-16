CC = mpicc

CFLAGS = -O3 -Wall -Wno-unused-result -g -rdynamic
LDLIBS = -lHYPRE -lm

.PHONY: clean

cavity: cavity.c
	$(CC) $< -o $@ $(CFLAGS) $(LDLIBS)

clean:
	rm -rf cavity
