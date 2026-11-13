# Builds the native PFA engine (pfa/anneal.py loads it).
# The SIMD path (SVE, NEON, AVX-512, AVX2 or scalar) follows from ARCH_FLAGS; the default targets this machine.
# -ffp-contract=off is required: the kernel rescores the winning chunk of spins and looks for the exact maximum
# the main loop found, so the compiler may not fuse multiply-adds in one place and not the other.

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
ARCH_FLAGS ?= -march=native
else
ARCH_FLAGS ?= -mcpu=native
endif

SRC := pfa/native/pfa.c
LIB := pfa/native/libpfa.so

# pfa.c includes the kernels (spin.c, categorical.c, sparse_spin.c, sparse_categorical.c), sums.c, eliminate.c and
# fold.c, and the first two kernels the <name>_<path>.c that match the target
$(LIB): $(SRC) $(wildcard pfa/native/*.c)
	$(CC) -O3 -ffp-contract=off -fPIC -shared $(ARCH_FLAGS) -o $@ $(SRC) -lm

.PHONY: clean
clean:
	rm -f $(LIB)
