# Builds the native PFA engine (ising_solvers/pfa_native.py loads it).
# The SIMD path (SVE, NEON, AVX-512, AVX2 or scalar) follows from ARCH_FLAGS; the default targets this machine.
# -ffp-contract=off is required: the kernel rescores the winning chunk of spins and looks for the exact maximum
# the main loop found, so the compiler may not fuse multiply-adds in one place and not the other.

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
ARCH_FLAGS ?= -march=native
else
ARCH_FLAGS ?= -mcpu=native
endif

SRC := ising_solvers/native/pfa_native.c
LIB := ising_solvers/native/libpfa_native.so

# pfa_native.c includes the one pfa_native_<path>.c that matches the target
$(LIB): $(SRC) $(wildcard ising_solvers/native/pfa_native_*.c)
	$(CC) -O3 -ffp-contract=off -fPIC -shared $(ARCH_FLAGS) -o $@ $(SRC)

.PHONY: clean
clean:
	rm -f $(LIB)
