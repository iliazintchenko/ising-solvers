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

# pfa_native.c includes the kernels (pfa_native_spin.c, pfa_native_categorical.c, pfa_native_sparse_spin.c,
# pfa_native_sparse_categorical.c), pfa_native_sums.c and pfa_native_eliminate.c, and the first two kernels the
# pfa_native_<name>_<path>.c that match the target
$(LIB): $(SRC) $(wildcard ising_solvers/native/pfa_native_*.c)
	$(CC) -O3 -ffp-contract=off -fPIC -shared $(ARCH_FLAGS) -o $@ $(SRC) -lm

.PHONY: clean
clean:
	rm -f $(LIB)
