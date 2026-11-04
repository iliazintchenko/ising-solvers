# Builds the native PFA engine (ising_solvers/pfa_native.py loads it).
# The SIMD path (SVE, NEON, AVX-512, AVX2 or scalar) follows from ARCH_FLAGS; the default targets this machine.
# -ffp-contract=off is required: fused multiply-adds would break bit-identity with pfa_plain.

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
ARCH_FLAGS ?= -march=native
else
ARCH_FLAGS ?= -mcpu=native
endif

SRC := ising_solvers/native/pfa_native.c
LIB := ising_solvers/native/libpfa_native.so

$(LIB): $(SRC)
	$(CC) -O3 -ffp-contract=off -fPIC -shared $(ARCH_FLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(LIB)
