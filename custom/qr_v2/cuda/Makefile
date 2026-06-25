NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math
MAGMA_HOME ?= /ictstr01/home/haicu/titouan.breton/micromamba/envs/magma-qr
MAGMA_CFLAGS ?= -I$(MAGMA_HOME)/include
MAGMA_LDFLAGS ?= -L$(MAGMA_HOME)/lib
BENCH_LIBS ?= -lmagma -lcusolver -lcublas -lcusparse
PROFILE_LIBS ?= -lcublas

KERNEL ?= copy_stub
PROFILE ?= blocked
PROFILE_FLAGS ?=
PROFILE_OUT ?= profile_$(PROFILE)

KERNEL_FILE = kernels/$(KERNEL).cu
PROFILE_FILE = profile/profile_$(PROFILE).cu
BENCH_COMPILE = $(NVCC) $(NVCCFLAGS) -I. $(MAGMA_CFLAGS)
PROFILE_COMPILE = $(NVCC) $(NVCCFLAGS) -I. $(PROFILE_FLAGS)

.PHONY: all correctness profile clean

all:
	mkdir -p build
	$(BENCH_COMPILE) bench.cu $(KERNEL_FILE) $(MAGMA_LDFLAGS) $(BENCH_LIBS) -o build/bench_$(KERNEL)

correctness:
	mkdir -p build
	$(BENCH_COMPILE) correctness.cu $(KERNEL_FILE) $(MAGMA_LDFLAGS) $(BENCH_LIBS) -o build/correctness_$(KERNEL)

profile:
	mkdir -p build
	$(PROFILE_COMPILE) $(PROFILE_FILE) $(KERNEL_FILE) $(PROFILE_LIBS) -o build/$(PROFILE_OUT)

clean:
	rm -rf build
