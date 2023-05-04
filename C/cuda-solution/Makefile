CXX=gcc
NVCC=nvcc
CXX_FLAGS=-O3
NVCC_FLAGS=-arch=sm_70 -O3

BASELINE_DIR=baseline
INCLUDE_DIR=include

VERSION=1
EX_NAME=nbody


gen_input: $(INCLUDE_DIR)/input-generator.c
	$(CXX) $(CXX_FLAGS) -o input-generator $(INCLUDE_DIR)/input-generator.c

base:
	$(CXX) $(CXX_FLAGS) -o $(EX_NAME)-c $(BASELINE_DIR)/$(EX_NAME)-c.c -lm

nbody_v$(VERSION):
	$(NVCC) $(NVCC_FLAGS) -o $(EX_NAME)-v$(VERSION) nbody_v$(VERSION)/$(EX_NAME)-cuda.cu
	


clean:
	rm *.txt input-generator nbody-*
