CXX=gcc
CXX_FLAGS=-O3

NVCC=nvcc
NVCC_FLAGS=-arch=sm_70 -O3

NSYS=nsys profile
NSYSFLAGS=--stats=true --force-overwrite=true

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
	

nbody_v$(VERSION)_profile:
	$(NSYS) $(NSYSFLAGS) -o nbody_v$(VERSION)/Profiling/nbody-report ./nbody_v$(VERSION)


clean:
	rm *.txt input-generator nbody-*
