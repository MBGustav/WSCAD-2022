CXX=gcc
CXX_FLAGS=-O3

NVCC=nvcc
NVCC_FLAGS=-arch=sm_70 -O3

BAS_DIR=baseline
INC_DIR=include


gen_input: $(INC_DIR)/input-generator.c
	$(CXX) $(CXX_FLAGS) -o input-generator $(INC_DIR)/input-generator.c
base:
	$(CXX) $(CXX_FLAGS) -o nbody-c $(BAS_DIR)/nbody-c.c -lm


clean:
	rm *.txt input-generator nbody-*
