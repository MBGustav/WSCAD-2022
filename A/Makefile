CC=gcc
NVCC=nvcc
EXEC=knn
FLAGS = -O3 -DINPUT -Wextra -DNDEBUG 

#CUDA execution 
GPU_EXEC=$(EXEC)-cuda
GPU_FLAGS= -DINPUT -DNDEBUG

# openMP execution
MP_EXEC=$(EXEC)-openMP
MP_FLAGS= $(FLAGS) -fopenmp

#generator input file
INPUT_GEN=input-generator
N_POINTS=1000
N_LABELS=4

generate_input:
	$(CC) -o $(INPUT_GEN) $(FLAGS) $(INPUT_GEN).c
	./$(INPUT_GEN) $(N_POINTS) $(N_LABELS)


build_gpu_cuda: 
	$(NVCC) -o $(GPU_EXEC) $(GPU_FLAGS) $(GPU_EXEC).cu

build_cpu_openmp:
	$(CC) -o $(MP_EXEC) $(MP_FLAGS) $(MP_EXEC).c 

build_cpu_serial:
	$(CC) -o $(EXEC) $(MP_FLAGS) $(EXEC).c 


run_gpu_cuda: build_gpu_cuda
	./$(GPU_EXEC)

run_cpu_openmp: build_cpu_openmp
	./$(MP_EXEC)

run_cpu_serial: build_cpu_serial
	./$(EXEC) 


clean:
	rm -rf $(EXEC) $(GPU_EXEC) $(MP_EXEC) $(INPUT_GEN) input.txt *.o

