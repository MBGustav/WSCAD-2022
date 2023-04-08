#include <stdio.h>
#include <stdlib.h>


//Structs Redefinition -- better to send to GPU
typedef struct {
    float x, y;
    char label;
} GPUPoint;

typedef struct Point{
    float x, y;
    // char label;
} Point;

// ==== About GPU =====
#define NTHREADS 256 
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
// ====================

void print_point(GPUPoint point){
    printf("\n(%f,%f,%c)\n",point.x, point.y, point.label);
}
void on_error() {
    printf("Invalid input file.\n");
    exit(1);
}
//Functions input

int read_number_of_points(){
    int n;
    if(scanf(" n_points=%d \n", &n) != 1) on_error();
    return n;
}

int read_k(){
    int n;
    if(scanf(" k=%d \n", &n) != 1) on_error();
    return n;
}


GPUPoint read_gpu_point() {
    float x, y;
    char c;
    if (scanf("(%f,%f,%c)\n", &x, &y,&c) != 3)  on_error();
    GPUPoint point;
    point.x     = x;
    point.y     = y;
    point.label = c;

    return point;
}

Point read_point() {
    float x, y;
    if (scanf(" (%f ,%f) ", &x, &y) != 2)  on_error();
    Point point;
    point.x = x;
    point.y = y;
    return point;
}

char most_frequent(char *array, int k){
    char most_freq = array[0];
    // printf("most freq %c", most_freq);
    int most_freq_count = 1; 
    int curr_freq = 1;
    for(int i = 1; i < k; i++){

        if(array[i] != array[i-1]){
            if(curr_freq > most_freq_count){
                most_freq = array[i-1];
                most_freq_count = curr_freq;
            }
            curr_freq = 1;
        } 
        curr_freq++;

        if(i == k-1 && curr_freq > most_freq_count){
            most_freq = array[i-1];
            most_freq_count = curr_freq;
        }
    }
    return most_freq;
}


//Kernels  && functions Declaration ===================
__forceinline__ __device__ float distance_no_sqrt(GPUPoint a, GPUPoint b){
    return ((b.x - a.x) * ((b.x - a.x))) + ((b.y - a.y) * (b.y - a.y));
}


//Smallest numbers from Point - Kernel
 __global__ void k_smallest(GPUPoint *arr,GPUPoint to_eval, int n, int k, char *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int i, j, min_index;
    GPUPoint temp;
    
    
    for (i = tid; i < k; i += stride) {
        min_index = i;
        for (j = i + 1; j < n; j++) {
            if(distance_no_sqrt(arr[j], to_eval) < distance_no_sqrt(arr[min_index], to_eval))
                min_index = j;
        }

        temp = arr[min_index];
        arr[min_index] = arr[i];
        arr[i] = temp;
    }
    
    // __syncthreads();
    //Write results --> smallest distances from "evaluate"
    if (tid == 0) {
        for (i = 0; i < k; i++) {
            result[i] = arr[i].label;
        }
    }
}

//Kernel Wrapper - making things easier =D 
inline char wrapper_kSmallest(GPUPoint *arr, int n, Point P , int k){

    // const values  -- to kernel definitions && most_freq function
    int sizeof_arr = n * sizeof(GPUPoint);
    int sizeof_res = k * sizeof(char);
    int NBLOCKS = (int) (NTHREADS + n-1)/NTHREADS; 

    GPUPoint *d_arr, to_eval;
    to_eval.x = P.x;
    to_eval.y = P.y;
    char *d_result, *result;
    result = (char*) malloc(sizeof(sizeof_res));
    //Alloc mem. space
    cudaMalloc(&d_arr, sizeof_arr);
    checkCudaErrors(cudaGetLastError());

    cudaMalloc(&d_result, sizeof_res);
    checkCudaErrors(cudaGetLastError());

    //Copy from host
    cudaMemcpy(d_arr, arr, sizeof_arr, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());


    k_smallest<<<NBLOCKS, NTHREADS>>>(d_arr, to_eval, n, k, d_result);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    //Returns an array ordered by smaller distances
    cudaMemcpy(result, d_result, sizeof_res, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());

    //TODO: send to gpu ?  histogram calculus 
    char most_freq = most_frequent(result, k);

    //free Memory from device
    cudaFree(d_arr);
    cudaFree(d_result);

    //returns the most near neighbout
    return most_freq;
}


