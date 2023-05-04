#ifndef __CUDAKERNEL_H__
#define __CUDAKERNEL_H__

#define DT 0.01f // time step
#define SOFTENING 1e-9f // to avoid zero divisors
#define EPSILON (0.000005f)
#define N_THREADS 32

#include "../include/helpers.cuh"

/*
 * Each body holds coordinate positions (i.e., x, y, and z) and
 * velocities (i.e., vx, vy, and vz).
 */

// Structures used in this example
__host__ __device__
 typedef struct { 
	float x, y, z, vx, vy, vz;
	} Body;

__host__ __device__
typedef struct { 
    int x, y, z, vx, vy, vz; 
    } Body_int;


//Kernel Declaration
__global__ void NbodyForceGPU(Body *p,float dt,int nbodies){

	int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;


    for(int i = tx; i < nbodies; i += stride){
		float dx,dy,dz, sqrd_dist, inv_dist, inv_dist3;
		float fx = 0.0f; 
		float fy = 0.0f; 
		float fz = 0.0f;
		//Calculate body forces
		for(int j = 0; j < nbodies; j++){
			dx = p[j].x - p[i].x;
			dy = p[j].y - p[i].y;
			dz = p[j].z - p[i].z;
			sqrd_dist = dx*dx + dy*dy + dz*dz + SOFTENING;
            inv_dist = rsqrt(sqrd_dist);
			// inv_dist = 1 / sqrt(sqrd_dist);
			inv_dist3 = inv_dist * inv_dist * inv_dist;

			fx += dx * inv_dist3; 
			fy += dy * inv_dist3; 
			fz += dz * inv_dist3;
		}
		__syncthreads();// devo deixar a sincronização ? 
        //ou usar cuda Atomic Add /??
		p[i].vx += dt*fx; 
		p[i].vy += dt*fy; 
		p[i].vz += dt*fz;   
        
        __syncthreads(); //barreira da oper, anterior
        p[i].x += p[i].vx * dt;
		p[i].y += p[i].vy * dt;
		p[i].z += p[i].vz * dt;
    }
}



void Nbody_wrapper(Body *h_bodies, int nbodies, int num_iter){
    
    const int NThreads = 32;
    const int Blocks = 80 * 32;
    const int sizeof_bodies = sizeof(float)* nbodies; 
    Timer T;
    Timer OverAll;
    Body *d_bodies, *result;
    
    T.start();OverAll.start();

    cudaMallocHost(&result, sizeof_bodies);
    cudaMalloc    (&d_bodies, sizeof_bodies);
    check_last_error();
    T.stop("Memory Allocation");
    //copy bodies to   GPU
    T.start();
    cudaMemcpy(d_bodies, h_bodies, sizeof_bodies, cudaMemcpyHostToDevice);
    check_last_error();
    T.stop("Memory Copy");
    T.start();
    for(int iter=0; iter < num_iter; iter++)
        NbodyForceGPU<<<Blocks,NThreads>>>(d_bodies, DT, nbodies);
    check_last_error();
    T.stop("Execution Time");
    //Copy back to Host
    T.start();
    cudaMemcpy(result, d_bodies, sizeof_bodies, cudaMemcpyHostToHost);
    check_last_error();
    T.stop("Memory Allocation");

    OverAll.stop("Execução Total do Kernel");
}


#endif // __CUDAKERNEL_H__ 