#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define N_ITER 10 // number of simulation iterations
#define DT 0.01f // time step
#define SOFTENING 1e-9f // to avoid zero divisors
#define EPSILON (0.000005f)
#define N_THREADS 32
#define FLOAT_EQ(X,Y)( (fabs((X) - (Y)) <= EPSILON) ? 1 : 0)

/*
 * Each body holds coordinate positions (i.e., x, y, and z) and
 * velocities (i.e., vx, vy, and vz).
 */

// Structures used in this example
typedef struct { 
	float x, y, z, vx, vy, vz;
	} Body;
	typedef struct { 
		int x, y, z, vx, vy, vz; 
		} Body_int;

inline void debugMode(){
	#ifndef ONLINE_JUDGE
	freopen("input.txt", "r", stdin);
	// freopen("output.txt", "w", stdout);
	#endif //ONLINE_JUDGE
}
int AreEqual(Body &p0, Body &p1){
	if(p0.x != p1.x || 
		 p0.y != p1.y ||
		 p0.z != p1.z ||
		 p0.vx != p1.vx ||
		 p0.vy != p1.vy ||
		 p0.vz != p1.vz ) return 0;
	return 1;
}
/*
 * Compute the gravitational impact among all pairs of bodies in 
 * the system.
 */

 int checkResults(Body* b_GPU, Body* b_CPU, int nbodies){
 	int c = 1;
 	for(int i = 0 ; i < nbodies; i++){
 		if(!AreEqual(b_CPU[i], b_GPU[i])){
 			printf("They are not\n");
 			return 0;
 		}
 	}
 	return c;

 }

/*
 * Read a binary dataset with initilized bodies.
 */
 Body* read_dataset(int nbodies) {

	Body *p = (Body *)malloc(nbodies * sizeof(Body));

	for(int i = 0; i < nbodies; i++)
	fscanf(stdin, "%f %f %f %f %f %f\n",&p[i].x,  &p[i].y,  &p[i].z, 
					&p[i].vx, &p[i].vy, &p[i].vz);
	return p;
}

/*
 * Write simulation results into a binary dataset.
 */
 void write_dataset(const int nbodies, Body *bodies, char *fname) {

	// Body_int *bodies_int = (Body_int *)malloc(nbodies * sizeof(Body_int)); 
	FILE *fp;
	fp = fopen(fname, "w");

	for (int i = 0; i < nbodies; i++) {
		fprintf(fp, "%f %f %f %f %f %f\n",bodies[i].x,  bodies[i].y,  bodies[i].z, 
		bodies[i].vx, bodies[i].vy, bodies[i].vz);
	}

}

__global__ void NbodyForceGPU(Body *p,float dt,int nbodies){


	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i<nbodies){

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
			inv_dist = 1 / sqrt(sqrd_dist);
			inv_dist3 = inv_dist * inv_dist * inv_dist;

			fx += dx * inv_dist3; 
			fy += dy * inv_dist3; 
			fz += dz * inv_dist3;
		}
		__syncthreads();
		p[i].vx += dt*fx; 
		p[i].vy += dt*fy; 
		p[i].vz += dt*fz;
	}

}

__global__ void NbodyIteractGPU(Body *bodies,float dt,int nbodies){

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if(i < nbodies){
		
		bodies[i].x += bodies[i].vx * dt;
		bodies[i].y += bodies[i].vy * dt;
		bodies[i].z += bodies[i].vz * dt;
	}
}


int main(int argc,char **argv) {

	debugMode();
	
	char file_gpu[] = "output-cuda.txt";
	int nbodies;
	fscanf(stdin,"%d",&(nbodies));

	int N_BLOCKS =(int) (N_THREADS + nbodies - 1)/N_THREADS;

	Body *bodies = read_dataset(nbodies);
	Body *GPU_bodies = (Body *) malloc(nbodies * sizeof(Body));

	//Allocate Dev pointers
	Body *d_bodies;
	cudaMalloc(&d_bodies, nbodies*sizeof(Body));
	
	//Copy to device
	cudaMemcpy(d_bodies, bodies, nbodies*sizeof(Body), cudaMemcpyHostToDevice);

	/*
	 * At each simulation iteration, interbody forces are computed,
	 * and bodies' positions are integrated.
	 */
	 for (int iter = 0; iter < N_ITER; iter++) {
	 	NbodyForceGPU<<<N_BLOCKS, N_THREADS>>>(d_bodies, DT, nbodies);
		cudaDeviceSynchronize();	//Wait NBodyForce
		NbodyIteractGPU<<<N_BLOCKS, N_THREADS>>>(d_bodies, DT, nbodies); 
	}

	cudaDeviceSynchronize();// Wait Last Result

	//copyToHost - Receive values from GPU
	cudaMemcpy(GPU_bodies, d_bodies, nbodies*sizeof(Body), cudaMemcpyDeviceToHost);

	write_dataset(nbodies, GPU_bodies, file_gpu);

	free(bodies);
	
	exit(EXIT_SUCCESS);
}
