#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuda-kernels.cuh"

#define N_ITER 10 // number of simulation iterations
#define FLOAT_EQ(X,Y)( (fabs((X) - (Y)) <= EPSILON) ? 1 : 0)

inline void debugMode(){
	#ifndef ONLINE_JUDGE
	freopen("input.txt", "r", stdin);
	// freopen("output.txt", "w", stdout);
	#endif //ONLINE_JUDGE
}
int AreEqual(Body &p0, Body &p1){
	bool equal = true;
	equal = equal && FLOAT_EQ(p0.x , p1.x);
	equal = equal && FLOAT_EQ(p0.y , p1.y);
	equal = equal && FLOAT_EQ(p0.z , p1.z);
	equal = equal && FLOAT_EQ(p0.vx, p1.vx);
	equal = equal && FLOAT_EQ(p0.vy, p1.vy);
	equal = equal && FLOAT_EQ(p0.vz, p1.vz);
	
	return equal;
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

	// Body *p = (Body *)malloc(nbodies * sizeof(Body));
	Body *p; 
	//Allocate in RAM 
	cudaMallocHost(&p, nbodies*sizeof(float));
	
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

int main(int argc,char **argv) {

	// debugMode();	
	char file_gpu[] = "output-cuda_v1.txt";
	int nbodies;
	
	fscanf(stdin,"%d",&(nbodies));

	Body *bodies = read_dataset(nbodies);
	
	/*
	 * At each simulation iteration, interbody forces are computed,
	 * and bodies' positions are integrated.
	 */
	Nbody_wrapper(bodies, nbodies, N_ITER);
	//copyToHost - Receive values from GPU

	write_dataset(nbodies, bodies, file_gpu);

	cudaFree(bodies);
	
	exit(EXIT_SUCCESS);
}
