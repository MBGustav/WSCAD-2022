#include <iostream>
#include <vector>
#include <cstdio>
#include <cassert>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
int main() {

	// auto start = high_resolution_clock::now();
	int n,s;
	// int total_threads = omp_get_num_threads();

	//n = o tamanho da matrix inicial
	//s = o tamanho da submatriz
	std::cin >> n >> s;
	// std::vector<std::vector<int> > matrix(n, std::vector<int>(n));
	int matrix[n][n];
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++) 
			scanf("%d", &(matrix[i][j]));

	assert(s<n);
	assert(n<8000);
	assert(s>0);
	assert(n>0);

	int sum = 0;
	//total sum matrix
	
	#pragma omp parallel for reduction(+:sum)
		for(int i=0;i<s;i++)
			for(int j=0;j<s;j++)
				sum += matrix[i][j];

	int mxSum = sum;

	int dir = 1;
	int i = 0;
	int j = 0;
	int k = 0;

	while(i+s <= n) {
		if( (j+s == n && dir==1) || (j==0 && dir==-1) ) { 
			int exitingRow = i;
			int enteringRow = exitingRow+s;
			if(enteringRow >=n) break;
			
			// # pragma omp parallel for
				for(k=0;k<s;k++) { 
						sum -= matrix[exitingRow][k+j]; 
						sum += matrix[enteringRow][k+j];
					} 

			// for(int k=0;k<s;k++) { sum += matrix[enteringRow][k+j];	} 
			
			i++;
			dir*=-1;
		} else {
			int exitingCol,enteringCol;
			if(dir==1) {
				exitingCol = j;
				enteringCol = exitingCol + s;
			} else {
				enteringCol = j-1;
				exitingCol = enteringCol + s;
			}

			// #pragma omp parallel for
			for(k=0;k<s;k++) {
				sum -= matrix[k+i][exitingCol]; 
				sum += matrix[k+i][enteringCol];
				} 
			j+=dir;
		}
		mxSum = std::max(mxSum,sum);
	}
	// auto stop = high_resolution_clock::now();
	
	printf("%d\n",mxSum);
	printf("time: %f\n",duration_cast<milliseconds>(stop - start).count()*1e3);
}