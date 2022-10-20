#include <iostream>
#include <vector>
#include <cstdio>
#include <cassert>

int main() {
	int n,s;
	std::cin >> n >> s;
	std::vector<std::vector<int> > matrix(n, std::vector<int>(n));
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++) 
			scanf("%d", &(matrix[i][j]));

	assert(s<n);
	assert(n<8000);
	assert(s>0);
	assert(n>0);

	int sum = 0;
	for(int i=0;i<s;i++)
		for(int j=0;j<s;j++)
			sum += matrix[i][j];

	int mxSum = sum;

	int dir = 1;
	int i = 0;
	int j = 0;
	while(i+s <= n) {
		if( (j+s == n && dir==1) || (j==0 && dir==-1) ) { 
			int exitingRow = i;
			int enteringRow = exitingRow+s;
			if(enteringRow >=n) break;

			for(int k=0;k<s;k++) { sum -= matrix[exitingRow][k+j]; } 
			for(int k=0;k<s;k++) { sum += matrix[enteringRow][k+j];	} 
			
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

			for(int k=0;k<s;k++) {sum -= matrix[k+i][exitingCol]; } 
			for(int k=0;k<s;k++) {sum += matrix[k+i][enteringCol];	} 
			j+=dir;
		}
		mxSum = std::max(mxSum,sum);
	}
	printf("%d\n",mxSum);
}