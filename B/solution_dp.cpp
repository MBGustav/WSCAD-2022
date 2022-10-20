#include <iostream>
#include <vector>
#include <cstdio>
#include <cassert>
//Parallel SYCL
#include <CL/sycl.hpp>


using namespace std;

int main() {
	
	sycl::gpu_selector selector;
	
	int n,s;
	//n = o tamanho da matrix inicial
	//s = o tamanho da submatriz
	std::cin >> n >> s;
	// std::vector<std::vector<int> > matrix(n, std::vector<int>(n));
	int matrix[n][n];
	int MaxSum = 0;
	int output = 0;
	int _diff = n-s;
	assert(s<n);
	assert(n<8000);
	assert(s>0);
	assert(n>0);

	//leitura da matriz
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++) 
			scanf("%d", &(matrix[i][j]));
	
	try{//Time to paralell ! =D 
		sycl::queue q(selector);
		std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
		
		//range da matriz
		sycl::range<2> m_size(n, n);
		sycl::range<2> sub_range(n-s, n-s);


		sycl::buffer<int, 2> matrix_buf(,m_size);
		sycl::buffer<int, 2> sub_buf(sub_range);
		sycl::buffer<int> out_buf{output};

		
		//Load to accelerator 
		q.submit([&](sycl::handler &h) {
			sycl::accessor matrix_acc(matrix_buf, h, sycl::read_only);
			
			//como melhorar para accessor local ?? 
			sycl::accessor<int, 1> MaxSum{, h};
			// sycl::accessor MaxSum(gray_buf, h, sycl::write_only, sycl::no_init);

			h.parallel_for(m_size, [=](auto idx) {
				//MaxSum in local_accessor
				MaxSum[0]+= matrix_acc[idx]; 
			});
		});

		q.submit([&](sycl::handler &h){ 
			//cria um accessor para armazenar todas as somas, depois pegamos o maior
			sycl::accessor sub_acc(sub_buf, h, sycl::write_only);
			// sycl::accessor<int> sub_acc{_diff*_diff, h};			
			h.parallel_for(sycl::range<2>(_diff,_diff), [=](auto idx){
				// int offst; 
				int sum = 0;
				int x = idx[0]; 
				int y = idx[1];
				for(auto i = 0; i < _diff; i++){
					for(auto j = 0; j < _diff; j++){
						sum += matrix[x+i][y+j];
					}
				}
				//store in array - 2D
				sub_acc[idx] = sum;
			});
		});
		
		//mais viavel paralelo ou sequencial ?
		q.submit([&](sycl::handler &h){ 
			// agora pegamos o maior			
			sycl::accessor sub_acc(sub_buf, h, sycl::read_only);
			sycl::accessor out_acc(Sum_buf, h, sycl::write_only);
			h.parallel_for(sycl::range<2>(_diff-1,_diff), [=](auto idx){
				int i = idx[0]+1;
				int j = idx[1];
				if(sub_acc[i][j]>sub_acc[0][0])
					sub_ac[0][0] = sub_acc[i][j];
			});
		});


	}catch (exception const &e)
	{
        cout << "An exception is caught!" << endl << e.what() << endl;
        return EXIT_FAILURE;

	}
				
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++) 
			scanf("%d", &(matrix[i][j]));




	

	//total sum matrix - parallel

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