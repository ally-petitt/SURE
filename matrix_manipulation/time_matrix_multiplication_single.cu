#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <iostream>
#include <fstream>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// export PROGRAM=time_matrix_multiplication_single.cu
// nvcc -c $PROGRAM.cu -o $PROGRAM.o && mpic++ -o $PROGRAM $PROGRAM.o -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
// mpirun -n1 ./$PROGRAM


using data_type = double;

__host__ std::vector<data_type> generateRandomMatrix(int size) {
    int totalElements = size * size;
    std::vector<data_type> matrix(totalElements);

    // Seed for random number generator
    //std::srand(statiResultcast<unsigned int>(std::time(nullptr)));

    // Initialize matrix with random values
    for (int i = 0; i < totalElements; ++i) {
        matrix[i] = static_cast<data_type>(std::rand() % 100); // Random integers (0-99)
    }

    return matrix;
}


__host__ void doRandomMatrixMutliplication(int start_size, int end_size, int increment, int iterations_per_size){
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;
	
	// begin printing our CSV data to stdout
	printf("Matrix multiplication\nMatrix Size, Execution Time(s), Number of Iterations\n");

	// begin timer
	clock_t start = clock();
	for (int size = start_size; size <= end_size; size = size+ increment){
		printf("%d",size);
		for (int i = 0; i <= iterations_per_size; i++){
			// create cuBLAS handle
			(cublasCreate(&cublasH));

			(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
			(cublasSetStream(cublasH, stream));
			cublasHandle_t cublasH = NULL;
			cudaStream_t stream = NULL;


			const std::vector<data_type> A = generateRandomMatrix(size);
			const std::vector<data_type> B = generateRandomMatrix(size);

			std::vector<data_type> C(size * size);
			const data_type alpha = 1.0;
			const data_type beta = 0.0;

			data_type *d_A = nullptr;
			data_type *d_B = nullptr;
			data_type *d_C = nullptr;

			cublasOperation_t transa = CUBLAS_OP_N;
			cublasOperation_t transb = CUBLAS_OP_N;


			/* step 1: create cublas handle, bind a stream */

			/* step 2: copy data to device */
			(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
			(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
			(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

			(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
					       stream));
			(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
					       stream));

			/* step 3: compute */

			clock_t start = clock();
			(
			cublasDgemm(cublasH, transa, transb, size, size, size, &alpha, d_A, size, d_B, size, &beta, d_C, size));

			/* step 4: copy data to host */
			(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
					       stream));

			(cudaStreamSynchronize(stream));
			
			


			/* free resources */
			(cudaFree(d_A));
			(cudaFree(d_B));
			(cudaFree(d_C));
			(cublasDestroy(cublasH));

			(cudaStreamDestroy(stream));
		}
		clock_t end = clock();
		long double num_seconds = (long double)(end - start) / CLOCKS_PER_SEC;
		printf(",%Lf,%d\n",  num_seconds, iterations_per_size); 
	}
	(cudaDeviceReset());

}



// __host__ void doRandomMatrixInversion(int size){
// 	cublasHandle_t cublasH = NULL;
// 	cudaStream_t stream = NULL;

// 	printf("Matrix inversion size: %d: ", size);


// 	const std::vector<data_type> A = generateRandomMatrix(size);
// 	const std::vector<data_type> Result = generateRandomMatrix(size);
// 	const std::vector<data_type> Info = generateRandomMatrix(size);

// 	std::vector<data_type> C(size * size);

// 	data_type *d_A = nullptr;
// 	data_type *d_Result = nullptr;
// 	data_type *d_Info = nullptr;

// 	/* step 1: create cublas handle, bind a stream */
// 	(cublasCreate(&cublasH));

// 	(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
// 	(cublasSetStream(cublasH, stream));

// 	/* step 2: copy data to device */
// 	(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
// 	(cudaMalloc(reinterpret_cast<void **>(&d_Result), sizeof(data_type) * Result.size()));
// 	(cudaMalloc(reinterpret_cast<void **>(&d_Info), sizeof(data_type) * Info.size()));

// 	(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
// 			       stream));
// 	(cudaMemcpyAsync(d_Info, Info.data(), sizeof(data_type) * Info.size(), cudaMemcpyHostToDevice,
// 			       stream));

// 	/* step 3: compute */
// 	//clock_t start = clock();
// 	(cublasDmatinvBatched(cublasH, 
// 		    size, 
// 		    (const double *const *) d_A, 
// 		    size, 
// 		    (double *const *)d_Result, 
// 		    size, 
// 		    (int *)d_Info, 
// 		    size));

// 	/* step 4: copy data to host */
// 	(cudaMemcpyAsync((void *)Result.data(), d_Result, sizeof(data_type) * Result.size(), cudaMemcpyDeviceToHost,
// 			       stream));

// 	/*
// 	cudaStreamSynchronize(stream);
// 	clock_t end = clock();
// 	long double num_seconds = (long double)(end - start) / CLOCKS_PER_SEC;

// 	printf(" %Lf seconds\n",  num_seconds);    
// 	*/	    

// 	/* free resources */
// 	(cudaFree(d_A));
// 	(cudaFree(d_Info));
// 	cudaFree(d_Result);
// 	//	    (cudaFree(d_Result));

        

//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
//             // Handle error appropriately
//         }
	            
// 	(cublasDestroy(cublasH));

// 	(cudaStreamDestroy(stream));

// 	(cudaDeviceReset());
// }




int main(int argc, char** argv) {
	
	doRandomMatrixMutliplication(1500, 15000, 1500, 20);
    
    return 0;
}
