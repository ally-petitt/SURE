#include <cstdio>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <iostream>
#include <fstream>

#include <cublas_v2.h>
#include <cuda_runtime.h>


using data_type = double;


std::vector<data_type> generateRandomMatrix(int size) {
    // Calculate total number of elements in the matrix
    int totalElements = size * size;

    // Create a vector to store the flattened matrix
    std::vector<data_type> matrix(totalElements);

    // Seed for random number generator
    //std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Initialize matrix with random values
    for (int i = 0; i < totalElements; ++i) {
        matrix[i] = static_cast<data_type>(std::rand() % 100); // Random integers (0-99)
    }

    return matrix;
}


int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    
	// create our output file
    FILE *output_file = fopen("matrix_inversion_short.csv", "w");	
    if (!output_file) {
        std::cerr << "Failed to open the file!" << std::endl;
        return 1;
    }
    
    fprintf(output_file, "Matrix Size, Execution Time (seconds)\n");


    for (int i = 1; i <= 85; i++) {
	    int n = 50 * i;
	    printf("Matrix size: %d: ", n);
	    

	    const std::vector<data_type> A = generateRandomMatrix(n); // initialize source matrix
	    const std::vector<data_type> C = generateRandomMatrix(n); // initialize destination matrix
        const std::vector<data_type> Info = generateRandomMatrix((int) n); // vector containing info on our matrix inversions


	    data_type *d_A = nullptr;
	    data_type *d_C = nullptr;
        data_type *d_Info = nullptr;

	    /* step 1: create cublas handle, bind a stream */
	    (cublasCreate(&cublasH));

	    (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	    (cublasSetStream(cublasH, stream));

	    /* step 2: copy data to device */
	    (cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
	    (cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));
        (cudaMalloc(reinterpret_cast<void **>(&d_Info), sizeof(data_type) * Info.size()));

	    (cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
				       stream));
        (cudaMemcpyAsync(d_Info, Info.data(), sizeof(data_type) * Info.size(), cudaMemcpyHostToDevice,
				       stream));

	    /* step 3: compute */
	    clock_t start = clock();
	    (cublasDmatinvBatched(cublasH, 
                            n, 
                            (const double *const *) d_A, 
                            n, 
                            (double *const *)d_C, 
                            n, 
                            (int *)d_Info, 
                            n));

	    /* step 4: copy data to host */
	    (cudaMemcpyAsync((void *)C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
				       stream));

        cudaStreamSynchronize(stream);
	    clock_t end = clock();
        long double num_seconds = (long double)(end - start) / CLOCKS_PER_SEC;

	     printf(" %Lf seconds\n",  num_seconds);    
	    

	    /* free resources */
	    (cudaFree(d_A));
	    (cudaFree(d_Info));
        cudaFree(d_C);

        

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            // Handle error appropriately
        }
	            
	    fprintf(output_file, "%d,%Lf\n", n, num_seconds);

    }

    // clean up used resources
    (cublasDestroy(cublasH));

    (cudaStreamDestroy(stream));

    (cudaDeviceReset());
    fclose(output_file);

    return EXIT_SUCCESS;
}
