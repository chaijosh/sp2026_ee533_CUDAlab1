#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024; // allow matrix size as input 
    size_t size = N * N * sizeof(float); 
 
    float *A = (float *)malloc(size); 
    float *B = (float *)malloc(size); 
    float *C = (float *)malloc(size); 

    // create GPU memory variables
    float *gpuA, *gpuB, *gpuC;
    cudaMalloc((void **)&gpuA, size);
    cudaMalloc((void **)&gpuB, size);
    cudaMalloc((void **)&gpuC, size);

    for (int i = 0; i < N * N; i++) { 
        A[i] = rand() % 100 / 100.0f; 
        B[i] = rand() % 100 / 100.0f; 
    } 

    // Copy contents of A and B to GPU memory
    cudaMemcpy(gpuA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, size, cudaMemcpyHostToDevice);

    // Create cuBLAS struct
    cublasHandle_t cublas1;
    cublasCreate(&cublas1);
    
    // initialize alpha and beta (multiplication weights during matmult operations)
    float alpha = 1.0f, beta = 0.0f;

    // Measure time by using CUDA events, because Naive code seemed faster with smaller N values
    // possible because small N means everything fits in cache, and tiling seemed slower due to:
    // 1. additional memory transfers to shared memory
    // 2. added sync overhead for threads
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Apparently, GPU resources required for this function are in sleep mode at 
    // the beginning of execution, and if I run this only once, execution times are
    // very high (80ms for N=1024)...
    
     // I realized this when I measured the time taken for 2 function calls, and it came out
     // identical to execution time of single function call.
     // Calling once before measuring execution time seems to enable the GPU HW, and the subsequent 
     // function calls are fast (as expected)
    cublasSgemm(cublas1,  
            CUBLAS_OP_N, CUBLAS_OP_N,  
            N, N, N,  
            &alpha,  
            gpuB, N,  
            gpuA, N,  
            &beta,  
            gpuC, N);
    
    clock_t start_cpu = clock();
    cudaEventRecord(start);
    cublasSgemm(cublas1,  
            CUBLAS_OP_N, CUBLAS_OP_N,  
            N, N, N,  
            &alpha,  
            gpuB, N,  
            gpuA, N,  
            &beta,  
            gpuC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    clock_t stop_cpu = clock();

    double elapsed_cpu = (double)(stop_cpu - start_cpu)/CLOCKS_PER_SEC;
    float elapsed_gpu = 0;
    cudaEventElapsedTime(&elapsed_gpu, start, stop);

    // Bring results back and print
    cudaMemcpy(C, gpuC, size, cudaMemcpyDeviceToHost);
    printf("CPU execution time (N=%d): %f milliseconds\n", N, elapsed_cpu*1000);
    printf("GPU execution time (N=%d): %f milliseconds\n", N, elapsed_gpu);

    // Deallocate both CPU and GPU memory
    free(A); free(B); free(C); 
    cudaFree(gpuA); cudaFree(gpuB); cudaFree(gpuC);
    cublasDestroy(cublas1);

    return 0; 
}