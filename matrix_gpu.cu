#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
 
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
 
    if (row < N && col < N) { 
        float sum = 0.0f; 
        for (int k = 0; k < N; k++) { 
            sum += A[row * N + k] * B[k * N + col]; 
        } 
        C[row * N + col] = sum; 
    } 
}
 
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
 
    // Define the grid and block dimensions
    // 16 is chosen as it is used for tile size in subsequent parts
    dim3 block(16, 16);
    dim3 grid((N + block.x-1) / block.x, (N + block.y) / block.y);


    // Measure time by using CUDA events, because Naive code seemed faster with smaller N values
    // possible because small N means everything fits in cache, and tiling seemed slower due to:
    // 1. additional memory transfers to shared memory
    // 2. added sync overhead for threads
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    clock_t start_cpu = clock();  
    
    cudaEventRecord(start);
    matrixMultiplyGPU<<<grid, block>>>(gpuA, gpuB, gpuC, N);
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

    return 0; 
}