#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
 
#define TILE_WIDTH 16
 
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) { 
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; 
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH]; 
 
    int bx = blockIdx.x; int by = blockIdx.y; 
    int tx = threadIdx.x; int ty = threadIdx.y; 
 
    int Row = by * TILE_WIDTH + ty; 
    int Col = bx * TILE_WIDTH + tx; 
 
    float Pvalue = 0.0; 
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) { 
        if (Row < N && (m*TILE_WIDTH+tx) < N) 
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx]; 
        else 
            ds_A[ty][tx] = 0.0f; 
 
        if (Col < N && (m*TILE_WIDTH+ty) < N) 
            ds_B[ty][tx] = B[(m*TILE_WIDTH + ty) * N + Col]; 
        else 
            ds_B[ty][tx] = 0.0f; 
 
        __syncthreads(); 
 
        for (int k = 0; k < TILE_WIDTH; ++k) 
            Pvalue += ds_A[ty][k] * ds_B[k][tx]; 
        __syncthreads(); 
    } 
 
    if (Row < N && Col < N) 
        C[Row * N + Col] = Pvalue; 
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
    dim3 grid(16, 16);
    dim3 block((size + 15) / 16, (size + 15) / 16);


    // Measure time by using CUDA events, because Naive code seemed faster with smaller N values
    // possible because small N means everything fits in cache, and tiling seemed slower due to:
    // 1. additional memory transfers to shared memory
    // 2. added sync overhead for threads
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    clock_t start = clock(); // get CPU start time 
    cudaEventRecord(start_gpu); // get GPU start time
    matrixMultiplyTiled<<<grid, block>>>(gpuA, gpuB, gpuC, size);
    cudaEventRecord(stop_gpu); // get GPU stop time
    cudaDeviceSynchronize(); // Wait for GPU
    clock_t end = clock(); // get CPU stop time

    cudaMemcpy(C, gpuC, size, cudaMemcpyDeviceToHost);
    
    double elapsed_cpu = (double)(end - start) / CLOCKS_PER_SEC; 
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    printf("CPU clocks execution time (N=%d): %f seconds\n", N, elapsed_cpu); 
    printf("GPU execution time (N=%d): %f seconds\n", N, milliseconds/1000); 
 

    // Deallocate both CPU and GPU memory
    free(A); free(B); free(C); 
    cudaFree(gpuA); cudaFree(gpuB); cudaFree(gpuC);

    return 0; 
}