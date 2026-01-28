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
    dim3 grid(16, 16);
    dim3 block((size + 15) / 16, (size + 15) / 16);

    clock_t start = clock(); 
    matrixMultiplyGPU<<<grid, block>>>(gpuA, gpuB, gpuC, size);
    // Wait for GPU
    cudaDeviceSynchronize();
    clock_t end = clock(); 
    cudaMemcpy(C, gpuC, size, cudaMemcpyDeviceToHost);
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC; 
    printf("GPU execution time (N=%d): %f seconds\n", N, elapsed); 
 

    // Deallocate both CPU and GPU memory
    free(A); free(B); free(C); 
    cudaFree(gpuA); cudaFree(gpuB); cudaFree(gpuC);

    return 0; 
}