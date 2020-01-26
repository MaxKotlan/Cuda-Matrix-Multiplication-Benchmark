#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

enum comptuationDevice{
    dev_both,
    dev_gpu,
    dev_cpu
};

/*Data structure, which holds a pointer to the elements 
of the matrix, and the number of rows/columns*/
struct squareMatrix{
    int* elements;
    int dimension;
};

__host__ squareMatrix createRandomSquareMatrix(int dimension){
    int  mat_elements = dimension * dimension;
    int* mat = (int*)malloc(sizeof(int)*mat_elements);
    for (int i = 0; i < mat_elements; i++)
        mat[i] = rand()%10;
    return {mat, dimension};
}

__host__ void printSquareMatrix(squareMatrix mat){
    for (int i = 0; i < mat.dimension*mat.dimension; i++){
        if (i % mat.dimension == 0 && i != 0) printf("\n");
        printf("%d ", mat.elements[i]);
    }
    printf("\n\n");
}

__host__ squareMatrix multiplyMatrices(squareMatrix a, squareMatrix b){
    if (a.dimension != b.dimension) exit(1);
    squareMatrix result = { (int*)malloc(sizeof(int)*a.dimension*a.dimension), a.dimension };
    
    for (int i = 0; i < result.dimension; i++){
        for (int j = 0; j < result.dimension; j++){
            result.elements[j + result.dimension*i] = 0;
            for (int k = 0; k < result.dimension; k++)
                result.elements[j + result.dimension*i] += a.elements[k + result.dimension*i] * b.elements[j + result.dimension*k];
        }
    }
    return result;
}

__global__ void multiplyMatricesParallel(int* mat_a, int* mat_b, int* mat_results, int dimension){
    int idx = blockDim.x * threadIdx.y + threadIdx.x;

    mat_results[idx] = 0;
    for (int i = 0; i < blockDim.x; i++){
        mat_results[idx] += mat_a[i + blockDim.x * threadIdx.y] * mat_b[threadIdx.x + blockDim.x * i];
    }
}

__host__ void printTime(clock_t totaltime, int dimension, char* type){
    int msec = totaltime * 1000 / CLOCKS_PER_SEC;
    printf("Multiplying %dx%d X %dx%d took %d msec using the %s\n--------------------------------------------------------------------------------------------\n", 
    dimension, dimension, dimension, dimension, msec, type);
}

/*
 Found on the stack overflow:  https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 Throws errors if cuda command doesn't return Success
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ void testHostPreformance(squareMatrix mat_a, squareMatrix mat_b){
    squareMatrix mat_results;
        
    clock_t before = clock();
    mat_results = multiplyMatrices(mat_a, mat_b);
    printTime(clock() - before, mat_results.dimension, (char*)"CPU");
}

__host__ void testDevicePreformance(squareMatrix mat_a, squareMatrix mat_b){
    if (mat_a.dimension != mat_b.dimension) exit(1);
    int allocationsize = mat_a.dimension * mat_b.dimension * sizeof(int);

    squareMatrix mat_results = {(int*)malloc(allocationsize), mat_a.dimension};

    int* dev_mat_a, *dev_mat_b, *dev_mat_results;

    clock_t before = clock();

    gpuErrchk(cudaMalloc((void **)&dev_mat_a,          allocationsize));
    gpuErrchk(cudaMalloc((void **)&dev_mat_b,          allocationsize));
    gpuErrchk(cudaMalloc((void **)&dev_mat_results,    allocationsize));
    
    gpuErrchk(cudaMemcpy(dev_mat_a, mat_a.elements, allocationsize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_mat_b, mat_b.elements, allocationsize, cudaMemcpyHostToDevice));

    dim3 dimBlock(mat_a.dimension, mat_a.dimension);
    dim3 dimGrid(mat_a.dimension, mat_a.dimension);
    multiplyMatricesParallel<<<dimGrid, dimBlock>>> (dev_mat_a, dev_mat_b, dev_mat_results, mat_a.dimension);

    gpuErrchk(cudaMemcpy(mat_results.elements, dev_mat_results, allocationsize, cudaMemcpyDeviceToHost));
    printTime(clock() - before, mat_results.dimension, "GPU");
    
    cudaFree(dev_mat_a); cudaFree(dev_mat_b); cudaFree(dev_mat_results);
    free(mat_results.elements);
    //printSquareMatrix(mat_results);
}

__host__ void testMatrixMultiplicationPreformance(int dimension, int computeDev){

    squareMatrix mat_a, mat_b;
    
    mat_a = createRandomSquareMatrix(dimension);
    mat_b = createRandomSquareMatrix(dimension);

    //printSquareMatrix(mat_a);
    //printSquareMatrix(mat_b);

    if (computeDev != dev_gpu) testHostPreformance(mat_a, mat_b);
    if (computeDev != dev_cpu) testDevicePreformance(mat_a, mat_b);

    free(mat_a.elements);
    free(mat_b.elements);
}

int main(int argc, char** argv) {

    comptuationDevice computeDev = dev_both; 

    for (int i = 0; i < argc; i++){
        if (strcmp(argv[i],  "--device=gpu")==0)  computeDev = dev_gpu;
        if (strcmp(argv[i],  "--device=cpu")==0)  computeDev = dev_cpu;
    }

    printf("Device Config: %d", computeDev);

    for (int i = 16; i < 8193*10; i*=2 )
        testMatrixMultiplicationPreformance(i, computeDev);

	return 0;
}