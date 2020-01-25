#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

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
        mat[i] = rand()%2;
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
    
}

__host__ void printTime(clock_t totaltime, int dimension){
    int msec = totaltime * 1000 / CLOCKS_PER_SEC;
    printf("Multiplying %dx%d X %dx%d took %d msec using the CPU\n--------------------------------------------------------------------------------------------\n", 
    dimension, dimension, dimension, dimension, msec);
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

__host__ void testMatrixMultiplicationPreformance(int dimension){

    squareMatrix        mat_a,        mat_b,        mat_results;
    squareMatrix device_mat_a, device_mat_b, device_mat_results;
    
    mat_a = createRandomSquareMatrix(dimension);
    mat_b = createRandomSquareMatrix(dimension);
    
    clock_t before = clock();
    mat_results = multiplyMatrices(mat_a, mat_b);
    printTime(clock() - before, mat_results.dimension);


    gpuErrchk(cudaMalloc(&device_mat_a.elements, dimension*dimension*sizeof(int)));
    gpuErrchk(cudaMalloc(&device_mat_b.elements, dimension*dimension*sizeof(int)));
    gpuErrchk(cudaMalloc(&device_mat_results.elements, dimension*dimension*sizeof(int)));
    
    gpuErrchk(cudaMemcpy(device_mat_a.elements, &mat_a.elements, dimension*dimension*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_mat_a.elements, &mat_b.elements, dimension*dimension*sizeof(int), cudaMemcpyHostToDevice));

    multiplyMatricesParallel<<<1,1024>>> (device_mat_a.elements, device_mat_b.elements, device_mat_results.elements, dimension);

    free(mat_results.elements);
    gpuErrchk(cudaMemcpy(&device_mat_results.elements, mat_results.elements, dimension*dimension*sizeof(int), cudaMemcpyDeviceToHost));


    free(mat_a.elements);
    free(mat_b.elements);
    free(mat_results.elements);
}


int main(void) {

    for (int i = 16; i < 1025; i*=2 )
        testMatrixMultiplicationPreformance(i);

	return 0;
}