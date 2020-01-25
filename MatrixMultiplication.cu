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

    int* dev_elements; //pointer to elements copied over to the gpu
};

__host__ void freeMemory(squareMatrix mat){
    free(mat.elements);
    cudaFree(mat.dev_elements);
};

__host__ squareMatrix createRandomSquareMatrix(int dimension){
    int  mat_elements = dimension * dimension;
    int* mat = (int*)malloc(sizeof(int)*mat_elements);
    for (int i = 0; i < mat_elements; i++)
        mat[i] = rand()%2;
    return {mat, dimension, nullptr};
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
    squareMatrix result = { (int*)malloc(sizeof(int)*a.dimension*a.dimension), a.dimension, nullptr };
    
    for (int i = 0; i < result.dimension; i++){
        for (int j = 0; j < result.dimension; j++){
            result.elements[j + result.dimension*i] = 0;
            for (int k = 0; k < result.dimension; k++)
                result.elements[j + result.dimension*i] += a.elements[k + result.dimension*i] * b.elements[j + result.dimension*k];
        }
    }
    return result;
}

__global__ void multiplyMatricesParallel(void){
    
}

__host__ void testMatrixMultiplicationPreformance(int dimension){
    /*Create Two Random NxN Matrices*/
    squareMatrix mat_a = createRandomSquareMatrix(dimension);
    squareMatrix mat_b = createRandomSquareMatrix(dimension);
    
    clock_t before = clock();
    squareMatrix mat_c = multiplyMatrices(mat_a, mat_b);
    clock_t totaltime = clock() - before;
    int msec = totaltime * 1000 / CLOCKS_PER_SEC;
    printf("Multiplying %dx%d X %dx%d took %d msec using the CPU\n--------------------------------------------------------------------------------------------\n", 
    dimension, dimension, dimension, dimension, msec);

    cudaMalloc((void**)&mat_a.dev_elements, mat_a.dimension*mat_a.dimension*sizeof(int));
    cudaMalloc((void**)&mat_b.dev_elements, mat_b.dimension*mat_b.dimension*sizeof(int));

    cudaMemcpy(mat_a.elements, mat_a.dev_elements, mat_a.dimension*mat_a.dimension*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_b.elements, mat_b.dev_elements, mat_b.dimension*mat_b.dimension*sizeof(int), cudaMemcpyHostToDevice);

    multiplyMatricesParallel<<<1,1>>> ();

    freeMemory(mat_a);
    freeMemory(mat_b);
    freeMemory(mat_c);
}


int main(void) {

    for (int i = 16; i < 1025; i*=2 )
        testMatrixMultiplicationPreformance(i);

	return 0;
}