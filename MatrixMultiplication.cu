#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void kernel(void){

}

struct squareMatrix{
    int* elements;
    int dimension;
}

__host__ squareMatrix createRandomSquareMatrix(int dimension){
    int  mat_elements = dimension * dimension;
    int* mat = (int*)malloc(sizeof(int)*mat_elements);
    for (int i = 0; i < mat_elements; i++)
        mat[i] = rand()%10;
    return mat;
}

__host__ void printSquareMatrix(int* mat, int dimension){
    for (int i = 0; i < 100; i++){
        if (i % dimension == 0 && i != 0) printf("\n");
        printf("%d ", mat[i]);
    }
}

int main(void) {

    int* mat_10 = createRandomSquareMatrix(10);
    
    printSquareMatrix(mat_10, 10);

    kernel<<<1,1>>> ();
    //printf("Hello World\n");
	return 0;
}