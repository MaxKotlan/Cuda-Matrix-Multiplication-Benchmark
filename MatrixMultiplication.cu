#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void kernel(void){

}

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

int main(void) {

    squareMatrix mat_a = createRandomSquareMatrix(10);
    squareMatrix mat_b = createRandomSquareMatrix(10);

    printSquareMatrix(mat_a);
    printSquareMatrix(mat_b);

    squareMatrix mat_c = multiplyMatrices(mat_a, mat_b);
    printSquareMatrix(mat_c);

    kernel<<<1,1>>> ();
    //printf("Hello World\n");
	return 0;
}