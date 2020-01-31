#include <stdio.h>
#include <cuda.h>
//#include <cuda_runtime.h>
//#include <curand_kernel.h>
#include <string.h>
#include <time.h>

enum comptuationDevice{
    dev_both,
    dev_gpu,
    dev_cpu
};

/*Global Startup Settings*/
struct Startup{
    comptuationDevice device = dev_gpu;
    int randomMod = 2;
    int seedValue = time(nullptr);
    int maxDimension = INT_MAX;
    int startDimension = 2;
    int threadsPerBlock = 256;
    int onlyMatrixSize = NULL;
    char* outputDirectory = ".";
    bool matSave  = false;
    bool matPrint = false;
} startup;

/*Matrix Datastructure*/
struct squareMatrix{
    int* elements;
    int dimension;
};

__host__ squareMatrix createRandomSquareMatrix(int dimension){
    int  mat_elements = dimension * dimension;
    int* mat = (int*)malloc(sizeof(int)*mat_elements);
    for (int i = 0; i < mat_elements; i++)
        mat[i] = rand()%startup.randomMod;
    return {mat, dimension};
}

__host__ void printSquareMatrix(squareMatrix mat){
    for (int i = 0; i < mat.dimension*mat.dimension; i++){
        if (i % mat.dimension == 0 && i != 0) printf("\n");
        printf("%d ", mat.elements[i]);
    }
    printf("\n\n");
}

__host__ void multiplyMatrices(squareMatrix a, squareMatrix b, squareMatrix result){
    if (a.dimension != b.dimension) exit(1);
    
    for (int i = 0; i < result.dimension; i++){
        for (int j = 0; j < result.dimension; j++){
            result.elements[j + result.dimension*i] = 0;
            for (int k = 0; k < result.dimension; k++)
                result.elements[j + result.dimension*i] += a.elements[k + result.dimension*i] * b.elements[j + result.dimension*k];
        }
    }
}

__global__ void multiplyMatricesParallel(int* mat_a, int* mat_b, int* mat_results, int dimension){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < dimension*dimension){
        mat_results[idx] = 0;
        for (int i = 0; i < dimension; i++){
            int row = idx / dimension;
            mat_results[idx] += mat_a[i +  dimension*row] * mat_b[idx%dimension + dimension*i];
        }
    }
}

__host__ void printMatrixInfo(int dimension, char* type){
    printf("--------------------------------------------------------------------------------------------\nMultiplying %dx%d X %dx%d using the %s ...\n\n", 
    dimension, dimension, dimension, dimension, type);
}

__host__ void saveMatrixToFile(squareMatrix mat_sav, char* label){

    char fileNameBuffer[256];

    char dim[30];
    itoa(mat_sav.dimension,dim,10);

    snprintf(fileNameBuffer, sizeof fileNameBuffer, "%s/%sx%s_X_%sx%s_%s", startup.outputDirectory, dim, dim, dim, dim, label, ".txt");
    
    FILE* fp = fopen( fileNameBuffer, "w");
    if (fp == nullptr) printf("Could not log to file\n");
    else {
        for (int i = 0; i < mat_sav.dimension*mat_sav.dimension; i++){
            if (i % mat_sav.dimension == 0 && i != 0) fprintf(fp, "\n");
            fprintf(fp, "%d ", mat_sav.elements[i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}


__host__ void printTime(clock_t totaltime){
    int msec = totaltime * 1000 / CLOCKS_PER_SEC;
    printf("Done in %d msec!\n", msec);
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
    printMatrixInfo(mat_a.dimension, (char*)"CPU");
    clock_t initalTime = clock();

    printf("\tAllocating Result Matrix To Ram...               ");
    clock_t before = clock();
    squareMatrix mat_results = {(int*)malloc(sizeof(int)*mat_a.dimension*mat_a.dimension), mat_a.dimension};
    printTime(clock() - before);

    before = clock();
    printf("\tPreforming Multiplication...                     ");
    multiplyMatrices(mat_a, mat_b, mat_results);
    printTime(clock() - before);

    printf("\tDeallocating Result Matrix From Ram...           ");
    before = clock();
    free(mat_results.elements);
    printTime(clock() - before);

    printf("\nTotal Time:                                      ");
    printTime(clock() - initalTime);
}

__host__ void testDevicePreformance(squareMatrix mat_a, squareMatrix mat_b){
    printMatrixInfo(mat_a.dimension, (char*)"GPU");
    clock_t initalTime = clock();

    if (mat_a.dimension != mat_b.dimension) exit(1);
    
    int allocationsize = mat_a.dimension * mat_b.dimension * sizeof(int);

    printf("\tAllocating Result Matrix To RAM...               ");
    clock_t before = clock();
        squareMatrix mat_results = {(int*)malloc(allocationsize), mat_a.dimension};
    printTime(clock() - before);

    int* dev_mat_a, *dev_mat_b, *dev_mat_results;

    printf("\tAllocating A, B, and RESULT Matrix To VRAM...    ");
    before = clock();
        gpuErrchk(cudaMalloc((void **)&dev_mat_a,          allocationsize));
        gpuErrchk(cudaMalloc((void **)&dev_mat_b,          allocationsize));
        gpuErrchk(cudaMalloc((void **)&dev_mat_results,    allocationsize));
    printTime(clock() - before);

    printf("\tCopying A, B To VRAM...                          ");
    before = clock();
        gpuErrchk(cudaMemcpy(dev_mat_a, mat_a.elements, allocationsize, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_mat_b, mat_b.elements, allocationsize, cudaMemcpyHostToDevice));
    printTime(clock() - before);

    printf("\tComputing and Copying Result...                  ");
    before = clock();
        int totalThreadsNeeded = mat_a.dimension*mat_a.dimension;
        multiplyMatricesParallel<<<totalThreadsNeeded / startup.threadsPerBlock + 1, startup.threadsPerBlock>>> (dev_mat_a, dev_mat_b, dev_mat_results, mat_a.dimension);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaMemcpy(mat_results.elements, dev_mat_results, allocationsize, cudaMemcpyDeviceToHost));
    printTime(clock() - before);


    if (startup.matPrint) printSquareMatrix(mat_results);
    if (startup.matSave)  {
        printf("\tSaving Result Matrix to Disk...                  ");
        before = clock();
        saveMatrixToFile(mat_results, "matrix_result");
        printTime(clock() - before);
    }

    printf("\tDeallocating Result Matrix...                    ");
    before = clock();
        cudaFree(dev_mat_a); cudaFree(dev_mat_b); cudaFree(dev_mat_results);
        free(mat_results.elements);
    printTime(clock() - before);

    printf("\nTotal Time: ");
    printTime(clock() - initalTime);
}

__host__ void testMatrixMultiplicationPreformance(int dimension){

    srand(startup.seedValue);

    squareMatrix mat_a, mat_b;
    
    mat_a = createRandomSquareMatrix(dimension);
    mat_b = createRandomSquareMatrix(dimension);

    if (startup.matPrint) {
        printSquareMatrix(mat_a);
        printSquareMatrix(mat_b);
    }

    if (startup.matSave) {
        saveMatrixToFile(mat_a, "matrix_A");
        saveMatrixToFile(mat_b, "matrix_B");
    }

    if (startup.device != dev_cpu) testDevicePreformance(mat_a, mat_b);
    if (startup.device != dev_gpu) testHostPreformance(mat_a, mat_b);

    free(mat_a.elements);
    free(mat_b.elements);
}

unsigned int getFreeGpuMem()
{
    size_t free_t;
    cudaMemGetInfo(&free_t,nullptr);    
    return (unsigned int)free_t;
}

__host__ unsigned int calculateLargestPossibleMatrixDimension(){
    unsigned int free = getFreeGpuMem();
    unsigned int memoryPerMatrix = free  / (sizeof(int) * 3);
    unsigned int maxMatrixDimension = sqrt( memoryPerMatrix ) - 1;
    return maxMatrixDimension;
}

int main(int argc, char** argv) {

    for (int i = 0; i < argc; i++){
        if (strcmp(argv[i],  "--device")==0 && i+1 < argc) 
            if (strcmp(argv[i+1], "gpu") == 0) startup.device = dev_gpu;
            else if (strcmp(argv[i+1], "cpu") == 0) startup.device = dev_cpu;
            else if (strcmp(argv[i+1], "both") == 0) startup.device = dev_both;
        if (strcmp(argv[i],  "--random_mod")==0 && i+1 < argc) startup.randomMod = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--max_dimension")==0 && i+1 < argc) startup.maxDimension = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--seed")==0 && i+1 < argc) startup.seedValue = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--start_dimension")==0 && i+1 < argc) startup.startDimension = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--only")==0 && i+1 < argc) startup.onlyMatrixSize = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--block_threads")==0 && i+1 < argc) startup.threadsPerBlock = atoi(argv[i+1]);
        if (strcmp(argv[i],  "--mat_print")==0) startup.matPrint = true;
        if (strcmp(argv[i],  "--mat_save")==0){ 
            startup.matSave = true;
            if   (i+1 < argc && strstr(argv[i+1], "--") == NULL) startup.outputDirectory = argv[i+1];
        }
    }

    /*Tests only one matrix if parameter passed in*/
    if (startup.onlyMatrixSize != NULL)
        testMatrixMultiplicationPreformance(startup.onlyMatrixSize);

    /*Otherwise, double matrix size until vram is completely filled*/
    else { 
        unsigned int maxMatrixDimension = calculateLargestPossibleMatrixDimension();
        for (int i = startup.startDimension; i != maxMatrixDimension*2 && i <= startup.maxDimension; i*=2 ) {
            maxMatrixDimension = calculateLargestPossibleMatrixDimension();
            if (i > maxMatrixDimension)
                i = maxMatrixDimension;
            testMatrixMultiplicationPreformance(i);
        }
    }
    
	return 0;
}