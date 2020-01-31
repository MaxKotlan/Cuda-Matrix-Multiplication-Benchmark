# Cuda Matrtix Multiplication Benchmark

This program preforms matrix multiplication on various sized square matrices using the standard ![enter image description here](https://latex.codecogs.com/gif.latex?O%28n%5E3%29) approach. It does not use other more efficient algorithms, such as the [*Strassen algorithm*](https://en.wikipedia.org/wiki/Strassen_algorithm) or the [*Coppersmith-Winograd*](https://en.wikipedia.org/wiki/Coppersmith%E2%80%93Winograd_algorithm)

### Building

To build you need to have have the NVCC compiler which is installed with [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads) and make. If you're on windows and you do not have make installed, you can build by running

`nvcc -o MatrixMultiplication MatrixMultiplication.cu`

### Long Cuda Computations on Windows

  

By default, the *Windows Display Driver Model* (WDDM) has [*Timeout Detection and Recovery* (TDR)](https://docs.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys) enabled to prevent kernels from hanging. If a kernel takes too long to process ( default is 2 seconds or more ), it will be killed by the system drivers. This program requires TDR to be disabled, which can be disabled by [Editing the Registry Manually](https://docs.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys), or by using [NVIDIA NSIGHT](https://docs.nvidia.com/gameworks/content/developertools/desktop/timeout_detection_recovery.htm).

Linux should work out of the box. No driver modification needed.

 ### Startup Parameters

The following are the available command-line arguments

```
 Shows what parameters are available
        --help

 Selects which device should be used:
        --device cpu
        --device gpu
        --device both

 sets seedvalue for random number generation (default: currentTime)
        --seed [int]

 sets mod value for random number generation (default: 2)
        --random_mod [int]

 sets max dimension to compute (default: max matrix that can fit in vram)
        --max_dimension [int]

 sets starting matrix dimension (default: 2)
        --start_dimension [int]

 only computes a single matrix of n size.
        --only [int]

 sets number of threads per block (default: 256). Should be a multiple of cuda cores
        --block_threads [int 1-1024]

 outputs matrix a, b and result. (not reccomented for extremely large matrices)
        --mat_print                    (prints to the console)
        --mat_save [filepath]          (saves to disk. filepath optional)
```

### PERFORMANCE RESULTS
[MY GPU RESULTS](GPU-RESULTS.md)

[MY CPU RESULTS](CPU-RESULTS.md)