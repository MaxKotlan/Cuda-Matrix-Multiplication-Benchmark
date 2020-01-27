
# GPU Results

The device I used was a Geforce GTX 980. I initialized with the startup parameters  `--seed 1234 --random_mod 2 --block_thread 256`. I used the same seed value (1234) as my cpu test to ensure each device was computing the same matrices. I used 256 threads per block because it was a multiple of the number of cuda cores the Geforce GTX 980 has ( 2048 Cuda Cores ). The largest calculation I could compute on this card was 17179 x 17179.  A total of  3541416492 bytes of VRAM was used for this computation. VRAM was the limiting scaling resource in this design. 
```
Multiplying 2x2 X 2x2 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 0 msec!
        Copying A, B To VRAM...                          Done in 0 msec!
        Computing and Copying Result...                  Done in 0 msec!
        Deallocating Result Matrix...                    Done in 0 msec!

Total Time: Done in 2 msec!
--------------------------------------------------------------------------------------------
Multiplying 4x4 X 4x4 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 0 msec!
        Copying A, B To VRAM...                          Done in 0 msec!
        Computing and Copying Result...                  Done in 0 msec!
        Deallocating Result Matrix...                    Done in 1 msec!

Total Time: Done in 2 msec!
--------------------------------------------------------------------------------------------
Multiplying 8x8 X 8x8 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 0 msec!
        Copying A, B To VRAM...                          Done in 0 msec!
        Computing and Copying Result...                  Done in 0 msec!
        Deallocating Result Matrix...                    Done in 1 msec!

Total Time: Done in 1 msec!
--------------------------------------------------------------------------------------------
Multiplying 16x16 X 16x16 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 0 msec!
        Copying A, B To VRAM...                          Done in 0 msec!
        Computing and Copying Result...                  Done in 0 msec!
        Deallocating Result Matrix...                    Done in 0 msec!

Total Time: Done in 1 msec!
--------------------------------------------------------------------------------------------
Multiplying 32x32 X 32x32 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 1 msec!
        Copying A, B To VRAM...                          Done in 0 msec!
        Computing and Copying Result...                  Done in 0 msec!
        Deallocating Result Matrix...                    Done in 0 msec!

Total Time: Done in 1 msec!
--------------------------------------------------------------------------------------------
Multiplying 64x64 X 64x64 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 1 msec!
        Copying A, B To VRAM...                          Done in 0 msec!
        Computing and Copying Result...                  Done in 0 msec!
        Deallocating Result Matrix...                    Done in 0 msec!

Total Time: Done in 10 msec!
--------------------------------------------------------------------------------------------
Multiplying 128x128 X 128x128 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 0 msec!
        Copying A, B To VRAM...                          Done in 0 msec!
        Computing and Copying Result...                  Done in 0 msec!
        Deallocating Result Matrix...                    Done in 0 msec!

Total Time: Done in 3 msec!
--------------------------------------------------------------------------------------------
Multiplying 256x256 X 256x256 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 1 msec!
        Copying A, B To VRAM...                          Done in 1 msec!
        Computing and Copying Result...                  Done in 1 msec!
        Deallocating Result Matrix...                    Done in 1 msec!

Total Time: Done in 4 msec!
--------------------------------------------------------------------------------------------
Multiplying 512x512 X 512x512 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 2 msec!
        Copying A, B To VRAM...                          Done in 1 msec!
        Computing and Copying Result...                  Done in 3 msec!
        Deallocating Result Matrix...                    Done in 1 msec!

Total Time: Done in 11 msec!
--------------------------------------------------------------------------------------------
Multiplying 1024x1024 X 1024x1024 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 2 msec!
        Copying A, B To VRAM...                          Done in 6 msec!
        Computing and Copying Result...                  Done in 20 msec!
        Deallocating Result Matrix...                    Done in 2 msec!

Total Time: Done in 35 msec!
--------------------------------------------------------------------------------------------
Multiplying 2048x2048 X 2048x2048 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 6 msec!
        Copying A, B To VRAM...                          Done in 25 msec!
        Computing and Copying Result...                  Done in 131 msec!
        Deallocating Result Matrix...                    Done in 2 msec!

Total Time: Done in 172 msec!
--------------------------------------------------------------------------------------------
Multiplying 4096x4096 X 4096x4096 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 18 msec!
        Copying A, B To VRAM...                          Done in 101 msec!
        Computing and Copying Result...                  Done in 1680 msec!
        Deallocating Result Matrix...                    Done in 6 msec!

Total Time: Done in 1813 msec!
--------------------------------------------------------------------------------------------
Multiplying 8192x8192 X 8192x8192 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 0 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 62 msec!
        Copying A, B To VRAM...                          Done in 412 msec!
        Computing and Copying Result...                  Done in 13487 msec!
        Deallocating Result Matrix...                    Done in 37 msec!

Total Time: Done in 14006 msec!
--------------------------------------------------------------------------------------------
Multiplying 16384x16384 X 16384x16384 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 2 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 241 msec!
        Copying A, B To VRAM...                          Done in 1638 msec!
        Computing and Copying Result...                  Done in 106866 msec!
        Deallocating Result Matrix...                    Done in 122 msec!

Total Time: Done in 108875 msec!
--------------------------------------------------------------------------------------------
Multiplying 17179x17179 X 17179x17179 using the GPU ...

        Allocating Result Matrix To RAM...               Done in 2 msec!
        Allocating A, B, and RESULT Matrix To VRAM...    Done in 262 msec!
        Copying A, B To VRAM...                          Done in 1798 msec!
        Computing and Copying Result...                  Done in 141119 msec!
        Deallocating Result Matrix...                    Done in 136 msec!

Total Time: Done in 143318 msec!
```