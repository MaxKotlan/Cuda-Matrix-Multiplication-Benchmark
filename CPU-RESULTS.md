
# CPU Results

The CPU I used was an Intel i5-3570k. I initialized with the startup parameters `--seed 1234 --random_mod 2 --device cpu`. I used the same seed value (1234) as my GPU test to ensure each device was computing the same matrices. Overall the computations took much longer than my GPU. I was only able to calculate up to an 8192 x 8192 matrix. I left my computer running for over 6 hours, and was unable to complete the 16384 x 16384 computation that completed in approximately 1 minute and 48 seconds using my GPU. 

 The 8192 x 8192  matrix completed in 497622 milliseconds (approx 8 minutes and 17 seconds), whereas on my GPU, an 8192 x 8192 matrix completed in 14006 milliseconds (approx 14 seconds). 
 
```
--------------------------------------------------------------------------------------------
Multiplying 2x2 X 2x2 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 0 msec!
        Deallocating Result Matrix From Ram...           Done in 0 msec!

Total Time:                                      Done in 1 msec!
--------------------------------------------------------------------------------------------
Multiplying 4x4 X 4x4 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 0 msec!
        Deallocating Result Matrix From Ram...           Done in 0 msec!

Total Time:                                      Done in 0 msec!
--------------------------------------------------------------------------------------------
Multiplying 8x8 X 8x8 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 0 msec!
        Deallocating Result Matrix From Ram...           Done in 0 msec!

Total Time:                                      Done in 0 msec!
--------------------------------------------------------------------------------------------
Multiplying 16x16 X 16x16 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 0 msec!
        Deallocating Result Matrix From Ram...           Done in 0 msec!

Total Time:                                      Done in 0 msec!
--------------------------------------------------------------------------------------------
Multiplying 32x32 X 32x32 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 0 msec!
        Deallocating Result Matrix From Ram...           Done in 0 msec!

Total Time:                                      Done in 0 msec!
--------------------------------------------------------------------------------------------
Multiplying 64x64 X 64x64 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 1 msec!
        Deallocating Result Matrix From Ram...           Done in 0 msec!

Total Time:                                      Done in 1 msec!
--------------------------------------------------------------------------------------------
Multiplying 128x128 X 128x128 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 12 msec!
        Deallocating Result Matrix From Ram...           Done in 0 msec!

Total Time:                                      Done in 12 msec!
--------------------------------------------------------------------------------------------
Multiplying 256x256 X 256x256 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 95 msec!
        Deallocating Result Matrix From Ram...           Done in 0 msec!

Total Time:                                      Done in 97 msec!
--------------------------------------------------------------------------------------------
Multiplying 512x512 X 512x512 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 997 msec!
        Deallocating Result Matrix From Ram...           Done in 1 msec!

Total Time:                                      Done in 1000 msec!
--------------------------------------------------------------------------------------------
Multiplying 1024x1024 X 1024x1024 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 15745 msec!
        Deallocating Result Matrix From Ram...           Done in 1 msec!

Total Time:                                      Done in 15750 msec!
--------------------------------------------------------------------------------------------
Multiplying 2048x2048 X 2048x2048 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 229734 msec!
        Deallocating Result Matrix From Ram...           Done in 2 msec!

Total Time:                                      Done in 229740 msec!
--------------------------------------------------------------------------------------------
Multiplying 4096x4096 X 4096x4096 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 718858 msec!
        Deallocating Result Matrix From Ram...           Done in 6 msec!

Total Time:                                      Done in 718867 msec!
--------------------------------------------------------------------------------------------
Multiplying 8192x8192 X 8192x8192 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 0 msec!
        Preforming Multiplication...                     Done in 497607 msec!
        Deallocating Result Matrix From Ram...           Done in 10 msec!

Total Time:                                      Done in 497622 msec!
--------------------------------------------------------------------------------------------
Multiplying 16384x16384 X 16384x16384 using the CPU ...

        Allocating Result Matrix To Ram...               Done in 2 msec!
        Preforming Multiplication...                     
```