#ifndef COMMON_H_
#define COMMON_H_

//---- NETWORK PARAMETERS:
#define NUM_WALL_BLOCKS 5	// Number of wall blocks (used in the RNG version)
#define MATRIX_DIM 512// Grid dimension (NUMEL is MATRIX_DIM*MATRIX_DIM)
#define SRC_MAG 3.126
#define REDRAW_LOOP 100
#define SAMPLING 100
#define SAMPLES_TO_AVERAGE SAMPLING

//---- SIMULATION PARAMETERS:
#define NUM_CPU_R 1 // Number of iterations

// choose mode:
// CPU_UNOPTIMIZED: first implementation, uses new/delete to allocate memory
// CPU_VECTOR: version of CPU_UNOPTIMIZED with new/delete replaced by <vector>
// CPU_VECTOR_1D: version of CPU_VECTOR but <vector>s are 1D instead of 2D
// CPU_OPT_NEW_DELETE: optimized implementation with new/delete
// CPU_OPT_NEW_DELETE_1D: optimized implementation with new/delete, matrix unrolled
// CPU_OPT_MALLOC: optimized CPU implementation with malloc/free
// GPU_UNOPTIMIZED: code stupidly thrown at the GPU
// GPU_DUAL_BUFFER: double-buffer computation and memory transfer for the GPU

#define TEST_CLASS GPU_PTR 
#define REFERENCE_CLASS CPU_REFERENCE // Don't change this?

#define CHECK_OUTPUT 0			// Compare output to known working version to make sure it is still correct
#define CHECK_TOLERANCE 0.000001	// Tolerance for comparing floats

//---- SETUP PARAMETERS (don't touch unless you know why):
#define PI 3.141592653589	// For god's sake, don't change Pi

#define WALL_MEMORY MEM_MAP // Determines where to put the wall matrix
#define MEM_STACK 1			// Stack (doesn't work for large matrices)
#define MEM_MAP 0			// Map because it's so sparse (slower than stack)
#define MEM_HEAP 2			// Put it on the heap!

#if (WALL_MEMORY == MEM_STACK)
//#warning("Remember to increase the stack allocation!")
#endif

#endif /* COMMON_H_ */

