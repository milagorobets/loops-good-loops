#ifndef COMMON_H_
#define COMMON_H_

//---- NETWORK PARAMETERS:
#define NUM_WALL_BLOCKS 5	// Number of wall blocks (used in the RNG version)
#define MATRIX_DIM 1024 // Grid dimension (NUMEL is MATRIX_DIM*MATRIX_DIM)
#define SRC_MAG 0.01 //3.126
#define REDRAW_LOOP 100

#define LIGHT_SPEED		300000000 // C
//#define CT				(LIGHT_SPEED/1.414214) // C/sqrt(2)
#define CT				(LIGHT_SPEED/2) // C/sqrt(2)
#define CUTOFF_FREQ		(CT/DELTA_LENGTH/4)
#define REC_MAX_FREQ	(CT/DELTA_LENGTH/10)
#define DELTA_LENGTH	0.01
#define SRC_FREQ		1500000000
#define WALL_DEC_PCM	0.85 // per 10cm thickness
#define WALL_DEC		WALL_DEC_PCM

#define SAMPLES_TO_AVERAGE (5 * 20) //(5*CT/SRC_FREQ) //(5*CT/REC_MAX_FREQ) //200
#define SAMPLING 20 //(CT/SRC_FREQ) //(1*CT/REC_MAX_FREQ) //20


//#if (SRC_FREQ>REC_MAX_FREQ)
//#warning("source frequency too high!")
//#endif

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

