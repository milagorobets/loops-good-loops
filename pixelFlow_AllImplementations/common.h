/*
 * common.h
 *
 *  Created on: 2015-02-20
 *      Author: cinnamon
 */

#ifndef COMMON_H_
#define COMMON_H_

#define NUM_WALL_BLOCKS 5

#define NUM_CPU_R 10

#define MATRIX_DIM 5

// choose mode
// CPU_UNOPTIMIZED: first implementation, uses new/delete to allocate memory
// CPU_VECTOR: version of CPU_UNOPTIMIZED with new/delete replaced by <vector>
// CPU_VECTOR_1D: version of CPU_VECTOR but <vector>s are 1D instead of 2D
// CPU_OPT_NEW_DELETE: optimized implementation with new/delete
// CPU_OPT_NEW_DELETE_1D: optimized implementation with new/delete, matrix unrolled
// CPU_OPT_MALLOC: optimized CPU implementation with malloc/free
// GPU_UNOPTIMIZED: code stupidly thrown at the GPU
// GPU_DUAL_BUFFER: double-buffer computation and memory transfer for the GPU
#define TEST_SPACE CPU_UNOPTIMIZED

#define CHECK_OUTPUT 1 // Compare output to known working version to make sure it is still correct


#endif /* COMMON_H_ */
