// TO DO:
// 1. coalesce memory accesses for m and nm
// 2. put FP_ptr_copy into pointer form
// 3. split kernels into edges and middle blocks (middle do not if checks in flow) -- started

#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cPFkernel_ptr.cuh"
#include "utils.h"
#include "common.h"

#define SoF sizeof(float)
#define CI(x,y,z,width,height) ((x) + (y)*(width) + (z) * (height) * (width))

#define WWAL_DIMx 4
#define WWAL_DIMy WWAL_DIMx
#define W_DIMx 4
#define W_DIMy W_DIMx

#define THREADx 16
#define THREADy 16
#define BLOCK_DIMx ((MATRIX_DIM>THREADx)?THREADx:MATRIX_DIM) // vary this
#define BLOCK_DIMy ((MATRIX_DIM>THREADy)?THREADy:MATRIX_DIM)
#define GRID_DIMx ((MATRIX_DIM + BLOCK_DIMx - 1)/BLOCK_DIMx)
#define GRID_DIMy ((MATRIX_DIM + BLOCK_DIMy - 1)/BLOCK_DIMy)

bool * host_Wall;
float * host_WWall;
float * host_W;

double coef = 1.0;

int gpu_iterations;

float *m_host;

double src_amplitude;
double src_frequency;
dim3 src_loc;

texture<float,2,cudaReadModeElementType> tex_m0;
texture<float,2,cudaReadModeElementType> tex_m1; 
texture<float,2,cudaReadModeElementType> tex_m2;
texture<float,2,cudaReadModeElementType> tex_m3;
texture<float,2,cudaReadModeElementType> tex_nm0;
texture<float,2,cudaReadModeElementType> tex_nm1;
texture<float,2,cudaReadModeElementType> tex_nm2;
texture<float,2,cudaReadModeElementType> tex_nm3;
texture<float,2,cudaReadModeElementType> tex_WWall;
texture<float,2,cudaReadModeElementType> tex_W;

__global__ void PF_ptr_copy(cudaPitchedPtr mPtr, cudaPitchedPtr nmPtr, cudaExtent mExt, dim3 matdim)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float *m = (float*)mPtr.ptr;
	float *nm = (float*)nmPtr.ptr;
	size_t pitch = mPtr.pitch;
	unsigned int e_per_row = pitch / SoF;
	size_t slice_pitch = pitch*mExt.height;

	if ((x < MATRIX_DIM) && (y < MATRIX_DIM))
	{
		m[CI(x, y, 0, e_per_row, matdim.y)] = nm[CI(x, y, 0, e_per_row, matdim.y)];
		m[CI(x, y, 1, e_per_row, matdim.y)] = nm[CI(x, y, 1, e_per_row, matdim.y)];
		m[CI(x, y, 2, e_per_row, matdim.y)] = nm[CI(x, y, 2, e_per_row, matdim.y)];
		m[CI(x, y, 3, e_per_row, matdim.y)] = nm[CI(x, y, 3, e_per_row, matdim.y)];

		//__syncthreads();

		// Edge Cases
		if (x == 0)
		{
			if (nm[CI(0, y, 0, e_per_row, matdim.y)] == 0)
			{
				m[CI(0, y, 0, e_per_row, matdim.y)] = nm[CI(1, y, 0, e_per_row, matdim.y)];
			}
		 }
		 if (x == MATRIX_DIM-1)
		 {
			if (nm[CI(MATRIX_DIM-1, y, 1, e_per_row, matdim.y)] == 0)
			{
				m[CI(MATRIX_DIM-1, y, 1, e_per_row, matdim.y)] = nm[CI(MATRIX_DIM-2, y, 1, e_per_row, matdim.y)];
			}
		 }
		 if (y == 0)
		 {
			if (nm[CI(x, 0, 2, e_per_row, matdim.y)] == 0)
			{
				m[CI(x, 0, 2, e_per_row, matdim.y)] = nm[CI(x, 1, 2, e_per_row, matdim.y)];
			}
		 }
		 if (y == MATRIX_DIM-1)
		 {
			if (nm[CI(x, MATRIX_DIM-1, 3, e_per_row, matdim.y)] == 0)
			{
				m[CI(x, MATRIX_DIM-1, 3, e_per_row, matdim.y)] = nm[CI(x, MATRIX_DIM-2, 3, e_per_row, matdim.y)];
			}
		 }
	}
}

__global__ void PF_padded_texture_copy(float*m0, float*m1, float*m2, float*m3, dim3 matdim, size_t pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int loc = x + y * pitch/sizeof(float);
	if ((x < MATRIX_DIM) && (y < MATRIX_DIM)) // Make sure cell is within the environment grid
	{
		float t0 = tex2D(tex_nm0, x+0.5f, y+0.5f); float t1 = tex2D(tex_nm1, x+0.5f, y+0.5f);
		float t2 = tex2D(tex_nm2, x+0.5f, y+0.5f); float t3 = tex2D(tex_nm3, x+0.5f, y+0.5f);

		// edge cases
		if ((x == 0) && (t0 == 0))
		{
			t0 = tex2D(tex_nm0, x+1.5f, y+0.5f);
		}
		if ((x == MATRIX_DIM-1) && (t1 == 0))
		{
			t1 = tex2D(tex_nm1, x-1.0f+0.5f, y+0.5f);
		}
		if ((y == 0) && (t2 == 0))
		{
			t2 = tex2D(tex_nm2, x+0.5f, y+1.0f+0.5f);
		}
		if ((y == MATRIX_DIM-1) && (t3 == 0))
		{
			t3 = tex2D(tex_nm3, x+0.5f, y-1.0f+0.5f);
		}

		// write values
		m0[loc] = t0;
		m1[loc] = t1;
		m2[loc] = t2;
		m3[loc] = t3;
	}	

}

__global__ void PF_padded_texture_flow(dim3 srcloc, float src, bool* wallLoc, float*nm0, float*nm1, float* nm2, float* nm3, dim3 matdim, float * WWall, float *W, size_t pitch)
{
	__shared__ float sWWall[16];
	__shared__ float sW[16];
	__shared__ float sMemM0[BLOCK_DIMx+2][BLOCK_DIMy+2]; 
	__shared__ float sMemM1[BLOCK_DIMx+2][BLOCK_DIMy+2];
	__shared__ float sMemM2[BLOCK_DIMx+2][BLOCK_DIMy+2];
	__shared__ float sMemM3[BLOCK_DIMx+2][BLOCK_DIMy+2];
	float x = threadIdx.x + blockIdx.x * blockDim.x;
	float y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int shX = threadIdx.x + 1;
	unsigned int shY = threadIdx.y + 1;
	int loc = x + y * pitch/sizeof(float);

	// copy coefficients to shared memory:
	if ((threadIdx.x < 4) && (threadIdx.y < 4))
	{
		sWWall[threadIdx.x + threadIdx.y * 4] = WWall[threadIdx.x + threadIdx.y * 4];
		sW[threadIdx.x + threadIdx.y * 4] = W[threadIdx.x + threadIdx.y * 4];
	}
	//__syncthreads();
	
	
	if ((x < MATRIX_DIM) && (y < MATRIX_DIM)) // Make sure cell is within the environment grid
	{
		sMemM0[shX][shY] = tex2D(tex_m0, x+0.5f, y+0.5f);
		sMemM1[shX][shY] = tex2D(tex_m1, x+0.5f, y+0.5f);
		sMemM2[shX][shY] = tex2D(tex_m2, x+0.5f, y+0.5f);
		sMemM3[shX][shY] = tex2D(tex_m3, x+0.5f, y+0.5f);
	
		// handle edges
		if (threadIdx.x == 0) // left
		{
			sMemM0[0][shY] = tex2D(tex_m0, x-0.5f, y); sMemM1[0][shY] = tex2D(tex_m1, x-0.5f, y);
			sMemM2[0][shY] = tex2D(tex_m2, x-0.5f, y); sMemM3[0][shY] = tex2D(tex_m3, x-0.5f, y);
		}
		else if (threadIdx.x == (BLOCK_DIMx - 1)) // right
		{
			sMemM0[BLOCK_DIMx+1][shY] = tex2D(tex_m0, x+1.5f, y); sMemM1[BLOCK_DIMx+1][shY] = tex2D(tex_m1, x+1.5f, y);
			sMemM2[BLOCK_DIMx+1][shY] = tex2D(tex_m2, x+1.5f, y); sMemM3[BLOCK_DIMx+1][shY] = tex2D(tex_m3, x+1.5f, y);
		}
		// MISSING THE CORNER BLOCK~ FIX IT

		if (threadIdx.y == 0) // up
		{
			sMemM0[shX][0] = tex2D(tex_m0, x, y-0.5f); sMemM1[shX][0] = tex2D(tex_m1, x, y-0.5f); 
			sMemM2[shX][0] = tex2D(tex_m2, x, y-0.5f); sMemM3[shX][0] = tex2D(tex_m3, x, y-0.5f); 
		}
		else if (threadIdx.y == (BLOCK_DIMy - 1)) // down
		{
			sMemM0[shX][BLOCK_DIMy+1] = tex2D(tex_m0, x, y+1.5f); sMemM1[shX][BLOCK_DIMy+1] = tex2D(tex_m1, x, y+1.5f);
			sMemM2[shX][BLOCK_DIMy+1] = tex2D(tex_m2, x, y+1.5f); sMemM3[shX][BLOCK_DIMy+1] = tex2D(tex_m3, x, y+1.5f);
		}
	}
	__syncthreads(); // sync the shared memory writes

	if ((x < MATRIX_DIM) && (y < MATRIX_DIM)) // Make sure cell is within the environment grid
	{
		// calculate nm 
		nm0[loc] = sW[4]*sMemM0[shX-1][shY] +	sW[5]*sMemM1[shX-1][shY] +	sW[6]*sMemM2[shX-1][shY] +	sW[7]*sMemM3[shX-1][shY];
		nm1[loc] = sW[0]*sMemM0[shX+1][shY] +	sW[1]*sMemM1[shX+1][shY] +	sW[2]*sMemM2[shX+1][shY] +	sW[3]*sMemM3[shX+1][shY];
		nm2[loc] = sW[12]*sMemM0[shX][shY-1] +	sW[13]*sMemM1[shX][shY-1] +	sW[14]*sMemM2[shX][shY-1] + sW[15]*sMemM3[shX][shY-1];
		nm3[loc] = sW[8]*sMemM0[shX][shY+1] +	sW[9]*sMemM1[shX][shY+1] +	sW[10]*sMemM2[shX][shY+1] + sW[11]*sMemM3[shX][shY+1];
	}

	//__syncthreads();
	//printf("loc %d, %d, val %f, %f, %f, %f. \n", x, y, (nm0[loc]), (float)(nm1[loc]), (float)(nm2[loc]), (float)(nm3[loc]));

}

__global__ void PF_texture_flow(dim3 srcloc, float src, bool* wallLoc, float* nm0, float* nm1, float* nm2, float* nm3, dim3 matdim, float * WWall, float * W)
{
	__shared__ float sWWall[16];
	__shared__ float sW[16];
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int offset = x + y * blockDim.x * gridDim.x;
	// copy coefficients to shared memory:
	if ((threadIdx.x < 4) && (threadIdx.y < 4))
	{
		sWWall[threadIdx.x + threadIdx.y * 4] = WWall[threadIdx.x + threadIdx.y * 4];
		sW[threadIdx.x + threadIdx.y * 4] = W[threadIdx.x + threadIdx.y * 4];
	}
	__syncthreads();
	float m0, m1, m2, m3;
	if ((x < MATRIX_DIM-1) && (y < MATRIX_DIM-1) && (x > 0) && (y > 0)) // Make sure cell is within the environment grid
	{
		m0 = tex2D(tex_m0, x, y);
		m1 = tex2D(tex_m1, x, y);
		m2 = tex2D(tex_m2, x, y);
		m3 = tex2D(tex_m3, x, y);

		float newF[4] = {0};

		if ((x == srcloc.x) && (y == srcloc.y))
		{
			m0 = src; m1 = src; m2 = src; m3 = src;
		}

		// Check if wall
		bool isWall = wallLoc[x + y * matdim.x];
		//bool isWall = *(wallLoc + x * sof 
		if (isWall)
		{
			// prefetch WWall into __shared__ -- commented out above (maybe textures are faster, check though)
			/*newF[0] = tex2D(tex_WWall,0,0)*m0 + tex2D(tex_WWall,1,0)*m1 + tex2D(tex_WWall,2,0)*m2 + tex2D(tex_WWall,3,0)*m3;
			newF[1] = tex2D(tex_WWall,0,1)*m0 + tex2D(tex_WWall,1,1)*m1 + tex2D(tex_WWall,2,1)*m2 + tex2D(tex_WWall,3,1)*m3;
			newF[2] = tex2D(tex_WWall,0,2)*m0 + tex2D(tex_WWall,1,2)*m1 + tex2D(tex_WWall,2,2)*m2 + tex2D(tex_WWall,3,2)*m3;
			newF[3] = tex2D(tex_WWall,0,3)*m0 + tex2D(tex_WWall,1,3)*m1 + tex2D(tex_WWall,2,3)*m2 + tex2D(tex_WWall,3,3)*m3;*/
			newF[0] = sWWall[0]	*m0 + sWWall[1] *m1 + sWWall[2] *m2 + sWWall[3] *m3;
			newF[1] = sWWall[4]	*m0 + sWWall[5] *m1 + sWWall[6] *m2 + sWWall[7] *m3;
			newF[2] = sWWall[8]	*m0 + sWWall[9] *m1 + sWWall[10]*m2 + sWWall[11]*m3;
			newF[3] = sWWall[12]*m0 + sWWall[13]*m1 + sWWall[14]*m2 + sWWall[15]*m3;
		}
		else
		{
			// prefetch W into __shared__ 
			/*newF[0] = tex2D(tex_W,0,0)*m0 + tex2D(tex_W,1,0)*m1 + tex2D(tex_W,2,0)*m2 + tex2D(tex_W,3,0)*m3;
			newF[1] = tex2D(tex_W,0,1)*m0 + tex2D(tex_W,1,1)*m1 + tex2D(tex_W,2,1)*m2 + tex2D(tex_W,3,1)*m3;
			newF[2] = tex2D(tex_W,0,2)*m0 + tex2D(tex_W,1,2)*m1 + tex2D(tex_W,2,2)*m2 + tex2D(tex_W,3,2)*m3;
			newF[3] = tex2D(tex_W,0,3)*m0 + tex2D(tex_W,1,3)*m1 + tex2D(tex_W,2,3)*m2 + tex2D(tex_W,3,3)*m3;*/
			newF[0] = sW[0]*m0 + sW[1]*m1 + sW[2]*m2 + sW[3]*m3;
			newF[1] = sW[4]*m0 + sW[5]*m1 + sW[6]*m2 + sW[7]*m3;
			newF[2] = sW[8]*m0 + sW[9]*m1 + sW[10]*m2 + sW[11]*m3;
			newF[3] = sW[12]*m0 + sW[13]*m1 + sW[14]*m2 + sW[15]*m3;
		}

		// if (x < MATRIX_DIM-1) nm0[x+1][y] = newF[1];
		if (x < MATRIX_DIM - 1)
			nm0[offset + 1] = newF[1];
		// if (x > 0) nm1[x-1][y] = newF[0];
		if (x > 0) 
			nm1[offset - 1] = newF[0];
		// if (y < MATRIX_DIM-1) nm2[x][y+1] = newF[3];
		if (y < MATRIX_DIM - 1) 
			nm2[offset + blockDim.x * gridDim.x] = newF[3];
		// if (y > 0) nm3[x][y-1] = newF[2];
		if (y > 0) 
			nm3[offset - blockDim.x * gridDim.x] = newF[2];
	}
}

__global__ void testTexturesLoop(void)
{
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			printf("%f, ",(float)(tex2D(tex_m0, (float)(x)+0.5f, (float)(y)+0.5f)));
		}
		printf("\n");
	}
}

__global__ void PF_ptr_flow(cudaPitchedPtr mPtr, cudaExtent mExt, 
									dim3 matrix_dimensions, double src, dim3 srcloc, 
									bool * wallLoc, float * WWall, float * W,
									cudaPitchedPtr nmPtr)
{
	__shared__ float sWWall[16];
	__shared__ float sW[16];
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((threadIdx.x < 4) && (threadIdx.y < 4))
	{
		sWWall[threadIdx.x + threadIdx.y * 4] = WWall[threadIdx.x + threadIdx.y * 4];
		sW[threadIdx.x + threadIdx.y * 4] = W[threadIdx.x + threadIdx.y * 4];
		//if ((threadIdx.x == 0) && (threadIdx.y == 0)) printf("Block %d,%d \n", blockIdx.x,blockIdx.y);
		//printf("x=%d, y=%d, WWall %f, sWWall %f, W %f, sW %f \n", x, y, WWall[x + y * 4], sWWall[x+y * 4], W[x+y*4], sW[x+y*4]);
	}
	__syncthreads();
	if ((x < MATRIX_DIM-1) && (y < MATRIX_DIM-1) && (x > 0) && (y > 0)) // Make sure cell is within the environment grid
	{		
		// Find location within the pitched memory
		float *m = (float*)mPtr.ptr;
		float *nm = (float*)nmPtr.ptr;
	
		size_t pitch = mPtr.pitch;
		unsigned int e_per_row = pitch / SoF;
		size_t slice_pitch = e_per_row * matrix_dimensions.y;

		size_t one_sp = 1 * slice_pitch;
		size_t two_sp = 2 * slice_pitch;
		size_t three_sp = 3 * slice_pitch;
		size_t yep = y * e_per_row;
		float *mxy = m + x + yep;
		float *nmxy = nm + x + yep;

		//float m0 = m[CI(x, y, 0, e_per_row, matrix_dimensions.y)];
		//float * m0ptr = (m + x + y * e_per_row + 0 * slice_pitch);
		//printf("m0ptr, x %d, y %d, is %d \n", x, y, m0ptr);
		//float m0 = *m0ptr;
		float m0 = *(mxy);
		//float m1 = m[CI(x, y, 1, e_per_row, matrix_dimensions.y)];
		float m1 = *(mxy + one_sp);
		//float m2 = m[CI(x, y, 2, e_per_row, matrix_dimensions.y)];
		float m2 = *(mxy + two_sp);
		//float m3 = m[CI(x, y, 3, e_per_row, matrix_dimensions.y)];
		float m3 = *(mxy + three_sp);

		float newF[4] = {0};
	
		// Check if source, assign value if it is
		if ((x == srcloc.x) && (y == srcloc.y))
		{
			m0 = src; m1 = src; m2 = src; m3 = src;
		}

		// Check if wall
		bool isWall = wallLoc[x + y * matrix_dimensions.x];
		//bool isWall = *(wallLoc + x * sof 
		if (isWall)
		{
			// prefetch WWall into __shared__ -- done
			newF[0] = sWWall[0]	*m0 + sWWall[1] *m1 + sWWall[2] *m2 + sWWall[3] *m3;
			newF[1] = sWWall[4]	*m0 + sWWall[5] *m1 + sWWall[6] *m2 + sWWall[7] *m3;
			newF[2] = sWWall[8]	*m0 + sWWall[9] *m1 + sWWall[10]*m2 + sWWall[11]*m3;
			newF[3] = sWWall[12]*m0 + sWWall[13]*m1 + sWWall[14]*m2 + sWWall[15]*m3;
		}
		else
		{
			// prefetch W into __shared__ -- done
			newF[0] = sW[0]*m0 + sW[1]*m1 + sW[2]*m2 + sW[3]*m3;
			newF[1] = sW[4]*m0 + sW[5]*m1 + sW[6]*m2 + sW[7]*m3;
			newF[2] = sW[8]*m0 + sW[9]*m1 + sW[10]*m2 + sW[11]*m3;
			newF[3] = sW[12]*m0 + sW[13]*m1 + sW[14]*m2 + sW[15]*m3;
		}

		//if (x < MATRIX_DIM-1) nm[CI(x + 1, y, 0, e_per_row, matrix_dimensions.y)] = newF[1];		// if (x < MATRIX_DIM-1) nm0[x+1][y] = newF[1];
		if (x < MATRIX_DIM - 1)
			*(nmxy + 1) = newF[1];
		//if (x > 0) nm[CI(x - 1, y, 1, e_per_row, matrix_dimensions.y)] = newF[0];					// if (x > 0) nm1[x-1][y] = newF[0];
		if (x > 0) 
			*(nmxy - 1 + one_sp) = newF[0];
		//if (y < MATRIX_DIM-1) nm[CI(x, y + 1, 2, e_per_row, matrix_dimensions.y)] = newF[3];		// if (y < MATRIX_DIM-1) nm2[x][y+1] = newF[3];
		if (y < MATRIX_DIM - 1) 
			*(nmxy + e_per_row + two_sp) = newF[3];
		//if (y > 0) nm[CI(x, y - 1, 3, e_per_row, matrix_dimensions.y)] = newF[2];					// if (y > 0) nm3[x][y-1] = newF[2];
		if (y > 0) 
			*(nmxy - e_per_row + three_sp) = newF[2];
	}
}

void cPFsetupDisplay(void)
{
	
}

float * dev_m0, *dev_m1, *dev_m2, *dev_m3;
float * dev_nm0, *dev_nm1, *dev_nm2, *dev_nm3;
float * dev_WWall, * dev_W;
bool * dev_wall;

void cPFcaller(unsigned int num_iterations, float * &m_ptr)
{
	gpu_iterations = num_iterations;
	cudaError_t status = cudaSuccess;
	float source = 0.0f;
	dim3 matdim;
	matdim.x = MATRIX_DIM;
	matdim.y = MATRIX_DIM;
	matdim.z = 4;

	dim3 threads(BLOCK_DIMx,BLOCK_DIMy,1);
	dim3 grids(GRID_DIMx,GRID_DIMy,1);

	size_t pitch;

	checkCudaErrors(cudaMallocPitch((void**)&dev_m0, &pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM));
	checkCudaErrors(cudaMallocPitch((void**)&dev_m1, &pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM));
	checkCudaErrors(cudaMallocPitch((void**)&dev_m2, &pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM));
	checkCudaErrors(cudaMallocPitch((void**)&dev_m3, &pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM));
	checkCudaErrors(cudaMallocPitch((void**)&dev_nm0, &pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM));
	checkCudaErrors(cudaMallocPitch((void**)&dev_nm1, &pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM));
	checkCudaErrors(cudaMallocPitch((void**)&dev_nm2, &pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM));
	checkCudaErrors(cudaMallocPitch((void**)&dev_nm3, &pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM));

	checkCudaErrors(cudaMalloc( (void**)&dev_WWall, WWAL_DIMx*WWAL_DIMy*sizeof(float))); // WWall
	checkCudaErrors(cudaMalloc( (void**)&dev_W, W_DIMx*W_DIMy*sizeof(float))); // W

	checkCudaErrors(cudaMemset2D(dev_m0, pitch, 0, MATRIX_DIM*sizeof(float), MATRIX_DIM)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_m1, pitch, 0, MATRIX_DIM*sizeof(float), MATRIX_DIM)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_m2, pitch, 0, MATRIX_DIM*sizeof(float), MATRIX_DIM)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_m3, pitch, 0, MATRIX_DIM*sizeof(float), MATRIX_DIM)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_nm0, pitch, 0, MATRIX_DIM*sizeof(float), MATRIX_DIM)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_nm1, pitch, 0, MATRIX_DIM*sizeof(float), MATRIX_DIM)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_nm2, pitch, 0, MATRIX_DIM*sizeof(float), MATRIX_DIM)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_nm3, pitch, 0, MATRIX_DIM*sizeof(float), MATRIX_DIM)); // set 0 to every BYTE

	checkCudaErrors(cudaMemcpy(dev_WWall, host_WWall, WWAL_DIMx*WWAL_DIMy*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_W, host_W, W_DIMx*W_DIMy*sizeof(float), cudaMemcpyHostToDevice));

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); // not happy?
	tex_m0.normalized = false;	tex_m0.filterMode = cudaFilterModeLinear; tex_m0.addressMode[0] = cudaAddressModeBorder;
	tex_m1.normalized = false;	tex_m1.filterMode = cudaFilterModeLinear;tex_m1.addressMode[0] = cudaAddressModeBorder;
	tex_m2.normalized = false;	tex_m2.filterMode = cudaFilterModeLinear;tex_m2.addressMode[0] = cudaAddressModeBorder;
	tex_m3.normalized = false;	tex_m3.filterMode = cudaFilterModeLinear;tex_m3.addressMode[0] = cudaAddressModeBorder;
	tex_nm0.normalized = false;	tex_nm0.filterMode = cudaFilterModeLinear;tex_nm0.addressMode[0] = cudaAddressModeBorder;
	tex_nm1.normalized = false;	tex_nm1.filterMode = cudaFilterModeLinear;tex_nm1.addressMode[0] = cudaAddressModeBorder;
	tex_nm2.normalized = false;	tex_nm2.filterMode = cudaFilterModeLinear;tex_nm2.addressMode[0] = cudaAddressModeBorder;
	tex_nm3.normalized = false;	tex_nm3.filterMode = cudaFilterModeLinear;tex_nm3.addressMode[0] = cudaAddressModeBorder;
	
	
	checkCudaErrors(cudaBindTexture2D(NULL, tex_m0, dev_m0, desc, MATRIX_DIM, MATRIX_DIM, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_m1, dev_m1, desc, MATRIX_DIM, MATRIX_DIM, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_m2, dev_m2, desc, MATRIX_DIM, MATRIX_DIM, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_m3, dev_m3, desc, MATRIX_DIM, MATRIX_DIM, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_nm0, dev_nm0, desc, MATRIX_DIM, MATRIX_DIM, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_nm1, dev_nm1, desc, MATRIX_DIM, MATRIX_DIM, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_nm2, dev_nm2, desc, MATRIX_DIM, MATRIX_DIM, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_nm3, dev_nm3, desc, MATRIX_DIM, MATRIX_DIM, pitch));

	// Allocate 2D array for wall (unrolled to 1D) -- implement hash table
	checkCudaErrors(cudaMalloc((void**)&dev_wall, matdim.x*matdim.y*sizeof(bool))); // x*y elements in a 1D array
	checkCudaErrors(cudaMemcpy(dev_wall, host_Wall, matdim.x*matdim.y*sizeof(bool), cudaMemcpyHostToDevice));
		
	source = 0.0f;
	checkCudaErrors(cudaDeviceSynchronize());

	int shared_mem_size = 2 * WWAL_DIMx * WWAL_DIMy * sizeof(float) + BLOCK_DIMx*BLOCK_DIMy*4*sizeof(float);
	
	float * p_src_m0 = dev_m0 + src_loc.y * pitch/sizeof(float) + src_loc.x;
	float * p_src_m1 = dev_m1 + src_loc.y * pitch/sizeof(float) + src_loc.y;
	float * p_src_m2 = dev_m2 + src_loc.y * pitch/sizeof(float) + src_loc.x;
	float * p_src_m3 = dev_m3 + src_loc.y * pitch/sizeof(float) + src_loc.y;

	clock_t t2; t2=clock(); // begin timing

	for (int iter = 0; iter < gpu_iterations; iter++)
	{
		source = src_amplitude * sin(2 * PI * src_frequency * (double)(iter) * 0.01);

		checkCudaErrors(cudaMemcpyAsync(p_src_m0, &source, sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(p_src_m1, &source, sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(p_src_m2, &source, sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(p_src_m3, &source, sizeof(float), cudaMemcpyHostToDevice));
		 
		checkCudaErrors(cudaDeviceSynchronize());

		//printf("Calculation \n");
		PF_padded_texture_flow<<<grids,threads,shared_mem_size>>>(src_loc, source, dev_wall, dev_nm0, dev_nm1, dev_nm2, dev_nm3, matdim, dev_WWall, dev_W, pitch);
		// PF_padded_texture_flow(dim3 srcloc, float src, bool* wallLoc, float*nm0, float*nm1, float* nm2, float* nm3, dim3 matdim, float * WWall, float *W)
		//checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		//printf("NM texture values \n");
		//testTexturesLoop<<<1,1>>>();
		//cudaDeviceSynchronize();
		
		PF_padded_texture_copy<<<grids,threads>>>(dev_m0, dev_m1, dev_m2, dev_m3, matdim, pitch);
		cudaDeviceSynchronize();

		//printf("M texture values \n");
		//testTexturesLoop<<<1,1>>>();
		//cudaDeviceSynchronize();
	}

	long int final=clock()-t2; printf("GPU iterations took %li ticks (%f seconds) \n", final, ((float)final)/CLOCKS_PER_SEC);
	
	m_host = (float *)malloc(sizeof(float)*MATRIX_DIM*MATRIX_DIM);
	m_ptr = m_host; // So that the class can access M values
	
	checkCudaErrors(cudaMemcpy(m_host, dev_m0, MATRIX_DIM*MATRIX_DIM*sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	//status = cudaMemcpy3D(&hm_p);

	//if (status != cudaSuccess){printf("Uhoh: %s \n", cudaGetErrorString(status));}

	// Free all allocated memory (move into separate delete function later)
	cudaFree(dev_m0);
	cudaFree(dev_m1);
	cudaFree(dev_m2);
	cudaFree(dev_m3);
	cudaFree(dev_nm0);
	cudaFree(dev_nm1);
	cudaFree(dev_nm2);
	cudaFree(dev_nm3);
	cudaFree(dev_WWall);
	cudaFree(dev_W);
	cudaUnbindTexture(tex_m0);
	cudaUnbindTexture(tex_m1);
	cudaUnbindTexture(tex_m2);
	cudaUnbindTexture(tex_m3);
	cudaUnbindTexture(tex_nm0);
	cudaUnbindTexture(tex_nm1);
	cudaUnbindTexture(tex_nm2);
	cudaUnbindTexture(tex_nm3);
	cudaUnbindTexture(tex_WWall);
	cudaUnbindTexture(tex_W);

	//cudaFree(m_device.ptr);
	//cudaFree(nm_device.ptr);
	//cudaFree(dev_wall);
	//cudaFree(dev_WWall);
	//cudaFree(dev_W);
}

void cPFinit(float matrixFlow[][4], float matrixWall[][4], float in_sourceLoc[])
{
	// Initialize some values
	coef = 1;
	src_amplitude = 1.0;
	src_frequency = 1.0;

	host_Wall = (bool *)malloc(sizeof(bool)*MATRIX_DIM*MATRIX_DIM); 
	memset(host_Wall, 0, MATRIX_DIM*MATRIX_DIM*sizeof(bool));

	host_WWall = (float *)malloc(sizeof(float)*WWAL_DIMx*WWAL_DIMy);
	host_W = (float *)malloc(sizeof(float)*W_DIMx*W_DIMy);
	
	for (int y = 0; y < WWAL_DIMy; y++)
	{
		for (int x = 0; x < WWAL_DIMx; x++)
		{
			host_WWall[x+y*WWAL_DIMx] = matrixWall[x][y]* (coef/2.0);
			host_W[x+y*W_DIMx] = matrixFlow[x][y]* (coef/2.0);
		}
	}

	// copy source loc:
	src_loc.x = in_sourceLoc[0];
	src_loc.y = in_sourceLoc[1];
}

void cPFaddWallLocation(int x, int y, bool val)
{
	if (host_Wall != NULL)	host_Wall[x+y*MATRIX_DIM] = val;
}

void cPFdelete(void)
{
	///*if (host_W != NULL) */free(host_W);
	///*if (host_WWall != NULL) */free(host_WWall);
	///*if (host_Wall != NULL) */free(host_Wall);
	//free(m_host);
}