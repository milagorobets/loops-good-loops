// TO DO:
// 1. coalesce memory accesses for m and nm
// 2. put FP_ptr_copy into pointer form
// 3. split kernels into edges and middle blocks (middle do not if checks in flow) -- started
// 4. try arrays for more coalesced accesses
// 5. reduce if statements

#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cPFkernel_ptr.cuh"
#include "utils.h"
#include "common.h"
#include <glew.h>
#include <freeglut.h>
#include "book.h"
#include "gpu_anim.h"
#include "cusparse.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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

#define DEL_MIN -10
#define DEL_MAX 10
#define WALL_R 2

#if (MATRIX_DIM < 512) 
	#define BIT_DIM 512
#else 
	#define BIT_DIM MATRIX_DIM
#endif

__constant__ float cW[16];
#define STR 0.0
#define BND 0.5
#define INC_EAST tex_m1
#define INC_WEST tex_m0
#define INC_NORTH tex_m2
#define INC_SOUTH tex_m3

int mWidth = 1, mHeight = 1;
dim3 colorgrid(MATRIX_DIM,MATRIX_DIM,1);
dim3 colorthreads(512/MATRIX_DIM,512/MATRIX_DIM,1);

byte * host_Wall;
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
texture<float,2,cudaReadModeElementType> tex_avg_m;

cudaStream_t v_stream1, v_stream2, v_stream3, v_stream4;
dim3 v_threads;
dim3 v_grids;
dim3 v_matdim;
float * v_p_src_m0;
float * v_p_src_m1;
float * v_p_src_m2;
float * v_p_src_m3;
size_t v_pitch;
int v_shared_mem_size;

float * dev_m0, *dev_m1, *dev_m2, *dev_m3, *dev_avg_m;
float * dev_nm0, *dev_nm1, *dev_nm2, *dev_nm3;
float * dev_WWall, * dev_W;
byte * dev_wall;

byte * host_src;
byte * dev_src;

// KEYBOARD BUTTON FUNCTIONS
// Adds a source at specified location (x, y) if no source exists there
void addSrc(int x, int y)
{
	if (!(host_src[x + (mHeight - 1 - y) * mWidth]))
	{
		host_src[x + (mHeight - 1 - y) * mWidth] = 1;
		checkCudaErrors(cudaMemcpy(dev_src, host_src, mWidth*mHeight*sizeof(byte), cudaMemcpyHostToDevice));
	}
}

// Adds a group of wall pixels around the specified location
void addWall(int x, int y)
{
	int ry = (mHeight - 1 - y); // Coordinates are flipped between OpenGL and CUDA
	for (int iy = -WALL_R; iy < WALL_R; iy++)
	{
		for (int ix = -WALL_R; ix < WALL_R; ix++)
		{
			if (((x + ix)>0) && ((ry + iy) > 0) && ((x + ix) < mWidth) && ((y + iy) < mHeight))
			{
				host_Wall[x + ix + (ry + iy) * mWidth] = 1;
			}
		}
	}
	checkCudaErrors(cudaMemcpy(dev_wall, host_Wall, mWidth*mHeight*sizeof(byte), cudaMemcpyHostToDevice));
}

// Remove a section of the wall pixels around the specified location
void removeWall(int x, int y)
{
	int ry = (mHeight - 1 - y);
	for (int iy = -WALL_R*2; iy < WALL_R*2; iy++)
	{
		for (int ix = -WALL_R*2; ix < WALL_R*2; ix++)
		{
			if (((x + ix)>0) && ((ry + iy) > 0) && ((x + ix)<mWidth) && ((y + iy)<mHeight))
			{
				host_Wall[x + ix + (ry + iy) * mWidth] = 0;
			}
		}
	}
	checkCudaErrors(cudaMemcpy(dev_wall, host_Wall, mWidth*mHeight*sizeof(byte), cudaMemcpyHostToDevice));
}

// Remove a source near the specified location
void removeSrc(int x, int y)
{
	int ry = (mHeight - 1 - y);
	for (int iy = DEL_MIN; iy < DEL_MAX; iy++)
	{
		for (int ix = DEL_MIN; ix < DEL_MAX; ix++)
		{
			if (((x + ix)>0) && ((ry + iy) > 0))
			{
				host_src[x + ix + (ry + iy) * mWidth] = 0;
			}
		}
	}
	checkCudaErrors(cudaMemcpy(dev_src, host_src, mWidth*mHeight*sizeof(byte), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset2D(dev_m0, v_pitch, 0, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMemset2D(dev_m1, v_pitch, 0, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMemset2D(dev_m2, v_pitch, 0, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMemset2D(dev_m3, v_pitch, 0, mWidth*sizeof(float), mHeight));
}

// Remove all sources
void removeAllSrc(void)
{
	memset(host_src, 0, mWidth*mHeight*sizeof(byte));
	checkCudaErrors(cudaMemset(dev_src, 0, mWidth*mHeight*sizeof(byte)));
	checkCudaErrors(cudaMemset2D(dev_m0, v_pitch, 0, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMemset2D(dev_m1, v_pitch, 0, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMemset2D(dev_m2, v_pitch, 0, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMemset2D(dev_m3, v_pitch, 0, mWidth*sizeof(float), mHeight));
}

// ~*~*~*~ GPU KERNELS ~*~*~*~

// Copy kernel that uses indexes (despite the name) to access locations within m and nm
__global__ void PF_ptr_copy(cudaPitchedPtr mPtr, cudaPitchedPtr nmPtr, cudaExtent mExt, dim3 matdim)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float *m = (float*)mPtr.ptr;
	float *nm = (float*)nmPtr.ptr;
	size_t pitch = mPtr.pitch;
	unsigned int e_per_row = pitch / SoF;
	size_t slice_pitch = pitch*mExt.height;

	if ((x < matdim.x) && (y < matdim.y))
	{
		m[CI(x, y, 0, e_per_row, matdim.y)] = nm[CI(x, y, 0, e_per_row, matdim.y)];
		m[CI(x, y, 1, e_per_row, matdim.y)] = nm[CI(x, y, 1, e_per_row, matdim.y)];
		m[CI(x, y, 2, e_per_row, matdim.y)] = nm[CI(x, y, 2, e_per_row, matdim.y)];
		m[CI(x, y, 3, e_per_row, matdim.y)] = nm[CI(x, y, 3, e_per_row, matdim.y)];

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

// Copy kernel that also attenuates values in walls and updates the source
__global__ void PF_copy_withWall(float*m0, float*m1, float*m2, float*m3, byte * wall, dim3 matdim, size_t pitch, byte * src, float source_val)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int loc = x + y * pitch/sizeof(float);
	if ((x < matdim.x) && (y < matdim.y)) // Make sure cell is within the environment grid
	{
		float t0 = tex2D(tex_nm0, x+0.5f, y+0.5f); float t1 = tex2D(tex_nm1, x+0.5f, y+0.5f);
		float t2 = tex2D(tex_nm2, x+0.5f, y+0.5f); float t3 = tex2D(tex_nm3, x+0.5f, y+0.5f);

		// edge cases
		if ((x == 0) && (t0 == 0))
		{
			t0 = tex2D(tex_nm0, x+1.5f, y+0.5f);
		}
		if ((x == matdim.x-1) && (t1 == 0))
		{
			t1 = tex2D(tex_nm1, x-1.0f+0.5f, y+0.5f);
		}
		if ((y == 0) && (t2 == 0))
		{
			t2 = tex2D(tex_nm2, x+0.5f, y+1.0f+0.5f);
		}
		if ((y == matdim.y-1) && (t3 == 0))
		{
			t3 = tex2D(tex_nm3, x+0.5f, y-1.0f+0.5f);
		}

		// write values
		if (wall[x + y * matdim.x] == 1)
		{
			m0[loc] = WALL_DEC*t0; // WALL_DEC can be changed in common.h
			m1[loc] = WALL_DEC*t1;
			m2[loc] = WALL_DEC*t2;
			m3[loc] = WALL_DEC*t3;
		}
		else if (src[x + y * matdim.x] == 1) // If this is a source cell
		{
			m0[loc] = source_val + t0;
			/*m1[loc] = t1;
			m2[loc] = t2;
			m3[loc] = t3;*/
			m1[loc] = source_val + t1;
			m2[loc] = source_val + t2;
			m3[loc] = source_val + t3;
		}
		else
		{
			m0[loc] = t0;
			m1[loc] = t1;
			m2[loc] = t2;
			m3[loc] = t3;
		}
	}	

}

// Copy kernel for padded textures (PF_copy_withWall is more up-to-date and does more)
__global__ void PF_padded_texture_copy(float*m0, float*m1, float*m2, float*m3, dim3 matdim, size_t pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int loc = x + y * pitch/sizeof(float);
	if ((x < matdim.x) && (y < matdim.y)) // Make sure cell is within the environment grid
	{
		float t0 = tex2D(tex_nm0, x+0.5f, y+0.5f); float t1 = tex2D(tex_nm1, x+0.5f, y+0.5f);
		float t2 = tex2D(tex_nm2, x+0.5f, y+0.5f); float t3 = tex2D(tex_nm3, x+0.5f, y+0.5f);

		// edge cases
		if ((x == 0) && (t0 == 0))
		{
			t0 = tex2D(tex_nm0, x+1.5f, y+0.5f);
		}
		if ((x == matdim.x-1) && (t1 == 0))
		{
			t1 = tex2D(tex_nm1, x-1.0f+0.5f, y+0.5f);
		}
		if ((y == 0) && (t2 == 0))
		{
			t2 = tex2D(tex_nm2, x+0.5f, y+1.0f+0.5f);
		}
		if ((y == matdim.y-1) && (t3 == 0))
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

// Propagation kernel cross-checked with the audio guy's code
// THIS ONE CREATES CIRCULAR WAVES
__global__ void PF_roundscatter(float *nm0, float *nm1, float *nm2, float *nm3, size_t pitch)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	float xt = x + 0.5f;
	float yt = y + 0.5f;

	//ScatteredNorth[x, y-1] = 0.5m * (IEast - INorth + IWest + ISouth);
	float sn = 0.5 * (	tex2D(INC_EAST, xt, yt - 1) -
						tex2D(INC_NORTH, xt, yt - 1) +
						tex2D(INC_WEST, xt,  yt - 1) +
						tex2D(INC_SOUTH, xt, yt - 1));

	//ScatteredEast[x-1, y] = 0.5m * (-IEast + INorth + IWest + ISouth);
	float se = 0.5 * (0-tex2D(INC_EAST, xt - 1, yt) +
						tex2D(INC_NORTH, xt - 1, yt) +
						tex2D(INC_WEST, xt - 1, yt) +
						tex2D(INC_SOUTH, xt - 1, yt));

	//ScatteredWest[x+1, y] = 0.5m * (IEast + INorth - IWest + ISouth);
	float sw = 0.5 * (	tex2D(INC_EAST, xt + 1, yt) +
						tex2D(INC_NORTH, xt + 1, yt) -
						tex2D(INC_WEST, xt + 1, yt) + 
						tex2D(INC_SOUTH, xt + 1, yt));
	
	//ScatteredSouth[x, y+1] = 0.5m * (IEast + INorth + IWest - ISouth);
	float ss = 0.5 * (	tex2D(INC_EAST, xt, yt + 1) +
						tex2D(INC_NORTH, xt, yt + 1) +
						tex2D(INC_WEST, xt, yt + 1) -
						tex2D(INC_SOUTH, xt, yt + 1));

	//IncomingEast[x, y] = ScatteredWest[x + 1, y];
	nm1[x + y * pitch/sizeof(float)] = sw;
	//IncomingNorth[x, y] = ScatteredSouth[x, y + 1];
	nm2[x + y * pitch/sizeof(float)] = ss;
	//IncomingWest[x, y] = ScatteredEast[x - 1, y];
	nm0[x + y * pitch/sizeof(float)] = se;
	//IncomingSouth[x, y] = ScatteredNorth[x, y - 1];*/
	nm3[x + y * pitch/sizeof(float)] = sn;

}

// THE PROPAGATION KERNELS BELOW NEED TO BE CHECKED
// Propagation kernel that handles wave motion to the right
__global__ void PF_texture_slideright(float *nm0, size_t pitch)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
#if 1
	nm0[x + y * pitch/sizeof(float)] =	tex2D(tex_m0, (float)(x) - 0.5f, (float)(y) + 0.5f)*cW[4] +
										tex2D(tex_m1, (float)(x) - 0.5f, (float)(y) + 0.5f)*cW[5] +
										tex2D(tex_m2, (float)(x) - 0.5f, (float)(y) + 0.5f)*cW[6] +
										tex2D(tex_m3, (float)(x) - 0.5f, (float)(y) + 0.5f)*cW[7];
#else
	nm0[x + y * pitch/sizeof(float)] =	tex2D(tex_m0, (float)(x) - 0.5f, (float)(y) + 0.5f) *(-STR) +
										tex2D(tex_m1, (float)(x) - 0.5f, (float)(y) + 0.5f) *(STR) +
										tex2D(tex_m2, (float)(x) - 0.5f, (float)(y) + 0.5f) *(STR) +
										tex2D(tex_m3, (float)(x) - 0.5f, (float)(y) + 0.5f) *(STR) +
										(0 +
										tex2D(tex_m2, (float)(x) - 0.5f, (float)(y) + 1.5f) -
										tex2D(tex_m0, (float)(x) - 0.5f, (float)(y) + 1.5f) -
										tex2D(tex_m0, (float)(x) - 0.5f, (float)(y) - 0.5f) +
										tex2D(tex_m3, (float)(x) - 0.5f, (float)(y) - 0.5f)										
										) * BND;
#endif
}

// Propagation kernel that handles wave motion to the left
__global__ void PF_texture_slideleft(float *nm1, size_t pitch)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
#if 1
	nm1[x + y * pitch/sizeof(float)] =	tex2D(tex_m0, (float)(x) + 1.5f, (float)(y) + 0.5f)*cW[0] +
										tex2D(tex_m1, (float)(x) + 1.5f, (float)(y) + 0.5f)*cW[1] +
										tex2D(tex_m2, (float)(x) + 1.5f, (float)(y) + 0.5f)*cW[2] +
										tex2D(tex_m3, (float)(x) + 1.5f, (float)(y) + 0.5f)*cW[3];
# else
	nm1[x + y * pitch/sizeof(float)] =	tex2D(tex_m0, (float)(x) + 1.5f, (float)(y) + 0.5f)*(STR) +
										tex2D(tex_m1, (float)(x) + 1.5f, (float)(y) + 0.5f)*(-STR) +
										tex2D(tex_m2, (float)(x) + 1.5f, (float)(y) + 0.5f)*STR +
										tex2D(tex_m3, (float)(x) + 1.5f, (float)(y) + 0.5f)*STR +
										(0 +
										tex2D(tex_m3, (float)(x) + 1.5f, (float)(y) - 0.5f) - 
										tex2D(tex_m1, (float)(x) + 1.5f, (float)(y) - 0.5f) -
										tex2D(tex_m1, (float)(x) + 1.5f, (float)(y) + 1.5f) +
										tex2D(tex_m2, (float)(x) + 1.5f, (float)(y) + 1.5f)										
										) * BND;
#endif
}

// Propagation kernel that handles wave motion up
__global__ void PF_texture_slideup(float *nm3, size_t pitch)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
#if 1
	nm3[x + y * pitch/sizeof(float)] =	tex2D(tex_m0, (float)(x) + 0.5f, (float)(y) - 0.5f)*cW[8] +
										tex2D(tex_m1, (float)(x) + 0.5f, (float)(y) - 0.5f)*cW[9] +
										tex2D(tex_m2, (float)(x) + 0.5f, (float)(y) - 0.5f)*cW[10] +
										tex2D(tex_m3, (float)(x) + 0.5f, (float)(y) - 0.5f)*cW[11];
#else
	nm3[x + y * pitch/sizeof(float)] =	tex2D(tex_m0, (float)(x) + 0.5f, (float)(y) + 1.5f)*STR +
										tex2D(tex_m1, (float)(x) + 0.5f, (float)(y) + 1.5f)*STR +
										tex2D(tex_m2, (float)(x) + 0.5f, (float)(y) + 1.5f)*STR +
										tex2D(tex_m3, (float)(x) + 0.5f, (float)(y) + 1.5f)*(-STR)+
										(0 +
										tex2D(tex_m0, (float)(x) + 1.5f, (float)(y) + 1.5f) -
										tex2D(tex_m3, (float)(x) + 1.5f, (float)(y) + 1.5f) +
										tex2D(tex_m1, (float)(x) - 0.5f, (float)(y) - 0.5f) -
										tex2D(tex_m3, (float)(x) - 0.5f, (float)(y) - 0.5f) 
										) * BND;
#endif
}

// Propagation kernel that handles wave motion down
__global__ void PF_texture_slidedown(float *nm2, size_t pitch)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
#if 1
	nm2[x + y * pitch/sizeof(float)] =	tex2D(tex_m0, (float)(x) + 0.5f, (float)(y) + 1.5f)*cW[12] +
										tex2D(tex_m1, (float)(x) + 0.5f, (float)(y) + 1.5f)*cW[13] +
										tex2D(tex_m2, (float)(x) + 0.5f, (float)(y) + 1.5f)*cW[14] +
										tex2D(tex_m3, (float)(x) + 0.5f, (float)(y) + 1.5f)*cW[15];
#else
	nm2[x + y * pitch/sizeof(float)] =	tex2D(tex_m0, (float)(x) + 0.5f, (float)(y) - 0.5f)*STR +
										tex2D(tex_m1, (float)(x) + 0.5f, (float)(y) - 0.5f)*STR +
										tex2D(tex_m2, (float)(x) + 0.5f, (float)(y) - 0.5f)*(-STR) +
										tex2D(tex_m3, (float)(x) + 0.5f, (float)(y) - 0.5f)*STR +
										(0 +
										tex2D(tex_m1, (float)(x) - 0.5f, (float)(y) - 0.5f) -
										tex2D(tex_m2, (float)(x) - 0.5f, (float)(y) - 0.5f) +
										tex2D(tex_m0, (float)(x) + 1.5f, (float)(y) - 0.5f) -
										tex2D(tex_m2, (float)(x) + 1.5f, (float)(y) - 0.5f)
										) * BND;
#endif
}	

// Propagation kernel that uses textures and registers (to reduce shared memory use)
__global__ void PF_registers_texture_flow(float * nm0, float * nm1, float * nm2, float * nm3, float * W, size_t pitch, dim3 matdim)
{
	__shared__ float sW[16];
	float t0, t1, t2, t3;
	float x = threadIdx.x + blockIdx.x * blockDim.x;
	float y = threadIdx.y + blockIdx.y * blockDim.y;
	int loc = x + y * pitch/sizeof(float);
	x += 0.5f; y += 0.5f;

	if ((threadIdx.x < 4) && (threadIdx.y < 4))
	{
		sW[threadIdx.x + threadIdx.y * 4] = W[threadIdx.x + threadIdx.y * 4];
	}
	
	if ((x < matdim.x) && (y < matdim.y))
	{
		nm0[loc] = tex2D(tex_m0, x-1,y)*sW[4] + tex2D(tex_m1, x-1, y)*sW[5] + tex2D(tex_m2, x-1, y)*sW[6] + tex2D(tex_m3, x-1, y)*sW[7];
		nm1[loc] = tex2D(tex_m0, x+1,y)*sW[0] + tex2D(tex_m1, x+1, y)*sW[1] + tex2D(tex_m2, x+1, y)*sW[2] + tex2D(tex_m3, x+1, y)*sW[3];
		nm2[loc] = tex2D(tex_m0, x,y-1)*sW[12] + tex2D(tex_m1, x, y-1)*sW[13] + tex2D(tex_m2, x, y-1)*sW[14] + tex2D(tex_m3, x, y-1)*sW[15];
		nm3[loc] = tex2D(tex_m0, x,y+1)*sW[8] + tex2D(tex_m1, x, y+1)*sW[9] + tex2D(tex_m2, x, y+1)*sW[10] + tex2D(tex_m3, x, y+1)*sW[11];
	}

}

// Propagation kernel that uses a lot of shared memory (too much)
__global__ void PF_padded_texture_flow(	dim3 srcloc, float src, bool* wallLoc, float*nm0, float*nm1, 
										float* nm2, float* nm3, dim3 matdim, float * WWall, float *W, size_t pitch)
{
	__shared__ float sWWall[16];
	__shared__ float sW[16];
	__shared__ float sMemM0[BLOCK_DIMx+2][BLOCK_DIMy+2]; 
	__shared__ float sMemM1[BLOCK_DIMx+2][BLOCK_DIMy+2];
	__shared__ float sMemM2[BLOCK_DIMx+2][BLOCK_DIMy+2];
	__shared__ float sMemM3[BLOCK_DIMx+2][BLOCK_DIMy+2];
	float x = threadIdx.x + blockIdx.x * blockDim.x + 0.5f;
	float y = threadIdx.y + blockIdx.y * blockDim.y + 0.5f;
	unsigned int shX = threadIdx.x + 1;
	unsigned int shY = threadIdx.y + 1;
	int loc = x + y * pitch/sizeof(float);

	// copy coefficients to shared memory:
	if ((threadIdx.x < 4) && (threadIdx.y < 4))
	{
		sWWall[threadIdx.x + threadIdx.y * 4] = WWall[threadIdx.x + threadIdx.y * 4];
		sW[threadIdx.x + threadIdx.y * 4] = W[threadIdx.x + threadIdx.y * 4];
	}
	__syncthreads();
		
	if ((x < matdim.x) && (y < matdim.y)) // Make sure cell is within the environment grid
	{
		sMemM0[shX][shY] = tex2D(tex_m0, x, y);
		sMemM1[shX][shY] = tex2D(tex_m1, x, y);
		sMemM2[shX][shY] = tex2D(tex_m2, x, y);
		sMemM3[shX][shY] = tex2D(tex_m3, x, y);
	
		// handle edges
		if (threadIdx.x == 0) // left
		{
			sMemM0[0][shY] = tex2D(tex_m0, x-1.0f, y); sMemM1[0][shY] = tex2D(tex_m1, x-1.0f, y);
			sMemM2[0][shY] = tex2D(tex_m2, x-1.0f, y); sMemM3[0][shY] = tex2D(tex_m3, x-1.0f, y);
		}
		else if (threadIdx.x == (blockDim.x - 1)) // right
		{
			sMemM0[blockDim.x+1][shY] = tex2D(tex_m0, x+1.0f, y); sMemM1[blockDim.x+1][shY] = tex2D(tex_m1, x+1.0f, y);
			sMemM2[blockDim.x+1][shY] = tex2D(tex_m2, x+1.0f, y); sMemM3[blockDim.x+1][shY] = tex2D(tex_m3, x+1.0f, y);
		}
		// MISSING THE CORNER BLOCK~ FIX IT

		if (threadIdx.y == 0) // up
		{
			sMemM0[shX][0] = tex2D(tex_m0, x, y-1); sMemM1[shX][0] = tex2D(tex_m1, x, y-1); 
			sMemM2[shX][0] = tex2D(tex_m2, x, y-1); sMemM3[shX][0] = tex2D(tex_m3, x, y-1); 
		}
		else if (threadIdx.y == (blockDim.y - 1)) // down
		{
			sMemM0[shX][blockDim.y+1] = tex2D(tex_m0, x, y+1); sMemM1[shX][blockDim.y+1] = tex2D(tex_m1, x, y+1);
			sMemM2[shX][blockDim.y+1] = tex2D(tex_m2, x, y+1); sMemM3[shX][blockDim.y+1] = tex2D(tex_m3, x, y+1);
		}
	}
	__syncthreads(); // sync the shared memory writes

	if ((x < matdim.x) && (y < matdim.y)) // Make sure cell is within the environment grid
	{
		// calculate nm 
		nm0[loc] = sW[4]*sMemM0[shX-1][shY] +	sW[5]*sMemM1[shX-1][shY] +	sW[6]*sMemM2[shX-1][shY] +	sW[7]*sMemM3[shX-1][shY];
		nm1[loc] = sW[0]*sMemM0[shX+1][shY] +	sW[1]*sMemM1[shX+1][shY] +	sW[2]*sMemM2[shX+1][shY] +	sW[3]*sMemM3[shX+1][shY];
		nm2[loc] = sW[12]*sMemM0[shX][shY-1] +	sW[13]*sMemM1[shX][shY-1] +	sW[14]*sMemM2[shX][shY-1] + sW[15]*sMemM3[shX][shY-1];
		nm3[loc] = sW[8]*sMemM0[shX][shY+1] +	sW[9]*sMemM1[shX][shY+1] +	sW[10]*sMemM2[shX][shY+1] + sW[11]*sMemM3[shX][shY+1];
	}
}

// Propagation kernel that uses textures
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
	if ((x < matdim.x-1) && (y < matdim.y-1) && (x > 0) && (y > 0)) // Make sure cell is within the environment grid
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
		if (isWall)
		{
			newF[0] = sWWall[0]	*m0 + sWWall[1] *m1 + sWWall[2] *m2 + sWWall[3] *m3;
			newF[1] = sWWall[4]	*m0 + sWWall[5] *m1 + sWWall[6] *m2 + sWWall[7] *m3;
			newF[2] = sWWall[8]	*m0 + sWWall[9] *m1 + sWWall[10]*m2 + sWWall[11]*m3;
			newF[3] = sWWall[12]*m0 + sWWall[13]*m1 + sWWall[14]*m2 + sWWall[15]*m3;
		}
		else
		{
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

// Propagation kernel that uses global memory and pointers
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
	if ((x < matrix_dimensions.x-1) && (y < matrix_dimensions.y-1) && (x > 0) && (y > 0)) // Make sure cell is within the environment grid
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
		if (x < matrix_dimensions.x - 1)
			*(nmxy + 1) = newF[1];
		//if (x > 0) nm[CI(x - 1, y, 1, e_per_row, matrix_dimensions.y)] = newF[0];					// if (x > 0) nm1[x-1][y] = newF[0];
		if (x > 0) 
			*(nmxy - 1 + one_sp) = newF[0];
		//if (y < MATRIX_DIM-1) nm[CI(x, y + 1, 2, e_per_row, matrix_dimensions.y)] = newF[3];		// if (y < MATRIX_DIM-1) nm2[x][y+1] = newF[3];
		if (y < matrix_dimensions.y - 1) 
			*(nmxy + e_per_row + two_sp) = newF[3];
		//if (y > 0) nm[CI(x, y - 1, 3, e_per_row, matrix_dimensions.y)] = newF[2];					// if (y > 0) nm3[x][y-1] = newF[2];
		if (y > 0) 
			*(nmxy - e_per_row + three_sp) = newF[2];
	}
}

// ~*~*~*~ COLORING CONVERSION FUNCTIONS ~*~*~*~

// Converts average power values into colors (stepped)
__global__ void float_to_color_power_dBm( uchar4 *optr, size_t pitch, byte* walls, dim3 matdim) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * matdim.x;

	float l = tex2D(tex_avg_m, (x)+0.5f, (y)+0.5f);
	
	// Convert l to dBm:
	l = 10 * log10f(abs(l)); // abs == 0 -l when negative, faster?

	if (l < -100) // RED
	{
		optr[offset].x = 255; optr[offset].y = 0; optr[offset].z = 0;
	}
	else if (l < -90) // BLUE
	{
		optr[offset].x = 0; optr[offset].y = 9; optr[offset].z = 255;
	}
	else if (l < -80) // ORANGE
	{
		optr[offset].x = 255; optr[offset].y = 154; optr[offset].z = 0;
	}
	else if (l < -70) // YELLOW
	{
		optr[offset].x = 255; optr[offset].y = 247; optr[offset].z = 0;
	}
	else // GREEN
	{
		optr[offset].x = 40; optr[offset].y = 172; optr[offset].z = 7;
	}

	if (walls[offset]) // Walls are black
	{
		optr[offset].x = 0; optr[offset].y = 0; optr[offset].z = 0;
	}

}

// Converts amplitudes to dBm to colors (stepped)
__global__ void float_to_color_dBm( uchar4 *optr, size_t pitch, dim3 matdim) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * matdim.y;
	int poffset = x + y * pitch/sizeof(float);

	float l =	(tex2D(tex_m0, (x)+0.5f, (y)+0.5f))+
				(tex2D(tex_m1, (x)+0.5f, (y)+0.5f))+
				(tex2D(tex_m2, (x)+0.5f, (y)+0.5f))+
				(tex2D(tex_m3, (x)+0.5f, (y)+0.5f));

	// Convert l to dBm:
	l = 20 * log10f(abs(l/4)); 

	if (l < -100)
	{
		optr[offset].x = 255; optr[offset].y = 0; optr[offset].z = 0;
	}
	else if (l < -90)
	{
		optr[offset].x = 0; optr[offset].y = 9; optr[offset].z = 255;
	}
	else if (l < -80)
	{
		optr[offset].x = 255; optr[offset].y = 154; optr[offset].z = 0;
	}
	else if (l < -70)
	{
		optr[offset].x = 255; optr[offset].y = 247; optr[offset].z = 0;
	}
	else
	{
		optr[offset].x = 40; optr[offset].y = 172; optr[offset].z = 7;
	}

	//l += 120; // put l between 0 and 100dBm (offset of 100dBm)
	//l /= 120; // divide by 100 to put between 0 and 1
	//optr[offset].w = 255;

	//if (l < 0)
	//{
	//		optr[offset].x = 10; 
	//		optr[offset].y = 0; 
	//		optr[offset].z = 155;
	//}
	//else if (l < 0.125)
	//	{
	//		optr[offset].x = (unsigned char)(10.0f - 80.0f*l);
	//		optr[offset].y = (unsigned char)(1000.0f * l); 
	//		optr[offset].z = 155;
	//	}
	//	else if (l < 0.375)
	//	{
	//		optr[offset].x = 0; 
	//		optr[offset].y = (unsigned char)(125.0f + (l - 0.125f) * 120.0f); 
	//		optr[offset].z = (unsigned char)(155.0f - (l - 0.125f) * 476.0f);
	//	}
	//	else if (l < 0.625)
	//	{
	//		optr[offset].x = (unsigned char)(820 * (l - 0.375f)); 
	//		optr[offset].y = (unsigned char)(155 + (l - 0.375f)*400.0f); 
	//		optr[offset].z = (unsigned char)(36 - 144 * (l - 0.375f));
	//	}
	//	else if (l < 0.875)
	//	{
	//		optr[offset].x = (unsigned char)(205 + 200 * (l - 0.625f)); 
	//		optr[offset].y = (unsigned char)(255 - 472 * (l - 0.625f)); 
	//		optr[offset].z = 0;
	//	}
	//	else if (l <= 1)
	//	{
	//		optr[offset].x = 255; 
	//		optr[offset].y = (unsigned char)(137 - (l -0.875) * 1096); 
	//		optr[offset].z = 0;
	//	}
	//	else
	//	{
	//		optr[offset].x = 255; 
	//		optr[offset].y = 255; 
	//		optr[offset].z = 255;
	//	}		
}

// Converts amplitudes to dBm to colors (smooth, pixelated)
__global__ void float_to_color_dBm_pixelate( uchar4 *optr,
                         size_t pitch, int ticks ) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * 512;
	int poffset = x + y * pitch/sizeof(float);


	float l =	(tex2D(tex_m0, (blockIdx.x)+0.5f, (blockIdx.y)+0.5f))+
				(tex2D(tex_m1, (blockIdx.x)+0.5f, (blockIdx.y)+0.5f))+
				(tex2D(tex_m2, (blockIdx.x)+0.5f, (blockIdx.y)+0.5f))+
				(tex2D(tex_m3, (blockIdx.x)+0.5f, (blockIdx.y)+0.5f));
	// Convert l to dBm:
	l = 20 * log10f(abs(l/4)); // abs == 0 -l when negative, faster?
	l += 100; // put l between 0 and 100dBm (offset of 100dBm)
	l /= 100; // divide by 100 to put between 0 and 1
	optr[offset].w = 255;

	if (l < 0)
	{
			optr[offset].x = 10; 
			optr[offset].y = 0; 
			optr[offset].z = 155;
	}
	else if (l < 0.125)
		{
			optr[offset].x = (unsigned char)(10.0f - 80.0f*l);
			optr[offset].y = (unsigned char)(1000.0f * l); 
			optr[offset].z = 155;
		}
		else if (l < 0.375)
		{
			optr[offset].x = 0; 
			optr[offset].y = (unsigned char)(125.0f + (l - 0.125f) * 120.0f); 
			optr[offset].z = (unsigned char)(155.0f - (l - 0.125f) * 476.0f);
		}
		else if (l < 0.625)
		{
			optr[offset].x = (unsigned char)(820 * (l - 0.375f)); 
			optr[offset].y = (unsigned char)(155 + (l - 0.375f)*400.0f); 
			optr[offset].z = (unsigned char)(36 - 144 * (l - 0.375f));
		}
		else if (l < 0.875)
		{
			optr[offset].x = (unsigned char)(205 + 200 * (l - 0.625f)); 
			optr[offset].y = (unsigned char)(255 - 472 * (l - 0.625f)); 
			optr[offset].z = 0;
		}
		else if (l <= 1)
		{
			optr[offset].x = 255; 
			optr[offset].y = (unsigned char)(137 - (l -0.875) * 1096); 
			optr[offset].z = 0;
		}
		else
		{
			optr[offset].x = 255; 
			optr[offset].y = 255; 
			optr[offset].z = 255;
		}
}

// Converts amplitudes to color (no dBm) (smooth)
__global__ void float_to_color_pitched( uchar4 *optr, size_t pitch, int ticks, dim3 matdim ) 
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * matdim.x;
	int poffset = x + y * pitch/sizeof(float);
	
	float l = (tex2D(tex_m0, (blockIdx.x)+0.5f, (blockIdx.y)+0.5f));

	if ((x < matdim.x) && (y < matdim.y))
	{
		optr[offset].w = 256;

		l = (l + SRC_MAG)/(2 * SRC_MAG);
		if (l < 0.125)
		{
			optr[offset].x = (unsigned char)(10.0f - 80.0f*l);
			optr[offset].y = (unsigned char)(1000.0f * l); 
			optr[offset].z = 155;
		}
		else if (l < 0.375)
		{
			optr[offset].x = 0; 
			optr[offset].y = (unsigned char)(125.0f + (l - 0.125f) * 120.0f); 
			optr[offset].z = (unsigned char)(155.0f - (l - 0.125f) * 476.0f);
		}
		else if (l < 0.625)
		{
			optr[offset].x = (unsigned char)(820 * (l - 0.375f)); 
			optr[offset].y = (unsigned char)(155 + (l - 0.375f)*400.0f); 
			optr[offset].z = (unsigned char)(36 - 144 * (l - 0.375f));
		}
		else if (l < 0.875)
		{
			optr[offset].x = (unsigned char)(205 + 200 * (l - 0.625f)); 
			optr[offset].y = (unsigned char)(255 - 472 * (l - 0.625f)); 
			optr[offset].z = 0;
		}
		else if (l <= 1)
		{
			optr[offset].x = 255; 
			optr[offset].y = (unsigned char)(137 - (l -0.875) * 1096); 
			optr[offset].z = 0;
		}
		else
		{
			optr[offset].x = 255; 
			optr[offset].y = 255; 
			optr[offset].z = 255;
		}
	}
}


// ~*~*~*~ UTILITIES ~*~*~*~

// For printing out texture values (hardcoded)
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

// Adding and averaging signal power in a cell
__global__ void add_and_average_signal(size_t pitch, int iter, float * avg_m)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int poffset = x + y * pitch/sizeof(float);

	float total=((tex2D(tex_m0, (x)+0.5f, (y)+0.5f))+
				(tex2D(tex_m1, (x)+0.5f, (y)+0.5f))+
				(tex2D(tex_m2, (x)+0.5f, (y)+0.5f))+
				(tex2D(tex_m3, (x)+0.5f, (y)+0.5f)))*0.5;

	total = total*total; // square for power
	
	//if (iter % SAMPLES_TO_AVERAGE)
	if (1)
	{
		float oldavg = tex2D(tex_avg_m, (x)+0.5, (y)+0.5);
		total = oldavg*(SAMPLES_TO_AVERAGE-1) + total;
		avg_m[poffset] = total/SAMPLES_TO_AVERAGE;
	}
	else
	{
		avg_m[poffset] = total;
	}
}


// ~*~*~*~ C++ FUNCTIONS ~*~*~*~
// ~*~*~*~ OPENGL STUFF ~*~*~*~

// Delete display resources
void cPFcaller_display_exit(void)
{
	// Free all allocated memory (move into separate delete function later)
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
}

size_t cPFinitMemories(void)
{
	size_t pitch;

	// Set up Textures and other matrices
	checkCudaErrors(cudaMallocPitch((void**)&dev_m0, &pitch, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMallocPitch((void**)&dev_m1, &pitch, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMallocPitch((void**)&dev_m2, &pitch, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMallocPitch((void**)&dev_m3, &pitch, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMallocPitch((void**)&dev_nm0, &pitch, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMallocPitch((void**)&dev_nm1, &pitch, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMallocPitch((void**)&dev_nm2, &pitch, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMallocPitch((void**)&dev_nm3, &pitch, mWidth*sizeof(float), mHeight));
	checkCudaErrors(cudaMallocPitch((void**)&dev_avg_m, &pitch, mWidth*sizeof(float), mHeight));

	checkCudaErrors(cudaMalloc( (void**)&dev_WWall, WWAL_DIMx*WWAL_DIMy*sizeof(float))); // WWall
	checkCudaErrors(cudaMalloc( (void**)&dev_W, W_DIMx*W_DIMy*sizeof(float))); // W

	checkCudaErrors(cudaMemset2D(dev_m0, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_m1, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_m2, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_m3, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_nm0, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_nm1, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_nm2, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_nm3, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	checkCudaErrors(cudaMemset2D(dev_avg_m, pitch, 0, mWidth*sizeof(float), mHeight));

	checkCudaErrors(cudaMemcpy(dev_WWall, host_WWall, WWAL_DIMx*WWAL_DIMy*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_W, host_W, W_DIMx*W_DIMy*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(cW, host_W, W_DIMx*W_DIMy*sizeof(float), 0U, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc( (void**)&dev_wall, mWidth*mHeight*sizeof(byte)));
	checkCudaErrors(cudaMemcpy(dev_wall, host_Wall, mWidth*mHeight*sizeof(byte), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&dev_src, mWidth*mHeight*sizeof(byte)));
	checkCudaErrors(cudaMemcpy(dev_src, host_src, mWidth*mHeight*sizeof(byte), cudaMemcpyHostToDevice));

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); 
	tex_m0.normalized = false;	tex_m0.filterMode = cudaFilterModeLinear; tex_m0.addressMode[0] = cudaAddressModeBorder;
	tex_m1.normalized = false;	tex_m1.filterMode = cudaFilterModeLinear;tex_m1.addressMode[0] = cudaAddressModeBorder;
	tex_m2.normalized = false;	tex_m2.filterMode = cudaFilterModeLinear;tex_m2.addressMode[0] = cudaAddressModeBorder;
	tex_m3.normalized = false;	tex_m3.filterMode = cudaFilterModeLinear;tex_m3.addressMode[0] = cudaAddressModeBorder;
	tex_nm0.normalized = false;	tex_nm0.filterMode = cudaFilterModeLinear;tex_nm0.addressMode[0] = cudaAddressModeBorder;
	tex_nm1.normalized = false;	tex_nm1.filterMode = cudaFilterModeLinear;tex_nm1.addressMode[0] = cudaAddressModeBorder;
	tex_nm2.normalized = false;	tex_nm2.filterMode = cudaFilterModeLinear;tex_nm2.addressMode[0] = cudaAddressModeBorder;
	tex_nm3.normalized = false;	tex_nm3.filterMode = cudaFilterModeLinear;tex_nm3.addressMode[0] = cudaAddressModeBorder;	
	tex_avg_m.normalized = false;	tex_avg_m.filterMode = cudaFilterModeLinear;tex_avg_m.addressMode[0] = cudaAddressModeBorder;
	
	checkCudaErrors(cudaBindTexture2D(NULL, tex_m0, dev_m0, desc, mWidth, mHeight, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_m1, dev_m1, desc, mWidth, mHeight, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_m2, dev_m2, desc, mWidth, mHeight, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_m3, dev_m3, desc, mWidth, mHeight, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_nm0, dev_nm0, desc, mWidth, mHeight, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_nm1, dev_nm1, desc, mWidth, mHeight, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_nm2, dev_nm2, desc, mWidth, mHeight, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_nm3, dev_nm3, desc, mWidth, mHeight, pitch));
	checkCudaErrors(cudaBindTexture2D(NULL, tex_avg_m, dev_avg_m, desc, mWidth, mHeight, pitch));

	return pitch;
}

// Setting up the display
GPUAnimBitmap bitmap;
void cPFcaller_display(unsigned int num_iterations, float * &m_ptr)
{
	uchar4 * devPtr; 
	size_t  size;
	size_t pitch;
	float source = 0.0f;
	dim3 matdim, threads, grids;

	if ((mWidth < MATRIX_DIM) && (mHeight < MATRIX_DIM))
	{
		bitmap.width = MATRIX_DIM; bitmap.height = MATRIX_DIM; 
	}
	else
	{
		bitmap.width = mWidth; bitmap.height = mHeight;
	}
	bitmap.GPUAnimBitmapSetup();

	matdim.x = mWidth;
	matdim.y = mHeight;
	matdim.z = 1;
	v_threads.x = ((mWidth>THREADx)?THREADx:mWidth);
	v_threads.y = ((mHeight>THREADy)?THREADy:mHeight);
	v_threads.z = 1;
	v_grids.x = ((mWidth + v_threads.x - 1)/ v_threads.x);
	v_grids.y = ((mHeight + v_threads.y - 1)/ v_threads.y);
	v_grids.z = 1;
	//printf("threads %d, %d, grid %d, %d \n", threads.x, threads.y, grids.x, grids.y); 

	pitch = cPFinitMemories();

	//// Set up Textures and other matrices
	//checkCudaErrors(cudaMallocPitch((void**)&dev_m0, &pitch, mWidth*sizeof(float), mHeight));
	//checkCudaErrors(cudaMallocPitch((void**)&dev_m1, &pitch, mWidth*sizeof(float), mHeight));
	//checkCudaErrors(cudaMallocPitch((void**)&dev_m2, &pitch, mWidth*sizeof(float), mHeight));
	//checkCudaErrors(cudaMallocPitch((void**)&dev_m3, &pitch, mWidth*sizeof(float), mHeight));
	//checkCudaErrors(cudaMallocPitch((void**)&dev_nm0, &pitch, mWidth*sizeof(float), mHeight));
	//checkCudaErrors(cudaMallocPitch((void**)&dev_nm1, &pitch, mWidth*sizeof(float), mHeight));
	//checkCudaErrors(cudaMallocPitch((void**)&dev_nm2, &pitch, mWidth*sizeof(float), mHeight));
	//checkCudaErrors(cudaMallocPitch((void**)&dev_nm3, &pitch, mWidth*sizeof(float), mHeight));
	//checkCudaErrors(cudaMallocPitch((void**)&dev_avg_m, &pitch, mWidth*sizeof(float), mHeight));

	//checkCudaErrors(cudaMalloc( (void**)&dev_WWall, WWAL_DIMx*WWAL_DIMy*sizeof(float))); // WWall
	//checkCudaErrors(cudaMalloc( (void**)&dev_W, W_DIMx*W_DIMy*sizeof(float))); // W

	//checkCudaErrors(cudaMemset2D(dev_m0, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	//checkCudaErrors(cudaMemset2D(dev_m1, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	//checkCudaErrors(cudaMemset2D(dev_m2, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	//checkCudaErrors(cudaMemset2D(dev_m3, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	//checkCudaErrors(cudaMemset2D(dev_nm0, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	//checkCudaErrors(cudaMemset2D(dev_nm1, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	//checkCudaErrors(cudaMemset2D(dev_nm2, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	//checkCudaErrors(cudaMemset2D(dev_nm3, pitch, 0, mWidth*sizeof(float), mHeight)); // set 0 to every BYTE
	//checkCudaErrors(cudaMemset2D(dev_avg_m, pitch, 0, mWidth*sizeof(float), mHeight));

	//checkCudaErrors(cudaMemcpy(dev_WWall, host_WWall, WWAL_DIMx*WWAL_DIMy*sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(dev_W, host_W, W_DIMx*W_DIMy*sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyToSymbol(cW, host_W, W_DIMx*W_DIMy*sizeof(float), 0U, cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMalloc( (void**)&dev_wall, mWidth*mHeight*sizeof(byte)));
	//checkCudaErrors(cudaMemcpy(dev_wall, host_Wall, mWidth*mHeight*sizeof(byte), cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMalloc((void**)&dev_src, mWidth*mHeight*sizeof(byte)));
	//checkCudaErrors(cudaMemcpy(dev_src, host_src, mWidth*mHeight*sizeof(byte), cudaMemcpyHostToDevice));

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); 
	//tex_m0.normalized = false;	tex_m0.filterMode = cudaFilterModeLinear; tex_m0.addressMode[0] = cudaAddressModeBorder;
	//tex_m1.normalized = false;	tex_m1.filterMode = cudaFilterModeLinear;tex_m1.addressMode[0] = cudaAddressModeBorder;
	//tex_m2.normalized = false;	tex_m2.filterMode = cudaFilterModeLinear;tex_m2.addressMode[0] = cudaAddressModeBorder;
	//tex_m3.normalized = false;	tex_m3.filterMode = cudaFilterModeLinear;tex_m3.addressMode[0] = cudaAddressModeBorder;
	//tex_nm0.normalized = false;	tex_nm0.filterMode = cudaFilterModeLinear;tex_nm0.addressMode[0] = cudaAddressModeBorder;
	//tex_nm1.normalized = false;	tex_nm1.filterMode = cudaFilterModeLinear;tex_nm1.addressMode[0] = cudaAddressModeBorder;
	//tex_nm2.normalized = false;	tex_nm2.filterMode = cudaFilterModeLinear;tex_nm2.addressMode[0] = cudaAddressModeBorder;
	//tex_nm3.normalized = false;	tex_nm3.filterMode = cudaFilterModeLinear;tex_nm3.addressMode[0] = cudaAddressModeBorder;	
	//tex_avg_m.normalized = false;	tex_avg_m.filterMode = cudaFilterModeLinear;tex_avg_m.addressMode[0] = cudaAddressModeBorder;
	//
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_m0, dev_m0, desc, mWidth, mHeight, pitch));
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_m1, dev_m1, desc, mWidth, mHeight, pitch));
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_m2, dev_m2, desc, mWidth, mHeight, pitch));
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_m3, dev_m3, desc, mWidth, mHeight, pitch));
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_nm0, dev_nm0, desc, mWidth, mHeight, pitch));
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_nm1, dev_nm1, desc, mWidth, mHeight, pitch));
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_nm2, dev_nm2, desc, mWidth, mHeight, pitch));
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_nm3, dev_nm3, desc, mWidth, mHeight, pitch));
	//checkCudaErrors(cudaBindTexture2D(NULL, tex_avg_m, dev_avg_m, desc, mWidth, mHeight, pitch));

	source = 0.0f;
	checkCudaErrors(cudaDeviceSynchronize());

	v_shared_mem_size = 2 * WWAL_DIMx * WWAL_DIMy * sizeof(float) + BLOCK_DIMx*BLOCK_DIMy*4*sizeof(float);
	
	v_p_src_m0 = dev_m0 + src_loc.y * pitch/sizeof(float) + src_loc.x;
	v_p_src_m1 = dev_m1 + src_loc.y * pitch/sizeof(float) + src_loc.x;
	v_p_src_m2 = dev_m2 + src_loc.y * pitch/sizeof(float) + src_loc.x;
	v_p_src_m3 = dev_m3 + src_loc.y * pitch/sizeof(float) + src_loc.x;
		
	cudaStreamCreate(&v_stream1);
	cudaStreamCreate(&v_stream2);
	cudaStreamCreate(&v_stream3);
	cudaStreamCreate(&v_stream4);

	v_pitch = pitch;
	v_matdim.x = matdim.x;
	v_matdim.y = matdim.y;
	v_matdim.z = matdim.z;

	bitmap.anim_and_exit((void(*)(uchar4*,void*,int))cPFcaller_generateFrame, (void(*)(void*))cPFcaller_display_exit);
	bitmap.free_resources();

}

// Generate a frame for OpenGL
void cPFcaller_generateFrame(uchar4 * dispPixels, void*, int ticks)
{
	static int t = 0;
	static float elapsed;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));
	static float source = 0.0f;
	for (int i = 0; i < SAMPLING; i++)
	{
		PF_roundscatter<<<v_grids, v_threads>>>(dev_nm0, dev_nm1, dev_nm2, dev_nm3, v_pitch);

		/*PF_texture_slideright<<<v_grids, v_threads, 0, v_stream1>>>(dev_nm0, v_pitch);
		PF_texture_slideleft<<<v_grids, v_threads, 0, v_stream2>>>(dev_nm1, v_pitch);
		PF_texture_slidedown<<<v_grids, v_threads, 0, v_stream3>>>(dev_nm2, v_pitch);
		PF_texture_slideup<<<v_grids, v_threads, 0, v_stream4>>>(dev_nm3, v_pitch);*/

		//PF_padded_texture_flow<<<v_grids,v_threads,v_shared_mem_size>>>(src_loc, source, dev_wall, dev_nm0, dev_nm1, dev_nm2, dev_nm3, v_matdim, dev_WWall, dev_W, v_pitch);

		cudaDeviceSynchronize();

		source = SRC_MAG * sin(PI * (i+t) * DELTA_LENGTH * SRC_FREQ/CT);

		PF_copy_withWall<<<v_grids,v_threads>>>(dev_m0, dev_m1, dev_m2, dev_m3, dev_wall, v_matdim, v_pitch, dev_src, source);
		add_and_average_signal<<<v_grids, v_threads>>>(v_pitch, i, dev_avg_m);
		cudaDeviceSynchronize();
	}
	//checkCudaErrors(cudaEventRecord(stop, 0));
	//cudaEventSynchronize(stop);
	//
	//cudaEventElapsedTime(&elapsed, start, stop);
#if DISPLAY_FRAME_TIMING
	//printf("Time for per calculation: %3.1f ms \n", elapsed/SAMPLING);
	printf("Time for frame (%d calculations): %3.1f ms \n", SAMPLING, elapsed);
#endif
	t += SAMPLING;
	//printf("matdim %d, %d, %d", v_matdim.x, v_matdim.y, v_matdim.z);

	if (mWidth < MATRIX_DIM)
	{
		//float_to_color_dBm_pixelate<<<colorgrid, colorthreads>>>(dispPixels, v_pitch, ticks);
	}
	else
	{
		//float_to_color_dBm<<<v_grids,v_threads>>>(dispPixels, v_pitch);
		float_to_color_power_dBm<<<v_grids, v_threads>>>(dispPixels, v_pitch, dev_wall, v_matdim);
	}
}

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
	checkCudaErrors(cudaMemcpyToSymbol(cW, host_W, W_DIMx*W_DIMy*sizeof(float), 0U, cudaMemcpyHostToDevice));

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

	

	cudaStream_t stream1, stream2, stream3, stream4;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);

	dim3 stream_threads;
	dim3 stream_blocks;
	stream_threads.x = 1;
	stream_threads.y = 256;
	stream_threads.z = 1;
	stream_blocks.x = 1;
	stream_blocks.y = (MATRIX_DIM + stream_threads.y - 1) /stream_threads.y; //((MATRIX_DIM + BLOCK_DIMy - 1)/BLOCK_DIMy)

	cudaEvent_t start,stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	clock_t t2; t2=clock(); // begin timing

	for (int iter = 0; iter < gpu_iterations; iter++)
	{
		source = src_amplitude * sin(2 * PI * src_frequency * (double)(iter) * 0.01);
		cudaMemcpy(p_src_m0, &source, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(p_src_m1, &source, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(p_src_m2, &source, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(p_src_m3, &source, sizeof(float), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		/*checkCudaErrors(cudaMemcpy(p_src_m0, &source, sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(p_src_m1, &source, sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(p_src_m2, &source, sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(p_src_m3, &source, sizeof(float), cudaMemcpyHostToDevice));
		 
		checkCudaErrors(cudaDeviceSynchronize());*/

		//printf("Calculation \n");
		//PF_padded_texture_flow<<<grids,threads,shared_mem_size>>>(src_loc, source, dev_wall, dev_nm0, dev_nm1, dev_nm2, dev_nm3, matdim, dev_WWall, dev_W, pitch);
		// PF_padded_texture_flow(dim3 srcloc, float src, bool* wallLoc, float*nm0, float*nm1, float* nm2, float* nm3, dim3 matdim, float * WWall, float *W)
		//PF_registers_texture_flow<<<grids,threads, (W_DIMx*W_DIMy*sizeof(float))>>>(dev_nm0, dev_nm1, dev_nm2, dev_nm3, dev_W, pitch);
		//__global__ void PF_registers_texture_flow(float * nm0, float * nm1, float * nm2, float * nm3, float * W, size_t pitch)
		//checkCudaErrors(cudaPeekAtLastError());

		/*PF_texture_slideright<<<stream_blocks, stream_threads, 0, stream1>>>(dev_nm0, pitch);
		PF_texture_slideleft<<<stream_blocks, stream_threads, 0, stream2>>>(dev_nm1, pitch);
		PF_texture_slidedown<<<stream_blocks, stream_threads, 0, stream3>>>(dev_nm2, pitch);
		PF_texture_slideup<<<stream_blocks, stream_threads, 0, stream4>>>(dev_nm3, pitch);*/

		PF_texture_slideright<<<grids, threads, 0, stream1>>>(dev_nm0, pitch);
		PF_texture_slideleft<<<grids, threads, 0, stream2>>>(dev_nm1, pitch);
		PF_texture_slidedown<<<grids, threads, 0, stream3>>>(dev_nm2, pitch);
		PF_texture_slideup<<<grids, threads, 0, stream4>>>(dev_nm3, pitch);

		//checkCudaErrors(cudaDeviceSynchronize());
		cudaDeviceSynchronize();

		/*printf("NM texture values \n");
		testTexturesLoop<<<1,1>>>();
		cudaDeviceSynchronize();*/
		
		PF_padded_texture_copy<<<grids,threads>>>(dev_m0, dev_m1, dev_m2, dev_m3, matdim, pitch);
		cudaDeviceSynchronize();

		/*printf("M texture values \n");
		testTexturesLoop<<<1,1>>>();
		cudaDeviceSynchronize();*/
	}

	cudaEventRecord(stop,0); cudaEventSynchronize(stop);
	float elapsedtime;
	cudaEventElapsedTime(&elapsedtime, start, stop);
	printf("CUDA measured: %3.1f ms \n", elapsedtime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	long int final=clock()-t2; printf("GPU iterations took %li ticks (%f seconds) \n", final, ((float)final)/CLOCKS_PER_SEC);
	
	m_host = (float *)malloc(sizeof(float)*MATRIX_DIM*MATRIX_DIM);
	m_ptr = m_host; // So that the class can access M values
	
	//checkCudaErrors(cudaMemcpy(m_host, dev_m0, MATRIX_DIM*MATRIX_DIM*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy2D(m_host, MATRIX_DIM*sizeof(float), dev_m0, pitch, MATRIX_DIM*sizeof(float), MATRIX_DIM, cudaMemcpyDeviceToHost));
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

using namespace cv; // For the file loading thing

void cPFhandleCommands(void)
{
	char cmd[100];
	bool ret = 0;
	//ShowWindow(GetConsoleWindow(), SW_SHOW);
	SetForegroundWindow(GetConsoleWindow());
	//printf("%d \n ", GetLastError());

	while(!ret)
	{
		printf("\nPlease enter the command ('help' for a list): \n");
		scanf("%s", cmd);

		if (!strcmp(cmd, "load")) 
			cPFload();
		else if (!strcmp(cmd, "save")) 
			cPFsave();
		else if (!strcmp(cmd, "help"))
		{
			printf("\n");
			printf("load [filename]: load an environment file, do not add extension. \n");
			printf("loadmap [filename]: a floorplan, file with a bmp extension. \n");
			printf("save [filename]: save the environment to a file, do not add extension. \n");
			printf("exit: close the application, nothing is saved. \n");
			printf("return: return to simulation. \n");
			printf("cell [dim]: cell dimension in m. \n");
			printf("src [x] [y] [type]: add a source at location (x,y), type is [type] \n");
		}
		else if (!strcmp(cmd, "return"))
		{
			HWND hwnd = FindWindowA(NULL, "PixelFlow");
			SetForegroundWindow(hwnd);
			ret = 1;
		}
	}
}

void cPFload(void)
{
	// Get file name from user
	fflush (stdout);
	printf("Please enter file name (no extension):\n");
	char name[100];
	char dt;
	
	scanf("%s", name);
	sprintf(name, "%s.pf", name);
	// Open file
	FILE * pSaveFile;
	pSaveFile = fopen(name, "r");
	if (pSaveFile == NULL)
	{
		printf("Error opening file. Make sure this file exists. \n"); return;
	}

	// Delete host and device variables!

	// Read in new data

	fscanf(pSaveFile, "%d %d\n", &mWidth, &mHeight);
	if ((mWidth < MATRIX_DIM) && (mHeight < MATRIX_DIM))
	{
		bitmap.width = MATRIX_DIM; bitmap.height = MATRIX_DIM; 
	}
	else
	{
		bitmap.width = mWidth; bitmap.height = mHeight;
	}
	bitmap.GPUAnimBitmap_changeWindow();

	// Initialize host and device variables again
	cPFinitMemories();

	v_threads.x = ((mWidth>THREADx)?THREADx:mWidth);
	v_threads.y = ((mHeight>THREADy)?THREADy:mHeight);
	v_threads.z = 1;
	v_grids.x = ((mWidth + v_threads.x - 1)/ v_threads.x);
	v_grids.y = ((mHeight + v_threads.y - 1)/ v_threads.y);
	v_grids.z = 1;

	v_pitch = pitch;
	v_matdim.x = mWidth;
	v_matdim.y = mHeight;
	v_matdim.z = 1;

	int x, y, vtype;
	fscanf(pSaveFile, "%c", dt);
	// WALLS
	while (dt == 'w')
	{
		fscanf(pSaveFile, "%d %d %d", x, y, vtype); 
		host_Wall[x + y * mWidth] = vtype;
		fscanf(pSaveFile, "%c", dt); 
	}
	// SOURCES
	while (dt == 's')
	{
		fscanf(pSaveFile, "%d %d %d", x, y, vtype);
		host_src[x + y * mWidth] = vtype;
		fscanf(pSaveFile, "%c", dt);
	}
	

	fclose(pSaveFile);
	printf("\nLoad Complete. \n");
}

void cPFsave(void)
{
	// Get file name from user
	//printf("Please enter file name (no extension):\n");
	fflush (stdout);
	//printf("Please enter file name (no extension):\n");
	char name[100];
	char dt;
	//gets(name);
	
	scanf("%s", name);
	sprintf(name, "%s.pf", name);

	// Open file
	FILE * pSaveFile;
	pSaveFile = fopen(name, "w");
	if (pSaveFile == NULL)
	{
		printf("Error opening save file. \n"); return;
	}
	
	// Save dimensions
	fprintf(pSaveFile, "%d %d\n", mWidth, mHeight);

	// Save walls
	for (int y = 0; y < mHeight; y++)
	{
		for (int x = 0; x < mWidth; x++)
		{
			if (host_Wall[x + y * mWidth] == 1)
			{
				fprintf(pSaveFile, "w %d %d %d \n", x, y, 0); // x, y, wall type 
			}
		}
	}

	// Save sources
	for (int y = 0; y < mHeight; y++)
	{
		for (int x = 0; x < mWidth; x++)
		{
			if (host_src[x + y * mWidth] != 0)
			{
				fprintf(pSaveFile, "s %d %d %d \n", x, y, host_src[x + y * mWidth]); // x, y, wall type 
			}
		}
	}

	// Save receivers -- not done yet
	// Close file
	fclose(pSaveFile);
	printf("Save Complete. \n");
}

void cPFinit(float matrixFlow[][4], float matrixWall[][4], float in_sourceLoc[])
{
	// Initialize some values
	coef = 1;

	Mat image;
	image = imread("test5.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	if(! image.data )                              // Check for invalid input
    {
        printf("invalid file :( \n");
        return;
    }
	//namedWindow("Imported environment", WINDOW_AUTOSIZE);
	//imshow("Imported environment", image);
	mWidth = image.cols;
	mHeight = image.rows;
	
	host_Wall = (byte *)malloc(sizeof(byte)*mWidth*mHeight); 
	memset(host_Wall, 0, mWidth*mHeight*sizeof(byte));
	host_src = (byte*) malloc(sizeof(byte)*mWidth*mHeight);
	memset(host_src, 0, mWidth*mHeight*sizeof(byte));

	for (int r = 0; r < image.rows ; r++)
	{
		for (int c = 0; c < image.cols; c++)
		{
			if (image.at<uchar>(r, c) == 0) // 0 is black, 
			{
				host_Wall[c + (mHeight - 1 - r) * mWidth] = 1;
			}
		}
	}	

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
}

// Depreciated
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