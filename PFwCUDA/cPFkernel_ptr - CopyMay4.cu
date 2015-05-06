// TO DO:
// 1. coalesce memory accesses for m and nm
// 2. put FP_ptr_copy into pointer form
// 3. split kernels into edges and middle blocks (middle do not if checks in flow) -- started

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cPFkernel_ptr.cuh"
#include "utils.h"
#include "common.h"

#define SoF sizeof(float)
#define CI(x,y,z,width,height) ((x) + (y)*(width) + (z) * (height) * (width))

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

//__device__ size_t pitch;
//__device__ size_t e_per_row;
//__device__ size_t slice_pitch;
//__device__ size_t one_sp, two_sp, three_sp;
//
//__global__ void PF_setup_globals(size_t ptrpitch, cudaExtent mExt, dim3 matdim)
//{
//	pitch = ptrpitch;
//	e_per_row = pitch/SoF;
//	slice_pitch = e_per_row * matdim.y;
//	one_sp = slice_pitch * 1.0f;
//	two_sp = slice_pitch * 2.0f;
//	three_sp = slice_pitch * 3.0f;
//}

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

void cPFsetupDisplay(void)
{
	
}

void cPFcaller(unsigned int num_iterations, float * &m_ptr)
{
	gpu_iterations = num_iterations;
	cudaError_t status = cudaSuccess;
	dim3 matdim;
	matdim.x = MATRIX_DIM;
	matdim.y = MATRIX_DIM;
	matdim.z = 4;

	dim3 threads(BLOCK_DIMx,BLOCK_DIMy,1);
	dim3 grids(GRID_DIMx,GRID_DIMy,1);

	// Allocate 3D array for m0-m3 (all together)
	cudaExtent m_extent = make_cudaExtent(sizeof(float)*matdim.x, matdim.y, matdim.z); // width, height, depth
	cudaPitchedPtr m_device;
	cudaMalloc3D(&m_device, m_extent);

	m_host = (float *)malloc(sizeof(float)*MATRIX_DIM*MATRIX_DIM*4); // need to initialize this somehow
	//memset(m_host, 0, sizeof(float)*MATRIX_DIM*MATRIX_DIM*4); // set all to 0  -- do it on gpu cudamemset
	m_ptr = m_host; // So that the class can access M values

	cudaMemset3D(m_device, 0, m_extent);

	// Allocate 3D array for nm0-nm3
	cudaExtent nm_extent = make_cudaExtent(sizeof(float)*matdim.x, matdim.y, matdim.z);
	cudaPitchedPtr nm_device;
	cudaMalloc3D(&nm_device, nm_extent); // don't need to init to 0 'cause we will just overwrite it anyways
	cudaMemset3D(nm_device, 0, nm_extent);
	
	// Allocate 2D array for wall (unrolled to 1D)
	bool * dev_wall;
	status = cudaMalloc((void**)&dev_wall, matdim.x*matdim.y*sizeof(bool)); // x*y elements in a 1D array
	if (status != cudaSuccess){printf("Wall cudaMalloc: %s \n", cudaGetErrorString(status));}
	// copy wall locations
	status = cudaMemcpy(dev_wall, host_Wall, matdim.x*matdim.y*sizeof(bool), cudaMemcpyHostToDevice);
	if (status != cudaSuccess){printf("Wall cudaMemcpy: %s \n", cudaGetErrorString(status));}

	// Allocate and initialize arrays for WWall and W
	float * dev_WWall;
	float * dev_W;
	status = cudaMalloc((void**)&dev_WWall, WWAL_DIMx*WWAL_DIMy*sizeof(float));
	if (status != cudaSuccess){printf("WWall cudaMalloc: %s \n", cudaGetErrorString(status));}
	status = cudaMalloc((void**)&dev_W, W_DIMx*W_DIMy*sizeof(float));
	if (status != cudaSuccess){printf("W cudaMalloc: %s \n", cudaGetErrorString(status));}

	status = cudaMemcpy(dev_WWall, host_WWall, WWAL_DIMx*WWAL_DIMy*sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess){printf("WWall cudaMemcpy: %s \n", cudaGetErrorString(status));}
	status = cudaMemcpy(dev_W, host_W, W_DIMx*W_DIMy*sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess){printf("W cudaMemcpy: %s \n", cudaGetErrorString(status));}
	double source = 0;

	cudaMemcpy3DParms hm_p = {0};
	int hx = m_extent.width/sizeof(float);
	int hy = m_extent.height;
	int hz = m_extent.depth;

	hm_p.srcPtr.ptr = m_device.ptr;
	hm_p.srcPtr.pitch = m_device.pitch;
	hm_p.srcPtr.xsize = hx;
	hm_p.srcPtr.ysize = hy;
	hm_p.dstPtr.ptr = ((void**)m_host);
	hm_p.dstPtr.pitch = hx *sizeof(float);
	hm_p.dstPtr.xsize = hx;
	hm_p.dstPtr.ysize = hy;
	hm_p.extent.width = hx * sizeof(float);
	hm_p.extent.height = hy;
	hm_p.extent.depth = hz;
	hm_p.kind = cudaMemcpyDeviceToHost;

	cudaDeviceSynchronize();
	//PF_setup_globals<<<1,1>>>(m_device.pitch, m_extent, matdim);
	//__global__ void PF_setup_globals(size_t ptrpitch, cudaExtent mExt, dim3 matdim)
	clock_t t2; t2=clock();
	for (int iter = 0; iter < gpu_iterations; iter++)
	{
		source = src_amplitude * sin(2 * PI * src_frequency * (double)(iter) * 0.01);
		PF_ptr_flow<<<grids,threads,2*16*sizeof(float)>>>(m_device, m_extent, matdim, source,
							src_loc, dev_wall, dev_WWall, dev_W, nm_device);
		cudaDeviceSynchronize();
		PF_ptr_copy<<<grids,threads>>>(m_device, nm_device, m_extent, matdim);
		cudaDeviceSynchronize();
		//status = cudaMemcpy3D(&hm_p);
	}	
	long int final=clock()-t2; printf("GPU iterations took %li ticks (%f seconds) \n", final, ((float)final)/CLOCKS_PER_SEC);
	
	status = cudaMemcpy3D(&hm_p);

	if (status != cudaSuccess){printf("Uhoh: %s \n", cudaGetErrorString(status));}

	// Free all allocated memory (move into separate delete function later)
	cudaFree(m_device.ptr);
	cudaFree(nm_device.ptr);
	cudaFree(dev_wall);
	cudaFree(dev_WWall);
	cudaFree(dev_W);
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
	/*if (host_W != NULL) */free(host_W);
	/*if (host_WWall != NULL) */free(host_WWall);
	/*if (host_Wall != NULL) */free(host_Wall);
	free(m_host);
}