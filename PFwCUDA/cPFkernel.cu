#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include <stdio.h>
#include "cPFkernel.cuh"
#include "utils.h"
#include "common.h"

#define SoF sizeof(float)
#define CI(x,y,z,width,height) ((x) + (y)*(width) + (z) * (height) * (width))

//__constant__ double src_amplitude = 1.0;
//__constant__ double src_frequency = 1.0;
__device__ unsigned int threads_report;

__global__ void PF_copymem_kernel(cudaPitchedPtr mPtr, cudaPitchedPtr nmPtr, cudaExtent mExt, dim3 matdim)
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
		 

	 /*__syncthreads();
	 printf("Location %d, %d, m0 = %f \n", x, y, m[CI(x, y, 2, e_per_row, matrix_dimensions.y)]);*/
	 //__syncthreads();
	}
}

__global__ void PF_iteration_kernel(cudaPitchedPtr mPtr, cudaExtent mExt, dim3 matrix_dimensions, 
									double src, dim3 srcloc, bool * wallLoc, float * WWall, float * W,
									cudaPitchedPtr nmPtr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//if ((x == 0) && (y == 0)) threads_report = 0;
		//if (x > MATRIX_DIM)
		//{
		//	printf("hello %d %d \n", x, y);
		//}
	if ((x < MATRIX_DIM) && (y < MATRIX_DIM))
	{		
	//__syncthreads();
	//if (x == 3){
	//	printf("hello, y = %d \n", y);
	//}
	//printf("Hello from thread %d, %d \n", x, y);

	//// Find location within the pitched memory
	float *m = (float*)mPtr.ptr;
	float *nm = (float*)nmPtr.ptr;
	
	//int sof = sizeof(float);
	size_t pitch = mPtr.pitch;
	unsigned int e_per_row = pitch / SoF;
	//size_t slice_pitch = pitch*mExt.height;
	///*float src = src_amplitude * sin(2 * PI * src_frequency * (double)(t) * 0.01);*/
	//char* m_addroff = (char*)(m + y * pitch + x * SoF);
	//printf("m(%d,%d) is %f \n", x, y, *(float*)(m_addroff)); 
	//*(float*)(m_addroff) = 1;
	//char* m1_addroff = m_addroff + 1 * slice_pitch; // Run kernel during init to set these up?
	//char* nm_addroff = (char*)(nm + y * pitch + x * SoF);
	float m0 = m[CI(x, y, 0, e_per_row, matrix_dimensions.y)];
	//m_ptr += slice_pitch; // just inc m_ptr by slice_pitch
	//float m1 = *(float*)(m_ptr);
	float m1 = m[CI(x, y, 1, e_per_row, matrix_dimensions.y)];
	//m_ptr += slice_pitch;
	//float m2 = *(float*)(m_ptr);
	float m2 = m[CI(x, y, 2, e_per_row, matrix_dimensions.y)];
	//m_ptr += slice_pitch;
	//float m3 = *(float*)(m_ptr);
	float m3 = m[CI(x, y, 3, e_per_row, matrix_dimensions.y)];

	float newF[4] = {0};
	//size_t row_w = pitch;

	// Check if source
	if ((x == srcloc.x) && (y == srcloc.y))
	{
		m0 = src; m1 = src; m2 = src; m3 = src;
	}

	// Check if wall
	bool isWall = wallLoc[x + y*matrix_dimensions.x];
	if (isWall)
	{
		// prefetch WWall into __shared__
		newF[0] = WWall[0]*m0 + WWall[1]*m1 + WWall[2]*m2 + WWall[3]*m3;
		newF[1] = WWall[4]*m0 + WWall[5]*m1 + WWall[6]*m2 + WWall[7]*m3;
		newF[2] = WWall[8]*m0 + WWall[9]*m1 + WWall[10]*m2 + WWall[11]*m3;
		newF[3] = WWall[12]*m0 + WWall[13]*m1 + WWall[14]*m2 + WWall[15]*m3;
	}
	else
	{
		// prefetch W into __shared__
		newF[0] = W[0]*m0 + W[1]*m1 + W[2]*m2 + W[3]*m3;
		newF[1] = W[4]*m0 + W[5]*m1 + W[6]*m2 + W[7]*m3;
		newF[2] = W[8]*m0 + W[9]*m1 + W[10]*m2 + W[11]*m3;
		newF[3] = W[12]*m0 + W[13]*m1 + W[14]*m2 + W[15]*m3;
	}

	//printf("newF %f, %f, %f, %f \n", newF[0], newF[1], newF[2], newF[3]);

	//if (x < MATRIX_DIM-1) *(float*)(nm_addroff + sof) = newF[1];					// if (x < MATRIX_DIM-1) nm0[x+1][y] = newF[1];
	//if (x > 0) *(float*)(nm_addroff - sof + slice_pitch) = newF[0];					// if (x > 0) nm1[x-1][y] = newF[0];
	//if (y < MATRIX_DIM-1) *(float*)(nm_addroff + pitch + 2 * slice_pitch) = newF[3];		// if (y < MATRIX_DIM-1) nm2[x][y+1] = newF[3];
	//if (y > 0) *(float*)(nm_addroff - pitch + 3*slice_pitch) = newF[2];						// if (y > 0) nm3[x][y-1] = newF[2];

	if (x < MATRIX_DIM-1) nm[CI(x + 1, y, 0, e_per_row, matrix_dimensions.y)] = newF[1];					// if (x < MATRIX_DIM-1) nm0[x+1][y] = newF[1];
	if (x > 0) nm[CI(x - 1, y, 1, e_per_row, matrix_dimensions.y)] = newF[0];					// if (x > 0) nm1[x-1][y] = newF[0];
	if (y < MATRIX_DIM-1) 
	{// access conflict?
		nm[CI(x, y + 1, 2, e_per_row, matrix_dimensions.y)] = newF[3];		// if (y < MATRIX_DIM-1) nm2[x][y+1] = newF[3];
		//printf("F3 write to nm[%d], x %d y %d:%f \n",CI(x, y + 1, 2, e_per_row, matrix_dimensions.y),x,y,newF[3]);
	}
	if (y > 0) 
	{
		nm[CI(x, y - 1, 3, e_per_row, matrix_dimensions.y)] = newF[2];						// if (y > 0) nm3[x][y-1] = newF[2];
	}

	}
}

#define WWAL_DIMx 4
#define WWAL_DIMy WWAL_DIMx
#define W_DIMx 4
#define W_DIMy W_DIMx

#define BLOCK_DIMx ((MATRIX_DIM>32)?32:MATRIX_DIM) // vary this
#define BLOCK_DIMy  BLOCK_DIMx
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
	//clock_t t1; t1=clock();
	for (int iter = 0; iter < gpu_iterations; iter++)
	{
		source = src_amplitude * sin(2 * PI * src_frequency * (double)(iter) * 0.01);
		PF_iteration_kernel<<<grids,threads>>>(m_device, m_extent, matdim, source, src_loc, dev_wall, dev_WWall, dev_W, nm_device);
		cudaDeviceSynchronize();
		PF_copymem_kernel<<<grids,threads>>>(m_device, nm_device, m_extent, matdim);

		cudaDeviceSynchronize();
		//status = cudaMemcpy3D(&hm_p);
	}	
	//long int final=clock()-t1; printf("GPU took %li ticks (%f seconds) \n", final, ((float)final)/CLOCKS_PER_SEC);
	
	status = cudaMemcpy3D(&hm_p);

	if (status != cudaSuccess){printf("Uhoh: %s \n", cudaGetErrorString(status));}

	// Free all allocated memory (move into separate delete function later)
	cudaFree(m_device.ptr);
	cudaFree(nm_device.ptr);
	cudaFree(dev_wall);
	cudaFree(dev_WWall);
	cudaFree(dev_W);
}

void cPFinit(double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[])
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

__global__ void testKernel(int a, int b, int *c)
{
	*c = a+b;
	int i = threadIdx.x;
	printf("hello from thread %d \n", i);
}

void callerblahblah(void)
{
	int c;
	int *dev_c;
	int dev;
	//HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
	cudaMalloc((void**)&dev_c, sizeof(int));
	testKernel<<<1,1>>>(2,7,dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	printf("2+7 = %d \n",c);
	cudaFree(dev_c);

	//PF_iteration_kernel<<<1,1>>>();
}