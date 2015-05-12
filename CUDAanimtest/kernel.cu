/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <glew.h>
#include <freeglut.h>
#include "book.h"
#include "gpu_anim.h"

#define DIM 5
#define THREADx 16
#define THREADy 16
#define BLOCK_DIMx ((DIM>THREADx)?THREADx:DIM) // vary this
#define BLOCK_DIMy ((DIM>THREADy)?THREADy:DIM)
#define GRID_DIMx ((DIM + BLOCK_DIMx - 1)/BLOCK_DIMx)
#define GRID_DIMy ((DIM + BLOCK_DIMy - 1)/BLOCK_DIMy)

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

texture<float, 2, cudaReadModeElementType> tex;
float * dev_mat;
float * host_mat;

__global__ void testTextures(void)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//int loc = x + y * DIM;

	if ((x < DIM) && (y < DIM))
	{
		printf("Thread %d, %d reporting %f \n", x, y, (float)(tex2D(tex, (float)(x)+0.5f, (float)(y)+0.5f)));
	}

}

__global__ void testTexturesLoop(void)
{
	for (int x = 0; x < DIM; x++)
	{
		for (int y = 0; y < DIM; y++)
		{
			printf("Texture location %d, %d reporting %f \n", x, y, (float)(tex2D(tex, (float)(x)+0.5f, (float)(y)+0.5f)));
		}
	}
}
__global__ void testTexturesLoop2(void)
{
	for (int x = 0; x < DIM; x++)
	{
		for (int y = 0; y < DIM; y++)
		{
			printf("%f, ",(float)(tex2D(tex, (float)(x)+0.5f, (float)(y)+0.5f)));
		}
		printf("\n");
	}
}

__global__ void testDeviceMem(float* dev, size_t pitch)
{
		for (int x = 0; x < DIM; x++)
	{
		for (int y = 0; y < DIM; y++)
		{
			printf("%d, %d: Value was: %f ", x, y, dev[x+y*pitch/sizeof(float)]);
			/*dev[x+y*pitch/sizeof(float)] = ((float)(y + x * DIM+5))/(float)(100);*/
			dev[x+y*pitch/sizeof(float)] = 0.5f*0.5f;
			printf(", now is %f", dev[x+y*pitch/sizeof(float)]);
			printf(", texture %f\n", (float)(tex2D(tex, (float)(x)+0.5f, (float)(y)+0.5f)));
		}
	}
}

GLuint rtexture;
GLuint rbuffer;

void setupDisplay(void)
{
	int dev;
	width = 512; height = 512;
	cudaDeviceProp prop;
	memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1; prop.minor = 0;
	checkCudaErrors(cudaChooseDevice(&dev, &prop));
	checkCudaErrors(cudaGLSetGLDevice(dev));

	// texture buffer
	glGenTextures(1, &rtexture);
	glBindTexture(GL_TEXTURE_2D, rtexture);

	// what happens if we try to fetch outside of the texture? well:
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // for coordinate S
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // for coordinate T
	// how many pixels of the texture are blended if the pixel is mapped to an area that is:
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // larger than a single texture element (no filtering)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); // smaller than a single texture element (no filtering)

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0); // unbind texture for future use

	// pixel buffer
	glGenBuffers(1, &rbuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, rbuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 
}

int main(void)
{
	dim3 threads(BLOCK_DIMx,BLOCK_DIMy,1);
	dim3 grids(GRID_DIMx,GRID_DIMy,1);

	size_t pitch;
	size_t host_size = DIM * DIM * sizeof(float);
	checkCudaErrors(cudaMallocPitch((void**)&dev_mat, &pitch, DIM*sizeof(float), DIM));
	checkCudaErrors(cudaMallocHost((void**)&host_mat, host_size,cudaHostAllocDefault));

	for (int y = 0; y < DIM; y++)
	{
		for (int x = 0; x < DIM; x++)
		{
			host_mat[x+y*DIM] = x + y*DIM; // populate with some values
			printf("Host: x = %d, y = %d, value = %f \n", x, y, host_mat[x+y*DIM]);
		}
	}

	//checkCudaErrors(cudaMemcpy(dev_mat, host_mat, DIM*DIM*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(dev_mat, pitch, host_mat, DIM*sizeof(float), DIM*sizeof(float), DIM, cudaMemcpyHostToDevice));

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); 

	tex.normalized = false;	tex.filterMode = cudaFilterModeLinear;
	checkCudaErrors(cudaBindTexture2D(NULL, tex, dev_mat, desc, DIM, DIM, pitch));
	
	checkCudaErrors(cudaDeviceSynchronize());
	float source = 66.0f;
	//float * ptrsrcloc = &dev_mat[3 + 3*DIM];
	float *ptrsrcloc = dev_mat + 3 * pitch/sizeof(float) + 3;
	checkCudaErrors(cudaMemcpy(ptrsrcloc, &source, sizeof(float), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy2D(ptrsrcloc, sizeof(float), &source, sizeof(float), 
	////testTextures<<<threads,grids>>>();
	//testTexturesLoop<<<1,1>>>();
	//testDeviceMem<<<1,1>>>(dev_mat);
	cudaDeviceSynchronize();
	printf("OLD:\n");
	testTexturesLoop2<<<1,1>>>();
	cudaDeviceSynchronize();
	testDeviceMem<<<1,1>>>(dev_mat, pitch);
	cudaDeviceSynchronize();
	printf("NEW: \n");
	testTexturesLoop2<<<1,1>>>();
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaUnbindTexture(tex));
	checkCudaErrors(cudaFree(dev_mat));
	checkCudaErrors(cudaFreeHost(host_mat));
	return 0;
}
