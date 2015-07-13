#include "cuda_runtime.h"
#include "utils.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    // Allocate GPU buffers for three vectors (two input, one output)    .
	checkCudaErrors(cudaMalloc((void**)&dev_c, size * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dev_a, size * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&dev_b, size * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    checkCudaErrors(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

	Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

size_t pitch;
texture<float, 2, cudaReadModeElementType> tex_env;
float * dev_env;
dim3 envdim;
dim3 threads, blocks;

__global__ void animKernel(uchar4 *optr, size_t pitch, float * env, dim3 matdim)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * matdim.x;

	//optr[offset].x = x%255; optr[offset].y = y%255; optr[offset].z = 255-x%255;
	
	if (x < matdim.x && y < matdim.y)
	{
		//optr[offset].w = 255;
		//optr[offset].x = 0; optr[offset].y = 0; optr[offset].z = 128;
		optr[offset].x = (x/2)%255; optr[offset].y = (y/2)%255; optr[offset].z = 255 - (y/2)%255;
		//optr[offset].w = 255;*/
		//printf("hello %d %d %d %d\n", x, y,offset, optr[offset].x);
	}
}

__device__ struct cuComplex
{
	float r;
	float i;
	__device__ cuComplex( float a, float b ) : r(a), i(b)  {}
	__device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
	__device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia(int x, int y, dim3 matdim)
{
	const float scale = 1.5;
	float jx = scale * (float)(matdim.x/2 - x)/(matdim.x/2);
	float jy = scale * (float)(matdim.y/2 - y)/(matdim.y/2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	int i = 0;
	/*if (x > matdim.x/2)
	{
		return 1;
	}*/
	for (i = 0; i < 200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000) return 0;
	}
	

	return 1;
}

__global__ void fractals(uchar4 *optr, size_t pitch, float *env, dim3 matdim)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * matdim.x;
	int poffset = x + y * pitch/sizeof(float);

	if (x < matdim.x && y < matdim.y)
	{
		int juliaValue = julia(x,y,matdim);
		optr[offset].x = 255*juliaValue; optr[offset].y = 0; optr[offset].z = 0;
		//if (x > matdim.x/2)
		//{
		//	optr[offset].x = 220; optr[offset].y = 25; optr[offset].z = 0;
		//}
		
	}
	
	
	//optr[offset].x = 0; optr[offset].y = 0; optr[offset].z = 0;
}

void prepAnimation(int width, int height)
{
	checkCudaErrors(cudaMallocPitch((void**)&dev_env, &pitch, width*sizeof(float), height));
	checkCudaErrors(cudaMemset2D(dev_env, pitch, 0, width*sizeof(float), height));
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	tex_env.normalized = false; tex_env.filterMode = cudaFilterModeLinear; tex_env.addressMode[0] = cudaAddressModeBorder;
	checkCudaErrors(cudaBindTexture2D(NULL, tex_env, dev_env, desc, width, height, pitch));
	envdim.x = width; envdim.y = height;

	threads.x = 16; threads.y = 16; threads.z = 1;
	blocks.x = (threads.x + width - 1)/threads.x;
	blocks.y = (threads.y + height - 1)/threads.y;
	blocks.z = 1;
}

void animate(uchar4* dispPtr)
{
	checkCudaErrors(cudaDeviceSynchronize());
	//printf("blocks: %d, %d, %d; threads: %d %d %d \n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
	//animKernel<<<blocks, threads>>>(dispPtr, pitch, dev_env, envdim);
	dim3 t; t.x = 500; t.y = 500; t.z = 1;
	animKernel<<<t,1>>>(dispPtr, pitch, dev_env, envdim);
	//printf("dim %d x %d x %d \n", envdim.x, envdim.y, envdim.z);
	//fractals<<<blocks,threads>>>(dispPtr, pitch, dev_env, envdim);
	checkCudaErrors(cudaDeviceSynchronize());
}