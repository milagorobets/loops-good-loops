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

#define DTYPE float

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

byte * host_Wall;
float * host_WWall;
float * host_W;

double coef = 1.0;

int gpu_iterations;

float *m_host;

double src_amplitude;
double src_frequency;
dim3 src_loc;



cudaStream_t v_stream1, v_stream2, v_stream3, v_stream4;
dim3 v_threads(BLOCK_DIMx,BLOCK_DIMy,1);
dim3 v_grids(GRID_DIMx,GRID_DIMy,1);
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


// new stuff:
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

static cudaArray *arr_m0 = NULL; static cudaArray *arr_nm0 = NULL;
static cudaArray *arr_m1 = NULL; static cudaArray *arr_nm1 = NULL;
static cudaArray *arr_m2 = NULL; static cudaArray *arr_nm2 = NULL;
static cudaArray *arr_m3 = NULL; static cudaArray *arr_nm3 = NULL;

static cudaArray * arr_avg_m = NULL;

extern GLUint vbo;
extern struct cudaGraphicsResource * cuda_vbo_resource;

void setupTextures(int x, int y)
{
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

	checkCudaErrors(cudaMallocArray(&arr_m0, &desc, x, y));
	checkCudaErrors(cudaMallocArray(&arr_m1, &desc, x, y));
	checkCudaErrors(cudaMallocArray(&arr_m2, &desc, x, y));
	checkCudaErrors(cudaMallocArray(&arr_m3, &desc, x, y));

	checkCudaErrors(cudaMallocArray(&arr_nm0, &desc, x, y));
	checkCudaErrors(cudaMallocArray(&arr_nm1, &desc, x, y));
	checkCudaErrors(cudaMallocArray(&arr_nm2, &desc, x, y));
	checkCudaErrors(cudaMallocArray(&arr_nm3, &desc, x, y));

	checkCudaErrors(cudaMallocArray(&arr_avg_m, &desc, x, y));
}

void bindTextures(void)
{
	checkCudaErrors(cudaBindTextureToArray(tex_m0, arr_m0));
	checkCudaErrors(cudaBindTextureToArray(tex_m1, arr_m1));
	checkCudaErrors(cudaBindTextureToArray(tex_m2, arr_m2));
	checkCudaErrors(cudaBindTextureToArray(tex_m3, arr_m3));

	checkCudaErrors(cudaBindTextureToArray(tex_nm0, arr_nm0));
	checkCudaErrors(cudaBindTextureToArray(tex_nm1, arr_nm1));
	checkCudaErrors(cudaBindTextureToArray(tex_nm2, arr_nm2));
	checkCudaErrors(cudaBindTextureToArray(tex_nm3, arr_nm3));

	checkCudaErrors(cudaBindTextureToArray(tex_avg_m, arr_avg_m));
}

void unbindTextures(void)
{
	checkCudaErrors(cudaUnbindTexture(tex_m0));
	checkCudaErrors(cudaUnbindTexture(tex_m1));
	checkCudaErrors(cudaUnbindTexture(tex_m2));
	checkCudaErrors(cudaUnbindTexture(tex_m3));

	checkCudaErrors(cudaUnbindTexture(tex_nm0));
	checkCudaErrors(cudaUnbindTexture(tex_nm1));
	checkCudaErrors(cudaUnbindTexture(tex_nm2));
	checkCudaErrors(cudaUnbindTexture(tex_nm3));

	checkCudaErrors(cudaUnbindTexture(tex_avg_m));
}

void updateTexture(DTYPE * data, size_t wib, size_t h, size_t pitch)
{
	checkCudaErrors(cudaMemcpy2DToArray(arr_avg_m, 0, 0, data, pitch, wib, h, cudaMemcpyDeviceToDevice));
}

void deleteTexture(void)
{
	cudaFreeArray(arr_avg_m);
}

void initGL(int wWidth, int wHeight)
{
	int c = 1; char* dummy = "";
	glutInit(&c, &dummy);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(wWidth, wHeight);
	glutCreateWindow("PixelFlow on a GPU");
	glutDisplayFunc(display);
	glutDisplayFunc(keyboard);
	glutMouseFunc(click);
	glutMotionFunc(motion);

	glewInit();
}

void cPFcaller_fast_display(unsigned int num_iterations, float * &m_ptr)
{
	int devID;
	cudaDeviceProp deviceProps;

	// Initialize OpenGL
	initGL(matX, matY);

	memset(&deviceProps, 0, sizeof(cudaDeviceProp));
    deviceProps.major = 1; deviceProps.minor = 0;
	checkCudaErrors(cudaChooseDevice(&devID, &deviceProps));
	checkCudaErrors(cudaGLSetGLDevice(devID));

	// Initialize various arrays here

	// Set up OpenGL buffers
	glGenBuffersARB(1, &vbo);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(DTYPE) * matX * matY, 

}