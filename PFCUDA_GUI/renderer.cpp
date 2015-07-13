#include "stdafx.h"
#include "cuda.h"
#include "utils.h"
#include "Form1.h"

// Default constructor:
OpenGLRenderer::CRender::CRender(){};

// Constructor 2
OpenGLRenderer::CRender::CRender(	System::Windows::Forms::Panel ^ parentForm, 
											GLsizei iWidth,
											GLsizei iHeight)
{}

// Constructor 3
int OpenGLRenderer::CRender::CRender_init(	GLsizei iWidth,
									GLsizei iHeight)
{
	CreateParams^ cp = gcnew CreateParams;

	//// Init environment here

	cp -> X = 0;
	cp -> Y = 0;
	cp -> Height = iHeight;
	cp -> Width = iWidth;

	////// Get ptr to openGL panel on the form
	System::Windows::Forms::Panel ^ viewPanel = f -> returnpanel();

	cp -> Parent = viewPanel -> Handle;
	viewPanel -> Focus();

	cp -> Style = WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS |  WS_CLIPCHILDREN;

	////// Create the window
	this -> CreateHandle(cp);

	m_hDC = GetDC((HWND)this-> Handle.ToPointer());

	fWidth = iWidth; fHeight = iHeight;

	if (m_hDC)
	{
		//MySetPixelFormat(m_hDC);
		//ReSizeGLScene(iWidth, iHeight);

		//glShadeModel(GL_SMOOTH);
		//glClearDepth(1.0f);

		//glEnable(GL_DEPTH_TEST);
		//glDepthFunc(GL_LEQUAL);
		//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);*/
			
		//UpdateWindow((HWND)this-> Handle.ToPointer());
		

		//int status = ReleaseDC((HWND)this->Handle.ToPointer(), m_hDC);
		int status = 1;
		if (status)
		{
			
			InitGL();
		}
	}
	return 0;
}

void OpenGLRenderer::CRender::blankscreen(void)
{
	//glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
	//SwapBuffers(m_hDC);
}

bool OpenGLRenderer::CRender::InitGL(GLvoid)
{
	
	cudaDeviceProp prop; int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1; prop.minor = 0;


	//cudaDeviceSynchronize();

	checkCudaErrors(cudaChooseDevice(&dev, &prop));
	checkCudaErrors(cudaGLSetGLDevice(dev));

	MySetPixelFormat(m_hDC);
		ReSizeGLScene(fWidth, fHeight);

		//glShadeModel(GL_SMOOTH);
		glClearDepth(1.0f);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);

	prepAnimation(fWidth, fHeight);

	glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");
	
	GLuint imagePBO; // OpenGL name for the buffer
	cudaGraphicsResource * cudaResourceBuf; // CUDA name for the buffer

	glGenBuffers(1, &imagePBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, imagePBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, fWidth*fHeight*4, NULL, GL_DYNAMIC_DRAW);	
	//

	checkCudaErrors(cudaGraphicsGLRegisterBuffer( &cudaResourceBuf, imagePBO, cudaGraphicsMapFlagsNone ));

	uchar4* devPtr;
	size_t size;	

	uchar4* host_devPtr;
	host_devPtr = (uchar4 *)malloc(sizeof(uchar4)*fWidth*fHeight); 
	memset(host_devPtr, 0, fWidth*fHeight*sizeof(float));
	
	if((wglMakeCurrent(m_hDC, m_hglrc)) == NULL)
	{
		MessageBox::Show("wglMakeCurrent Failed");
		return 0;
	}

	// copy avg back to CPU
	//checkCudaErrors(cudaMemcpy2D(tempAvg, mWidth*sizeof(float), dev_avg_m, v_pitch, mWidth*sizeof(float), mHeight, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaGraphicsMapResources(1, &cudaResourceBuf, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResourceBuf));

	//glClearColor(1.0f, 0.5f, 0.0f, 0.0f);
	//glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	//glLoadIdentity();
	//SwapBuffers(m_hDC);

	animate(devPtr);
	//animate(devPtr);	

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResourceBuf, NULL));
	
	glDrawPixels(fWidth, fHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	SwapBuffers(m_hDC);
		
	checkCudaErrors(cudaMemcpy2D(host_devPtr, fWidth*sizeof(uchar4), devPtr, fWidth*sizeof(uchar4), 
								fWidth*sizeof(uchar4), fHeight, cudaMemcpyDeviceToHost));
	uchar4* test_data = (uchar4 *)malloc(sizeof(uchar4)*10*10);
	glReadPixels(0,0,10,10,GL_RED, GL_UNSIGNED_BYTE, test_data);
	for (int i = 0; i < 10; i++)
	{
		printf("CUDA memory i %d val %u \n", i, host_devPtr[i].x);
	}
	for (int i = 0; i < 10; i++)
	{
		printf("glReadPixels i %d val %u, %u, %u \n", i, test_data[i].x, test_data[i].y, test_data[i].z);
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	//SwapBuffers(m_hDC);
	/*animate(devPtr);
	SwapBuffers(m_hDC);
	animate(devPtr);
	SwapBuffers(m_hDC);
	animate(devPtr);*/
	//SwapBuffers(m_hDC);
	//cudaDeviceSynchronize();

	//glClearColor(1.0f, 1.0f, 1.0f, 1.5f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glLoadIdentity();
	////SwapBuffers(m_hDC);
	//animate(devPtr);
	//SwapBuffers(m_hDC);
	
	/*checkCudaErrors(cudaGraphicsUnmapResources(1, &resourceID, NULL));*/
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferID);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	//glFlush();
	//SwapBuffers(m_hDC);
	//SwapBuffers(m_hDC);
	

	//glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	//glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	//
	//checkCudaErrors(cudaGraphicsMapResources(1, &resourceID, NULL));
	//checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resourceID));

	//animate(devPtr);
	//
	//checkCudaErrors(cudaGraphicsUnmapResources(1, &resourceID, NULL));
	//SwapBuffers(m_hDC);
	//glDrawPixels(fWidth, fHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	
	//blankscreen();

	 /*int c[5] = {};
	 int a[5] = {0, 1, 2, 3, 4};
	 int b[5] = {10, 11, 12, 13, 14};
	 addWithCuda(c, a, b, 5);
	 printf("%d %d %d %d %d \n", c[0], c[1], c[2], c[3], c[4]);*/
				 

	return TRUE;
}

GLint OpenGLRenderer::CRender::MySetPixelFormat(HDC hdc)
{
	static	PIXELFORMATDESCRIPTOR pfd=				// pfd Tells Windows How We Want Things To Be
			{
				sizeof(PIXELFORMATDESCRIPTOR),				// Size Of This Pixel Format Descriptor
				1,											// Version Number
				PFD_DRAW_TO_WINDOW |						// Format Must Support Window
				PFD_SUPPORT_OPENGL |						// Format Must Support OpenGL
				PFD_DOUBLEBUFFER,							// Must Support Double Buffering
				PFD_TYPE_RGBA,								// Request An RGBA Format
				8,											// Select Our Color Depth
				0, 0, 0, 0, 0, 0,							// Color Bits Ignored
				0,											// No Alpha Buffer
				0,											// Shift Bit Ignored
				0,											// No Accumulation Buffer
				0, 0, 0, 0,									// Accumulation Bits Ignored
				32,											// 16Bit Z-Buffer (Depth Buffer)  
				0,											// No Stencil Buffer
				0,											// No Auxiliary Buffer
				PFD_MAIN_PLANE,								// Main Drawing Layer
				0,											// Reserved
				0, 0, 0										// Layer Masks Ignored
			};
			
		GLint  iPixelFormat; 
		 
		// get the device context's best, available pixel format match 
		if((iPixelFormat = ChoosePixelFormat(hdc, &pfd)) == 0)
		{
			MessageBox::Show("ChoosePixelFormat Failed");
			return 0;
		}
			 
		// make that match the device context's current pixel format 
		if(SetPixelFormat(hdc, iPixelFormat, &pfd) == FALSE)
		{
			MessageBox::Show("SetPixelFormat Failed");
			return 0;
		}

		if((m_hglrc = wglCreateContext(m_hDC)) == NULL)
		{
			MessageBox::Show("wglCreateContext Failed");
			return 0;
		}

		if((wglMakeCurrent(m_hDC, m_hglrc)) == NULL)
		{
			MessageBox::Show("wglMakeCurrent Failed");
			return 0;
		}		
		


		return 1;
}

int OpenGLRenderer::CRender::DrawFrame(void)
{
	return 1;
}

GLvoid  OpenGLRenderer::CRender::KILLGLWindow(GLvoid)
{
	
}

GLvoid  OpenGLRenderer::CRender::ReSizeGLScene(GLsizei iWidth, GLsizei iHeight)
{
	if (iHeight==0)										// Prevent A Divide By Zero By
	{
		iHeight=1;										// Making Height Equal One
	}

	glViewport(0, 0, iWidth, iHeight);						// Reset The Current Viewport

	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();
	//gluPerspective(60.0f,778.0f/728.0f,0.1f,500.0f);
	////		
	//glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix

	glEnable(GL_DEPTH_TEST);
	//glClearColor(0.0f, 1.0f, 1.0f, 0.5f);
	
}