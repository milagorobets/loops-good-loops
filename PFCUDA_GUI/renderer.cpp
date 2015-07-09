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

	// Init environment here

	cp -> X = 0;
	cp -> Y = 0;
	cp -> Height = iHeight;
	cp -> Width = iWidth;

	//// Get ptr to openGL panel on the form
	System::Windows::Forms::Panel ^ viewPanel = f -> returnpanel();

	cp -> Parent = viewPanel -> Handle;
	viewPanel -> Focus();

	cp -> Style = WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_DISABLED;

	//// Create the window
	this -> CreateHandle(cp);

	m_hDC = GetDC((HWND)this-> Handle.ToPointer());

	fWidth = iWidth; fHeight = iHeight;
	//cudaDeviceReset();
	//cudaDeviceSynchronize();

	/*int co = 1; char* dummy = "";

    glutInit( &co, &dummy );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( fWidth, fHeight );*/

	//glutCreateWindow( "PixelFlow" );

	//cudaDeviceProp prop; int dev;
	//memset(&prop, 0, sizeof(cudaDeviceProp));
	//prop.major = 1; prop.minor = 0;

	////cudaSetDevice(0);
	////cudaDeviceSynchronize();

	//checkCudaErrors(cudaChooseDevice(&dev, &prop));
	//checkCudaErrors(cudaGLSetGLDevice(dev));

	if (m_hDC)
	{
		MySetPixelFormat(m_hDC);
		int status = ReleaseDC((HWND)this->Handle.ToPointer(), m_hDC);
		if (status)
		{
			ReSizeGLScene(iWidth, iHeight);
			InitGL();
		}
	}
	return 0;
}

bool OpenGLRenderer::CRender::InitGL(GLvoid)
{
	cudaDeviceProp prop; int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1; prop.minor = 0;


	//cudaDeviceSynchronize();

	checkCudaErrors(cudaChooseDevice(&dev, &prop));
	checkCudaErrors(cudaGLSetGLDevice(dev));

	glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");


	pin_ptr<GLuint> pin_bufferObj = &bufferObj;
	glGenBuffers(1, pin_bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, fWidth * fHeight * 4,
                  NULL, GL_DYNAMIC_DRAW_ARB );

	pin_ptr<cudaGraphicsResource*> pin_resource = &resource;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer( pin_resource, bufferObj, cudaGraphicsMapFlagsNone ));

	SwapBuffers(m_hDC);
	//cudaDeviceSynchronize();

	//cudaSetDevice(0);

	 int c[5] = {};
	 int a[5] = {0, 1, 2, 3, 4};
	 int b[5] = {10, 11, 12, 13, 14};
	 addWithCuda(c, a, b, 5);
	 printf("%d %d %d %d %d \n", c[0], c[1], c[2], c[3], c[4]);
				 

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
				16,											// Select Our Color Depth
				0, 0, 0, 0, 0, 0,							// Color Bits Ignored
				0,											// No Alpha Buffer
				0,											// Shift Bit Ignored
				0,											// No Accumulation Buffer
				0, 0, 0, 0,									// Accumulation Bits Ignored
				16,											// 16Bit Z-Buffer (Depth Buffer)  
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

	glViewport(0,0,iWidth,iHeight);						// Reset The Current Viewport

	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();		
			
	// Calculate The Aspect Ratio Of The Window
	gluPerspective(60.0f,778.0f/728.0f,0.1f,500.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();									// Reset The Modelview Matrix
}