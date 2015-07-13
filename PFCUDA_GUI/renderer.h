#ifndef H_RENDERER
#define H_RENDERER

//#pragma comment(lib, "opengl32.lib")
//#pragma comment(lib, "glu32.lib")

#pragma once

#include <Windows.h>
#include "cuda.h"
#include "gl_helper.h"
#include "cuda_gl_interop.h"
#include <gl/GL.h>
#include <gl/GLU.h>

namespace PFCUDA_GUI{ref class Form1;}

namespace OpenGLRenderer
{
	using namespace System::Windows::Forms;

	public ref class CRender: public System::Windows::Forms::NativeWindow
	{
	public:
		static PFCUDA_GUI::Form1^ f;
		CRender();
		OpenGLRenderer::CRender::CRender(	System::Windows::Forms::Panel ^ parentForm, 
											GLsizei iWidth,
											GLsizei iHeight);
		int CRender_init(	GLsizei iWidth,
											GLsizei iHeight);

		int DrawFrame(void);
		void blankscreen(void);
		GLvoid KILLGLWindow(GLvoid);

	private:
		HDC m_hDC;
		HGLRC m_hglrc;

		int fWidth, fHeight;

		GLuint bufferObj;

		void (*fAnim)(uchar4*, void*, int);
		void (*animExit)(void*);
		cudaGraphicsResource *resource;

		PFNGLBINDBUFFERARBPROC    glBindBuffer     ;
		PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  ;
		PFNGLGENBUFFERSARBPROC    glGenBuffers     ;
		PFNGLBUFFERDATAARBPROC    glBufferData     ;

	protected:
		~CRender(System::Void)
		{
			this->DestroyHandle();
		}
		bool InitGL(GLvoid);
		GLvoid ReSizeGLScene(GLsizei iWidth, GLsizei iHeight);
		GLint MySetPixelFormat(HDC hdc);
	};


}


#endif