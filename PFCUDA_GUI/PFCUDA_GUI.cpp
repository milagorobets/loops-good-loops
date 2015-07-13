// PFCUDA_GUI.cpp : main project file.

#include "stdafx.h"
#include "Form1.h"

namespace PFCUDA_GUI{ref class Form1;}
namespace OpenGLRenderer{ref class CRender;}

using namespace PFCUDA_GUI;

[STAThreadAttribute]
int main(array<System::String ^> ^args)
{
	//// Enabling Windows XP visual effects before any controls are created
	//Application::EnableVisualStyles();
	//Application::SetCompatibleTextRenderingDefault(false); 

	//// Create the main window and run it
	//Application::Run(gcnew Form1());
	//return 0;

	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);

	Form1 ^ mainform = gcnew Form1();
	OpenGLRenderer::CRender ^ renderframe = gcnew OpenGLRenderer::CRender();

	mainform -> render = renderframe;
	renderframe -> f = mainform;

	//mainform -> render -> CRender_init(500,500);
	
	Application::Run(mainform);

	return 0;
}
