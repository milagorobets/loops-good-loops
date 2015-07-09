#include "stdafx.h"
#include "Form1.h"

namespace PFCUDA_GUI{
	
	Form1::Form1(void)
	{
		InitializeComponent();
	}

	System::Windows::Forms::Panel ^ Form1::returnpanel()
	{
		return this -> pViewer;
	}

}