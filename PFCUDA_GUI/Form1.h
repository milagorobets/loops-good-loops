#include "tlm2d.cuh"
#include "renderer.h"

#pragma once

namespace PFCUDA_GUI {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for Form1
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:
		static OpenGLRenderer::CRender^ render;
		System::Windows::Forms::Panel ^ Form1::returnpanel();
		Form1(void);
		//Form1(void)
		//{
		//	InitializeComponent();
		//	//
		//	//TODO: Add the constructor code here
		//	//
		//}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Panel^  pViewer;
	private: System::Windows::Forms::Button^  btnTest;
	private: System::Windows::Forms::Label^  lblTest;
	private: System::Windows::Forms::TextBox^  txtMessages;
	protected: 

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->pViewer = (gcnew System::Windows::Forms::Panel());
			this->btnTest = (gcnew System::Windows::Forms::Button());
			this->lblTest = (gcnew System::Windows::Forms::Label());
			this->txtMessages = (gcnew System::Windows::Forms::TextBox());
			this->SuspendLayout();
			// 
			// pViewer
			// 
			this->pViewer->BackColor = System::Drawing::SystemColors::ActiveCaption;
			this->pViewer->Location = System::Drawing::Point(13, 13);
			this->pViewer->Name = L"pViewer";
			this->pViewer->Size = System::Drawing::Size(737, 684);
			this->pViewer->TabIndex = 0;
			// 
			// btnTest
			// 
			this->btnTest->Location = System::Drawing::Point(775, 26);
			this->btnTest->Name = L"btnTest";
			this->btnTest->Size = System::Drawing::Size(75, 23);
			this->btnTest->TabIndex = 1;
			this->btnTest->Text = L"button1";
			this->btnTest->UseVisualStyleBackColor = true;
			this->btnTest->Click += gcnew System::EventHandler(this, &Form1::btnTest_Click);
			// 
			// lblTest
			// 
			this->lblTest->AutoSize = true;
			this->lblTest->Location = System::Drawing::Point(798, 66);
			this->lblTest->Name = L"lblTest";
			this->lblTest->Size = System::Drawing::Size(35, 13);
			this->lblTest->TabIndex = 2;
			this->lblTest->Text = L"label1";
			// 
			// txtMessages
			// 
			this->txtMessages->Location = System::Drawing::Point(784, 83);
			this->txtMessages->Name = L"txtMessages";
			this->txtMessages->Size = System::Drawing::Size(100, 20);
			this->txtMessages->TabIndex = 3;
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::SystemColors::ControlLightLight;
			this->ClientSize = System::Drawing::Size(1114, 709);
			this->Controls->Add(this->txtMessages);
			this->Controls->Add(this->lblTest);
			this->Controls->Add(this->btnTest);
			this->Controls->Add(this->pViewer);
			this->Name = L"Form1";
			this->Text = L"Wave-O-Rama";
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void btnTest_Click(System::Object^  sender, System::EventArgs^  e) {
				 //addWithCuda(int *c, const int *a, const int *b, unsigned int size);
				 this-> Focus();
				 int c[5] = {};
				 int a[5] = {0, 1, 2, 3, 4};
				 int b[5] = {10, 11, 12, 13, 14};
				 addWithCuda(c, a, b, 5);
				 this->lblTest->Text = Convert::ToString(c[0]) + " " + Convert::ToString(c[1]) + " " +
										Convert::ToString(c[2]) + " " + Convert::ToString(c[3]) + " " +
										Convert::ToString(c[4]);
				 //this->txtMessages->AppendText("hello%d", a[1]);
				 //printf("hello");
			 }
	};
}

