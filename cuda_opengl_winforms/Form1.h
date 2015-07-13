#pragma once

#include "cuda.h"
#include "kernel.cuh"

namespace cuda_opengl_winforms {

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
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

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
	protected: 
	private: System::Windows::Forms::Button^  btnCudaAction;
	private: System::Windows::Forms::TextBox^  txtMessages;

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
			this->btnCudaAction = (gcnew System::Windows::Forms::Button());
			this->txtMessages = (gcnew System::Windows::Forms::TextBox());
			this->SuspendLayout();
			// 
			// pViewer
			// 
			this->pViewer->Location = System::Drawing::Point(13, 13);
			this->pViewer->Name = L"pViewer";
			this->pViewer->Size = System::Drawing::Size(285, 315);
			this->pViewer->TabIndex = 0;
			// 
			// btnCudaAction
			// 
			this->btnCudaAction->Location = System::Drawing::Point(304, 13);
			this->btnCudaAction->Name = L"btnCudaAction";
			this->btnCudaAction->Size = System::Drawing::Size(100, 23);
			this->btnCudaAction->TabIndex = 1;
			this->btnCudaAction->Text = L"GO CUDA";
			this->btnCudaAction->UseVisualStyleBackColor = true;
			this->btnCudaAction->Click += gcnew System::EventHandler(this, &Form1::btnCudaAction_Click);
			// 
			// txtMessages
			// 
			this->txtMessages->Location = System::Drawing::Point(304, 42);
			this->txtMessages->Multiline = true;
			this->txtMessages->Name = L"txtMessages";
			this->txtMessages->Size = System::Drawing::Size(100, 286);
			this->txtMessages->TabIndex = 2;
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(412, 340);
			this->Controls->Add(this->txtMessages);
			this->Controls->Add(this->btnCudaAction);
			this->Controls->Add(this->pViewer);
			this->Name = L"Form1";
			this->Text = L"CUDA OpenGL in WinForms!";
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void btnCudaAction_Click(System::Object^  sender, System::EventArgs^  e) {
				 this-> Focus();
				 int c[5] = {};
				 int a[5] = {0, 1, 2, 3, 4};
				 int b[5] = {10, 11, 12, 13, 14};
				 addWithCuda(c, a, b, 5);
				 this->txtMessages->AppendText(Convert::ToString(c[0]) + " " + Convert::ToString(c[1]) + " " +
										Convert::ToString(c[2]) + " " + Convert::ToString(c[3]) + " " +
										Convert::ToString(c[4]) + "\n");
			 }
	};
}

