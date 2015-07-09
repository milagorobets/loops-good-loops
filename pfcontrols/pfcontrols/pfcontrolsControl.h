#pragma once

using namespace System;
using namespace System::ComponentModel;
using namespace System::Collections;
using namespace System::Windows::Forms;
using namespace System::Data;
using namespace System::Drawing;


namespace pfcontrols {

	/// <summary>
	/// Summary for pfcontrolsControl
	/// </summary>
	public ref class pfcontrolsControl : public System::Windows::Forms::UserControl
	{
	public:
		pfcontrolsControl(void)
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
		~pfcontrolsControl()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TextBox^  txtSrcX;
	protected: 
	private: System::Windows::Forms::Button^  btnAddSrc;

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->txtSrcX = (gcnew System::Windows::Forms::TextBox());
			this->btnAddSrc = (gcnew System::Windows::Forms::Button());
			this->SuspendLayout();
			// 
			// txtSrcX
			// 
			this->txtSrcX->Location = System::Drawing::Point(15, 16);
			this->txtSrcX->Name = L"txtSrcX";
			this->txtSrcX->Size = System::Drawing::Size(100, 20);
			this->txtSrcX->TabIndex = 0;
			// 
			// btnAddSrc
			// 
			this->btnAddSrc->Location = System::Drawing::Point(136, 14);
			this->btnAddSrc->Name = L"btnAddSrc";
			this->btnAddSrc->Size = System::Drawing::Size(75, 23);
			this->btnAddSrc->TabIndex = 1;
			this->btnAddSrc->Text = L"Add Source";
			this->btnAddSrc->UseVisualStyleBackColor = true;
			// 
			// pfcontrolsControl
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->Controls->Add(this->btnAddSrc);
			this->Controls->Add(this->txtSrcX);
			this->Name = L"pfcontrolsControl";
			this->Size = System::Drawing::Size(535, 540);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	};
}
