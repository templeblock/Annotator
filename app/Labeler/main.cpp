#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"
#include "SvgIcons.h"
#include "UI.h"

void xoMain(xo::SysWnd* wnd) {
	using namespace imqs::anno;
	imqs::video::VideoFile::Initialize();

	wnd->Doc()->ClassParse("font-medium", "font-size: 14ep");
	wnd->Doc()->ClassParse("shortcut", "font-size: 15ep; color: #000; width: 1em");

	svg::LoadAll(wnd->Doc());
	wnd->SetPosition(xo::Box(0, 0, 1600, 1020), xo::SysWnd::SetPosition_Size);

	auto ui = new UI(&wnd->Doc()->Root);

#ifdef IMQS_AI_API
	auto err         = ui->Model.Load("c:\\mldata\\cp\\model.cntk");
	ui->ModelLoadErr = err.Message();
#endif

	ui->Classes.push_back({'U', "unlabeled"}); // first class must be unlabeled - as embodied by UI::UnlabeledClass
	ui->Classes.push_back({'R', "normal road"});
	ui->Classes.push_back({'C', "crocodile cracks"});
	ui->Classes.push_back({'B', "bricks"});
	ui->Classes.push_back({'P', "pothole"});
	ui->Classes.push_back({'S', "straight crack"});
	ui->Classes.push_back({'M', "manhole cover"});
	ui->Classes.push_back({'X', "pockmarks"});
	ui->Classes.push_back({'E', "road edge"});

	ui->VideoFilename = "c:\\mldata\\GOPR0080.MP4";
	//ui->VideoFilename = "C:\\mldata\\StellenboschFuji\\4K-F2\\DSCF1008.MOV";
	if (!ui->OpenVideo())
		ui->Render();
}
