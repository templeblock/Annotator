#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"
#include "VideoDecode.h"
#include "SvgIcons.h"
#include "LabelIO.h"
#include "UI.h"

void xoMain(xo::SysWnd* wnd) {
	using namespace imqs::anno;
	VideoFile::Initialize();

	wnd->Doc()->ClassParse("font-medium", "font-size: 14ep");
	wnd->Doc()->ClassParse("shortcut", "font-size: 15ep; color: #000; width: 1em");

	svg::LoadAll(wnd->Doc());
	wnd->SetPosition(xo::Box(0, 0, 1500, 970), xo::SysWnd::SetPosition_Size);

	auto ui = new UI(&wnd->Doc()->Root);

	ui->SaveThread = std::thread(UI::SaveThreadFunc, ui);

	ui->Classes.push_back({'U', "unlabeled"}); // first class must be unlabeled - as embodied by UI::UnlabeledClass
	ui->Classes.push_back({'R', "normal road"});
	ui->Classes.push_back({'C', "crocodile cracks"});
	ui->Classes.push_back({'B', "bricks"});
	ui->Classes.push_back({'P', "pothole"});
	ui->Classes.push_back({'S', "straight crack"});
	ui->Classes.push_back({'M', "manhole cover"});
	ui->Classes.push_back({'X', "pockmarks"});

	ui->VideoFilename = "D:\\mldata\\GOPR0080.MP4";
	if (!ui->OpenVideo())
		ui->Render();
}
