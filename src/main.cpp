#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"
#include "VideoDecode.h"

void xoMain(xo::SysWnd* wnd) {

	imqs::anno::VideoFile::Initialize();

	auto root = &wnd->Doc()->Root;
	auto canvas = root->AddCanvas();
	canvas->SetSize(800, 450);
	canvas->OnTimer([canvas](xo::Event& ev) {
		ev.CancelTimer();
		imqs::anno::TestDecode(canvas);
	}, 0);
}
