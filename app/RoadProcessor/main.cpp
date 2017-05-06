#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"

void xoMain(xo::SysWnd* wnd) {
	using namespace imqs::anno;
	VideoFile::Initialize();

	auto root = &wnd->Doc()->Root;
	root->ParseAppend("Hello!");
}
