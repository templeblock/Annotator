#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"
#include "VideoDecode.h"

void xoMain(xo::SysWnd* wnd) {
	using namespace imqs::anno;
	VideoFile::Initialize();

	auto root   = &wnd->Doc()->Root;
	auto canvas = root->AddCanvas();
	canvas->SetSize(1200, 700);

	VideoFile* video = new VideoFile();
	auto       err   = video->OpenFile("D:\\mldata\\GOPR0080.MP4");
	if (!err.OK())
		return;

	canvas->OnTimer([video, canvas](xo::Event& ev) {
		auto cx = canvas->GetCanvas2D();
		video->DecodeFrameRGBA(cx->Width(), cx->Height(), cx->RowPtr(0), cx->Stride());
		cx->Invalidate();
		canvas->ReleaseCanvas(cx);
	},
	                16);

	canvas->OnDestroy([video] {
		delete video;
	});
}
