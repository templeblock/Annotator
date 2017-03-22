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
	auto      err = video->OpenFile("D:\\mldata\\GOPR0080.MP4");
	if (!err.OK())
		return;

	canvas->OnTimer([video, canvas](xo::Event& ev) {
		//ev.CancelTimer();
		//imqs::anno::TestDecode(canvas);

		AVFrame* frame;
		auto     err = video->DecodeNextFrame(frame);
		if (err.OK()) {
			auto   cx = canvas->GetCanvas2D();
			size_t w  = std::min<size_t>(cx->Width(), frame->width);
			size_t h  = std::min<size_t>(cx->Height(), frame->height);
			for (size_t y = 0; y < h; y++) {
				const uint8_t* src = frame->data[0];
				src += y * (size_t) frame->linesize[0];
				uint8_t* dst = (uint8_t*) cx->RowPtr((int) y);
				for (size_t x = 0; x < w; x++) {
					uint8_t v = src[0];
					dst[0] = v;
					dst[1] = v;
					dst[2] = v;
					dst[3] = 255;
					src++;
					dst += 4;
				}
			}
			cx->Invalidate();
			canvas->ReleaseCanvas(cx);
		}

	},
	                16);
}
