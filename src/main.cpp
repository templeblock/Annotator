#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"
#include "VideoDecode.h"
#include "SvgIcons.h"

namespace imqs {
namespace anno {

// This captures our UI state, and maintains the DOM state
class UI {
public:
	enum class PlayStates {
		Stop,
		Play
	} PlayState = PlayStates::Stop;

	xo::DomNode*   Root          = nullptr;
	xo::DomNode*   TimeSliderBox = nullptr;
	xo::DomCanvas* VideoCanvas   = nullptr;
	int64_t        PlayTimer     = 0;
	int64_t        OnDestroyEv   = 0;
	std::string    VideoFilename;
	VideoFile      Video;

	UI(xo::DomNode* root) {
		Root = root;
		Render();
	}

	~UI() {
	}

	void Render() {
		auto ui = this;
		Root->Clear();
		auto videoArea = Root->ParseAppendNode("<div style='break:after; box-sizing: margin; width: 100%; margin: 10ep'></div>");
		VideoCanvas    = videoArea->AddCanvas();
		VideoCanvas->StyleParse("hcenter: hcenter; border: 1px #888; border-radius: 1.5ep;");
		VideoCanvas->SetSize(1200, 700);

		TimeSliderBox = Root->ParseAppendNode("<div></div>");
		RenderTimeSlider(true);

		auto mediaControlBox = Root->ParseAppendNode("<div style='break:after; padding: 5ep'></div>");
		auto btnW            = "20ep";
		auto btnH            = "20ep";
		auto btnPlay         = xo::controls::Button::NewSvg(mediaControlBox, "media-play", btnW, btnH);
		auto btnPause        = xo::controls::Button::NewSvg(mediaControlBox, "media-pause", btnW, btnH);
		auto btnStepB        = xo::controls::Button::NewSvg(mediaControlBox, "media-step-backward", btnW, btnH);
		auto btnStepF        = xo::controls::Button::NewSvg(mediaControlBox, "media-step-forward", btnW, btnH);

		btnPlay->OnClick([ui] {
			ui->Play();
		});
		btnPause->OnClick([ui] {
			ui->Stop();
		});
		btnStepF->OnClick([ui] {
			ui->Stop();
			ui->NextFrame();
		});

		// Setup once-off things
		if (OnDestroyEv == 0) {
			OnDestroyEv = Root->OnDestroy([ui] {
				delete ui;
			});

			Root->OnKeyChar([ui](xo::Event& ev) {
				switch (ev.KeyChar) {
				case '.': ui->NextFrame(); break;
				case ',': ui->NextFrame(); break;
				}
			});
		}
	}

	void RenderTimeSlider(bool first = false) {
		if (!Video.IsOpen())
			return;

		TimeSliderBox->Clear();
		TimeSliderBox->StyleParse("break:after; margin: 16ep 8ep; padding: 0ep; box-sizing: margin; width: 100%; height: 34ep; border-radius: 3ep; background: #eee");
		TimeSliderBox->StyleParse("");
		auto caret = TimeSliderBox->ParseAppendNode("<div style='background: #ccc; border: 1px #888; width: 5ep; height: 120%; vcenter: vcenter; border-radius: 2.5ep'></div>");

		if (first) {
			TimeSliderBox->OnMouseDown([this]() {
				TimeSliderBox->SetCapture();
			});

			TimeSliderBox->OnMouseUp([this]() {
				TimeSliderBox->ReleaseCapture();
			});

			TimeSliderBox->OnMouseMove([this](xo::Event& ev) {
				if (ev.IsPressed(xo::Button::MouseLeft)) {
					auto tester = (xo::DomNode*) TimeSliderBox->ChildByIndex(0);
					auto r = ev.LayoutResult->Node(TimeSliderBox);
					float mouseX = ev.PointsRel[0].x / r->ContentWidthPx(); // percentage
					mouseX = xo::Clamp(mouseX, 0.f, 1.f);
					tester->StyleParsef("hcenter: %.1f%%", mouseX * 100.f);
				}
			});
		}

		auto info = Video.GetVideoStreamInfo();
		double dur = info.DurationSeconds();
		double pos = Video.LastFrameTimeSeconds();
		
		caret->StyleParsef("hcenter: %.1f%%", pos * 100.0 / dur);
	}

	void Play() {
		if (PlayState == PlayStates::Play)
			return;
		PlayState = PlayStates::Play;
		VideoCanvas->OnTimer([this](xo::Event& ev) {
			if (this->PlayState == PlayStates::Play)
				this->NextFrame();
		},
		                     16);
	}

	void Stop() {
		if (PlayState == PlayStates::Stop)
			return;
		PlayState = PlayStates::Stop;
		VideoCanvas->RemoveHandler(PlayTimer);
		PlayTimer = 0;
	}

	void NextFrame() {
		if (!Video.IsOpen())
			return;
		auto cx = VideoCanvas->GetCanvas2D();
		Video.DecodeFrameRGBA(cx->Width(), cx->Height(), cx->RowPtr(0), cx->Stride());
		cx->Invalidate();
		VideoCanvas->ReleaseCanvas(cx);
		RenderTimeSlider();
	}

	bool OpenVideo() {
		if (Video.GetFilename() == VideoFilename)
			return true;
		Video.Close();

		auto err = Video.OpenFile(VideoFilename);
		if (!err.OK())
			xo::controls::MsgBox::Show(Root->GetDoc(), err.Message());
		return err.OK();
	}
};
}
}

void xoMain(xo::SysWnd* wnd) {
	using namespace imqs::anno;
	VideoFile::Initialize();

	svg::LoadAll(wnd->Doc());
	wnd->SetPosition(xo::Box(0, 0, 1300, 840), xo::SysWnd::SetPosition_Size);

	auto ui           = new UI(&wnd->Doc()->Root);
	ui->VideoFilename = "D:\\mldata\\GOPR0080.MP4";
	ui->OpenVideo();
	ui->Render();
	ui->NextFrame();
}
