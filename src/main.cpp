#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"
#include "VideoDecode.h"
#include "SvgIcons.h"
#include "LabelIO.h"

namespace imqs {
namespace anno {

// This captures our UI state, and maintains the DOM state
class UI {
public:
	enum class PlayStates {
		Stop,
		Play
	} PlayState = PlayStates::Stop;

	enum class ActionStates {
		None,
		AssignLabel,
	} ActionState = ActionStates::None;

	xo::DomNode*   Root          = nullptr;
	xo::DomNode*   TimeSliderBox = nullptr;
	xo::DomCanvas* VideoCanvas   = nullptr;
	xo::DomNode*   PlayBtn       = nullptr;
	int64_t        PlayTimer     = 0;
	int64_t        OnDestroyEv   = 0;
	std::string    VideoFilename;
	VideoFile      Video;

	std::vector<LabelClass> Classes;
	VideoLabels             Labels;
	int                     LabelGridSize = 128;
	bool                    GridTopDown   = false; // For road markings, we prefer bottom up, because the interesting stuff is at the bottom of the frame

	UI(xo::DomNode* root) {
		Root = root;
		Render();
	}

	~UI() {
	}

	void Render() {
		auto ui = this;
		Root->Clear();

		auto   videoArea  = Root->ParseAppendNode("<div style='break:after; box-sizing: margin; width: 100%; margin: 10ep'></div>");
		double aspect     = 1920.0 / 1080.0;
		double videoWidth = 1400;

		VideoCanvas = videoArea->AddCanvas();
		VideoCanvas->StyleParse("hcenter: hcenter; border: 1px #888; border-radius: 1.5ep;");
		VideoCanvas->SetSize((int) videoWidth, (int) (videoWidth / aspect));
		VideoCanvas->OnMouseMove([ui](xo::Event& ev) { ui->OnCanvasMouseMove(ev); });

		auto fileArea = videoArea->ParseAppendNode("<div style='break:after; margin: 4ep'></div>");
		fileArea->StyleParsef("hcenter: hcenter; width: %vpx", videoWidth);
		auto fileName = fileArea->ParseAppendNode("<div style='cursor: hand'></div>");
		fileName->AddClass("font-medium");
		fileName->AddText(VideoFilename);
		fileName->OnClick([this] {
			std::vector<std::pair<std::string, std::string>> types = {
			    {"Video Files", "*.mp4;*.avi;*.mkv"},
			};
			if (xo::osdialogs::BrowseForFileOpen(Root->GetDoc(), types, VideoFilename))
				OpenVideo();
		});

		TimeSliderBox = Root->ParseAppendNode("<div></div>");
		RenderTimeSlider(true);

		auto bottomControlBoxes = Root->ParseAppendNode("<div style='padding: 5ep'></div>");
		auto mediaControlBox    = bottomControlBoxes->ParseAppendNode("<div></div>");
		auto btnW               = "20ep";
		auto btnH               = "20ep";
		PlayBtn                 = xo::controls::Button::NewSvg(mediaControlBox, "media-play", btnW, btnH);
		//auto btnStepB = xo::controls::Button::NewSvg(mediaControlBox, "media-step-backward", btnW, btnH);
		auto btnStepF = xo::controls::Button::NewSvg(mediaControlBox, "media-step-forward", btnW, btnH);
		//auto btnTest  = xo::controls::Button::NewText(mediaControlBox, "test");

		PlayBtn->OnClick([ui] {
			ui->FlipPlayState();
		});
		btnStepF->OnClick([ui] {
			ui->Stop();
			ui->NextFrame();
		});
		/*
		btnTest->OnClick([ui] {
			ui->Labels.Frames.clear();
			ui->Labels.Frames.push_back(VideoLabels::Frame());
			auto& f = ui->Labels.Frames[0];
			f.Time  = 12345;
			f.Labels.Labels.push_back(Label());
			f.Labels.Labels.back().Class = "crack";
			f.Labels.Labels.back().Rect  = Rect(3, 4, 5, 6);
			SaveFrameLabels(ui->VideoFilename, ui->Labels.Frames[0]);
		});
		*/

		auto        labelGroup      = bottomControlBoxes->ParseAppendNode("<div style='padding-left: 20ep; color: #333'></div>");
		std::string shortcutsTop    = "";
		std::string shortcutsBottom = "";
		auto        addShortcut     = [](std::string& shortcuts, const char* key, const char* title) {
            shortcuts += tsf::fmt("<div style='width: 10em'><span class='shortcut'>%v</span>%v</div>", key, title);
        };
		size_t half = Classes.size() / 2;
		for (size_t i = 0; i < half; i++)
			addShortcut(shortcutsTop, Classes[i].KeyStr().c_str(), Classes[i].Class.c_str());
		for (size_t i = half; i < Classes.size(); i++)
			addShortcut(shortcutsBottom, Classes[i].KeyStr().c_str(), Classes[i].Class.c_str());
		labelGroup->ParseAppend("<div style='break:after'>" + shortcutsTop + "</div>");
		labelGroup->ParseAppend("<div style='break:after'>" + shortcutsBottom + "</div>");

		// Setup once-off things
		if (OnDestroyEv == 0) {
			OnDestroyEv = Root->OnDestroy([ui] {
				delete ui;
			});

			Root->OnKeyChar([ui](xo::Event& ev) { ui->OnKeyChar(ev); });
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
			TimeSliderBox->OnMouseDown([this](xo::Event& ev) {
				TimeSliderBox->SetCapture();
				SeekFromUI(ev);
			});

			TimeSliderBox->OnMouseUp([this]() {
				TimeSliderBox->ReleaseCapture();
			});

			TimeSliderBox->OnMouseMove([this](xo::Event& ev) {
				if (ev.IsPressed(xo::Button::MouseLeft))
					SeekFromUI(ev);
			});
		}

		auto   info = Video.GetVideoStreamInfo();
		double dur  = info.DurationSeconds();
		double pos  = Video.LastFrameTimeSeconds();

		caret->StyleParsef("hcenter: %.1f%%", pos * 100.0 / dur);
	}

	void FlipPlayState() {
		if (PlayState == PlayStates::Play)
			Stop();
		else
			Play();
	}

	void SeekFromUI(xo::Event& ev) {
		auto  box    = ev.LayoutResult->Node(TimeSliderBox);
		float mouseX = ev.PointsRel[0].x / box->ContentWidthPx(); // percentage
		mouseX       = xo::Clamp(mouseX, 0.f, 1.f);
		Video.SeekToFraction(mouseX);
		// I've been too lazy to understand why we need to burn a frame after seeking.
		NextFrame();
		NextFrame();
	}

	void Play() {
		if (PlayState == PlayStates::Play)
			return;
		xo::controls::Button::SetSvg(PlayBtn, "media-pause");
		PlayState = PlayStates::Play;
		PlayTimer = VideoCanvas->OnTimer([this](xo::Event& ev) {
			if (this->PlayState == PlayStates::Play)
				this->NextFrame();
		},
		                                 16);
	}

	void Stop() {
		if (PlayState == PlayStates::Stop)
			return;
		xo::controls::Button::SetSvg(PlayBtn, "media-play");
		PlayState = PlayStates::Stop;
		VideoCanvas->RemoveHandler(PlayTimer);
		PlayTimer = 0;
	}

	void DrawLabelBoxes() {
		auto cx = VideoCanvas->GetCanvas2D();
		int vwidth, vheight;
		Video.Dimensions(vwidth, vheight);
		auto  vinfo = Video.GetVideoStreamInfo();
		float sx    = (float) cx->Width() / (float) vwidth;
		float sy    = (float) cx->Height() / (float) vheight;
		int gwidth, gheight;
		GridDimensions(gwidth, gheight);
		for (int x = 0; x < gwidth; x++) {
			for (int y = 0; y < gheight; y++) {
				xo::Box r = xo::Box::Inverted();
				r.ExpandToFit(GridPosToVideo(x, y));
				r.ExpandToFit(GridPosToVideo(x + 1, y + 1));
				xo::BoxF rscaled = r;
				rscaled.Expand(-3, -3);
				rscaled.Scale(sx, sy);
				cx->StrokeRect(rscaled, xo::Color::RGBA(200, 0, 0, 150), 1);
			}
		}

		VideoCanvas->ReleaseAndInvalidate(cx);
	}

	void OnKeyChar(xo::Event& ev) {
		switch (ev.KeyChar) {
		case ',':
		case ' ':
			FlipPlayState();
			break;
		case '.':
			NextFrame();
			break;
		}
		for (auto c : Classes) {
			if (toupper(c.Key) == toupper(ev.KeyChar)) {
				AssignLabel(c);
				break;
			}
		}
	}

	void AssignLabel(LabelClass c) {
	}

	void OnCanvasMouseMove(xo::Event& ev) {
	}

	void GridDimensions(int& width, int& height) {
		int vwidth, vheight;
		Video.Dimensions(vwidth, vheight);
		width = (int) ceil((float) vwidth / (float) LabelGridSize);
		height = (int) ceil((float) vheight / (float) LabelGridSize);
	}

	xo::Point VideoPosToGrid(int x, int y) {
		int vwidth, vheight;
		Video.Dimensions(vwidth, vheight);
		if (!GridTopDown)
			y = vheight - y;
		return xo::Point((int) (x / LabelGridSize), (int) (y / LabelGridSize));
	}

	xo::Point GridPosToVideo(int x, int y) {
		int vwidth, vheight;
		Video.Dimensions(vwidth, vheight);
		x *= LabelGridSize;
		y *= LabelGridSize;
		if (!GridTopDown)
			y = vheight - y;
		return xo::Point(x, y);
	}

	void NextFrame() {
		if (!Video.IsOpen())
			return;
		auto cx = VideoCanvas->GetCanvas2D();
		Video.DecodeFrameRGBA(cx->Width(), cx->Height(), cx->RowPtr(0), cx->Stride());
		VideoCanvas->ReleaseAndInvalidate(cx);
		DrawLabelBoxes();
		RenderTimeSlider();
	}

	bool OpenVideo() {
		if (Video.GetFilename() == VideoFilename)
			return true;
		Video.Close();

		auto err = Video.OpenFile(VideoFilename);
		if (!err.OK()) {
			xo::controls::MsgBox::Show(Root->GetDoc(), err.Message());
			return false;
		}
		Render();
		NextFrame();
		return true;
	}
};

} // namespace anno
} // namespace imqs

void xoMain(xo::SysWnd* wnd) {
	using namespace imqs::anno;
	VideoFile::Initialize();

	wnd->Doc()->ClassParse("font-medium", "font-size: 14ep");
	wnd->Doc()->ClassParse("shortcut", "font-size: 15ep; color: #000; width: 1em");

	svg::LoadAll(wnd->Doc());
	wnd->SetPosition(xo::Box(0, 0, 1500, 970), xo::SysWnd::SetPosition_Size);

	auto ui = new UI(&wnd->Doc()->Root);

	ui->Classes.push_back({'R', "normal road"});
	ui->Classes.push_back({'C', "crocodile cracks"});
	ui->Classes.push_back({'B', "bricks"});
	ui->Classes.push_back({'P', "pothole"});
	ui->Classes.push_back({'S', "straight crack"});
	ui->Classes.push_back({'M', "manhole cover"});
	ui->Classes.push_back({'X', "pockmarks"});
	ui->Classes.push_back({'U', "unlabeled"});

	ui->VideoFilename = "D:\\mldata\\GOPR0080.MP4";
	if (!ui->OpenVideo())
		ui->Render();
}
