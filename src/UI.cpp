#include "pch.h"
#include "UI.h"

namespace imqs {
namespace anno {

void UI::SaveThreadFunc(UI* ui) {
	std::string saveErrorStorage;

	while (true) {
		if (ui->IsExiting)
			break;

		SavePackage* package = ui->SaveQueue;
		if (package) {
			if (ui->SaveQueue.compare_exchange_strong(package, nullptr)) {
				for (const auto& frame : package->Labels.Frames) {
					auto err = SaveFrameLabels(package->VideoFilename, frame);
					if (!err.OK()) {
						saveErrorStorage = err.Message();
						ui->SaveError    = saveErrorStorage.c_str();
						break;
					}
				}
				delete package;
			}
		} else {
			os::Sleep(200 * time::Millisecond);
		}
	}
}

UI::UI(xo::DomNode* root) {
	IsExiting = false;
	SaveQueue = nullptr;
	Root      = root;
	Render();
}

UI::~UI() {
	IsExiting = true;
	SaveThread.join();
}

void UI::Render() {
	auto ui = this;
	Root->Clear();

	auto   videoArea  = Root->ParseAppendNode("<div style='break:after; box-sizing: margin; width: 100%; margin: 10ep'></div>");
	double aspect     = 1920.0 / 1080.0;
	double videoWidth = 1400;

	VideoCanvas = videoArea->AddCanvas();
	VideoCanvas->StyleParse("hcenter: hcenter; border: 1px #888; border-radius: 1.5ep;");
	VideoCanvas->SetSize((int) videoWidth, (int) (videoWidth / aspect));
	VideoCanvas->OnMouseMove([ui](xo::Event& ev) { ui->OnPaintLabel(ev); });
	VideoCanvas->OnMouseDown([ui](xo::Event& ev) { ui->OnPaintLabel(ev); });

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
		Root->OnTimer([ui]() { ui->OnSaveTimer(); }, 1000);
	}
}

void UI::RenderTimeSlider(bool first) {
	if (!Video.IsOpen())
		return;

	if (first) {
		TimeSliderBox->Clear();
		TimeSliderBox->StyleParse("break:after; margin: 16ep 8ep; padding: 0ep; box-sizing: margin; width: 100%; height: 34ep; border-radius: 3ep; background: #eee");
		TimeSliderBox->ParseAppendNode("<div style='position: absolute; width: 100%; height: 50%; background: #cfc'></div>");
		TimeSliderBox->ParseAppendNode("<div style='background: #ccc; border: 1px #888; width: 5ep; height: 120%; vcenter: vcenter; border-radius: 2.5ep'></div>");

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

	auto tickContainer = TimeSliderBox->NodeByIndex(0);
	auto caret = TimeSliderBox->NodeByIndex(1);

	auto   info = Video.GetVideoStreamInfo();
	double dur  = info.DurationSeconds();
	double pos  = Video.LastFrameTimeSeconds();

	tickContainer->Clear();
	for (const auto& frame : Labels.Frames) {
		double p = 100.0 * frame.TimeSeconds() / dur;
		auto tick = tickContainer->ParseAppendNode("<div></div>");
		tick->StyleParsef("background: #c005; position: absolute; hcenter: %.1f%%; width: 3px; height: 100%%", p);
	}

	caret->StyleParsef("hcenter: %.1f%%", pos * 100.0 / dur);
}

void UI::FlipPlayState() {
	if (PlayState == PlayStates::Play)
		Stop();
	else
		Play();
}

void UI::SeekFromUI(xo::Event& ev) {
	auto  box    = ev.LayoutResult->Node(TimeSliderBox);
	float mouseX = ev.PointsRel[0].x / box->ContentWidthPx(); // percentage
	mouseX       = xo::Clamp(mouseX, 0.f, 1.f);
	Video.SeekToFraction(mouseX);
	// I've been too lazy to understand why we need to burn a frame after seeking.
	NextFrame();
	NextFrame();
}

void UI::Play() {
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

void UI::Stop() {
	if (PlayState == PlayStates::Stop)
		return;
	xo::controls::Button::SetSvg(PlayBtn, "media-play");
	PlayState = PlayStates::Stop;
	VideoCanvas->RemoveHandler(PlayTimer);
	PlayTimer = 0;
}

// The labels in our label store use raw video pixel coordinate rectangles.
// Here we map those arbitrary rectangles back to our fixed grid.
ohash::map<uint64_t, Label> UI::ExtractLabelsForGrid() {
	ohash::map<uint64_t, Label> map;
	auto                        frame = Labels.FindFrame(Video.LastFrameTimeMicrosecond());
	if (!frame)
		return map;

	for (const auto& lab : frame->Labels) {
		auto c       = lab.Rect.Center();
		auto gridPos = VideoPosToGrid(c.X, c.Y);
		auto gridCo  = MakeGridCoord64(gridPos.X, gridPos.Y);
		map.insert(gridCo, lab, true);
	}

	return map;
}

Label* UI::FindOrInsertLabel(ImageLabels* frame, int gridX, int gridY) {
	for (auto& lab : frame->Labels) {
		auto c       = lab.Rect.Center();
		auto gridPos = VideoPosToGrid(c.X, c.Y);
		if (gridPos.X == gridX && gridPos.Y == gridY)
			return &lab;
	}
	Label lab;
	auto  p1 = GridPosToVideo(gridX, gridY);
	auto  p2 = GridPosToVideo(gridX, gridY);
	lab.Rect = Rect(p1.X, p1.Y, p2.X, p2.Y);
	frame->Labels.push_back(lab);
	frame->IsDirty = true;
	return &frame->Labels.back();
}

bool UI::RemoveLabel(ImageLabels* frame, int gridX, int gridY) {
	for (size_t i = 0; i < frame->Labels.size(); i++) {
		const auto& lab     = frame->Labels[i];
		auto        c       = lab.Rect.Center();
		auto        gridPos = VideoPosToGrid(c.X, c.Y);
		if (gridPos.X == gridX && gridPos.Y == gridY) {
			frame->IsDirty = true;
			frame->Labels.erase(frame->Labels.begin() + i);
			return true;
		}
	}
	return false;
}

void UI::DrawLabelBoxes() {
	auto cx = VideoCanvas->GetCanvas2D();
	cx->GetImage()->CopyFrom(&LastFrameImg);
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	auto  labelMap = ExtractLabelsForGrid();
	float sx       = (float) cx->Width() / (float) vwidth;
	float sy       = (float) cx->Height() / (float) vheight;
	int   gwidth, gheight;
	GridDimensions(gwidth, gheight);
	for (int x = 0; x < gwidth; x++) {
		for (int y = 0; y < gheight; y++) {
			auto    label = labelMap.get(MakeGridCoord64(x, y));
			xo::Box r     = xo::Box::Inverted();
			auto    p1    = GridPosToVideo(x, y);
			auto    p2    = GridPosToVideo(x + 1, y + 1);
			r.ExpandToFit(p1);
			r.ExpandToFit(p2);
			xo::BoxF rscaled = r;
			rscaled.Expand(-3, -3);
			rscaled.Scale(sx, sy);
			float centx = 0.5f * (rscaled.Left + rscaled.Right);
			float centy = 0.5f * (rscaled.Top + rscaled.Bottom);
			cx->StrokeRect(rscaled, xo::Color::RGBA(200, 0, 0, 150), 1);
			size_t      k         = FindClass(label.Class);
			float       fontSize  = 24;
			auto        color     = xo::Color(180, 0, 0, 180);
			const char* font      = "Segoe UI Bold";
			bool        isLabeled = true;
			if (k == -1) {
				isLabeled        = false;
				k                = UnlabeledClass;
				fontSize         = 14;
				color            = xo::Color(200, 0, 0, 100);
				const char* font = "Segoe UI";
			}
			char labelStr[2];
			labelStr[0] = (char) Classes[k].Key;
			labelStr[1] = 0;
			float tx    = (float) (centx - fontSize / 2);
			float ty    = (float) (centy + fontSize / 2);
			if (isLabeled) {
				for (float dx = -1; dx <= 1; dx++) {
					for (float dy = -1; dy <= 1; dy++) {
						cx->Text(tx + dx, ty + dy, 0, fontSize, xo::Color(255, 255, 255, 150), font, labelStr);
					}
				}
			}
			cx->Text(tx, ty, 0, fontSize, color, font, labelStr);
		}
	}

	VideoCanvas->ReleaseAndInvalidate(cx);
}

void UI::OnKeyChar(xo::Event& ev) {
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

// Find any dirty frames and add them to a save package
void UI::OnSaveTimer() {
	// Wait for previous package to be saved away
	if (SaveQueue.load() != nullptr)
		return;

	SavePackage* package   = new SavePackage();
	package->VideoFilename = VideoFilename;
	for (auto& frame : Labels.Frames) {
		if (frame.IsDirty) {
			// This assumes that the save will succeed. To manage failures is quite a bit more work, and
			// I'm not convinced it's worth the effort right now.
			frame.IsDirty = false;
			package->Labels.Frames.emplace_back(frame);
		}
	}
	if (package->Labels.Frames.size() == 0) {
		delete package;
		return;
	}

	SaveQueue = package;
}

void UI::AssignLabel(LabelClass c) {
	ActionState        = ActionStates::AssignLabel;
	CurrentAssignClass = c;
}

void UI::OnPaintLabel(xo::Event& ev) {
	if (ActionState == ActionStates::AssignLabel && (ev.Type == xo::EventMouseDown || ev.IsPressed(xo::Button::MouseLeft))) {
		auto vpos  = CanvasToVideoPos((int) ev.PointsRel[0].x, (int) ev.PointsRel[0].y);
		auto gpos  = VideoPosToGrid(vpos.X, vpos.Y);
		auto time  = Video.LastFrameTimeMicrosecond();
		auto frame = Labels.FindOrInsertFrame(time);
		if (CurrentAssignClass.Class == Classes[UnlabeledClass].Class) {
			if (RemoveLabel(frame, gpos.X, gpos.Y))
				DrawLabelBoxes();
			return;
		}
		auto label = FindOrInsertLabel(frame, gpos.X, gpos.Y);
		if (label->Class != CurrentAssignClass.Class) {
			frame->IsDirty = true;
			label->Class   = CurrentAssignClass.Class;
			DrawLabelBoxes();
		}
	}
}

void UI::GridDimensions(int& width, int& height) {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	width  = (int) ceil((float) vwidth / (float) LabelGridSize);
	height = (int) ceil((float) vheight / (float) LabelGridSize);
}

// If video is shown at half it's actual resolution, then scale = 0.5
xo::Vec2f UI::VideoScaleOnCanvas() {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	auto cx      = VideoCanvas->GetCanvas2D();
	int  cwidth  = cx->Width();
	int  cheight = cx->Height();
	VideoCanvas->ReleaseCanvas(cx);
	return xo::Vec2f(vwidth / (float) cwidth, vheight / (float) cheight);
}

xo::Point UI::CanvasToVideoPos(int x, int y) {
	auto scale = VideoScaleOnCanvas();
	return xo::Point((int) (x * scale.x), (int) (y * scale.y));
}

xo::Point UI::VideoPosToGrid(int x, int y) {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	if (!GridTopDown)
		y = vheight - y;
	return xo::Point((int) (x / LabelGridSize), (int) (y / LabelGridSize));
}

xo::Point UI::GridPosToVideo(int x, int y) {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	x *= LabelGridSize;
	y *= LabelGridSize;
	if (!GridTopDown)
		y = vheight - y;
	return xo::Point(x, y);
}

void UI::NextFrame() {
	if (!Video.IsOpen())
		return;
	auto cx = VideoCanvas->GetCanvas2D();
	Video.DecodeFrameRGBA(cx->Width(), cx->Height(), cx->RowPtr(0), cx->Stride());
	LastFrameImg.Alloc(xo::TexFormatRGBA8, cx->Width(), cx->Height());
	LastFrameImg.CopyFrom(cx->GetImage());
	VideoCanvas->ReleaseAndInvalidate(cx);
	DrawLabelBoxes();
	RenderTimeSlider();
}

bool UI::OpenVideo() {
	if (Video.GetFilename() == VideoFilename)
		return true;
	Video.Close();

	auto err = Video.OpenFile(VideoFilename);
	if (!err.OK()) {
		xo::controls::MsgBox::Show(Root->GetDoc(), err.Message());
		return false;
	}
	LoadVideoLabels(VideoFilename, Labels);
	Render();
	NextFrame();
	return true;
}

size_t UI::FindClass(const std::string& klass) {
	for (size_t i = 0; i < Classes.size(); i++) {
		if (Classes[i].Class == klass)
			return i;
	}
	return -1;
}

} // namespace anno
} // namespace imqs
