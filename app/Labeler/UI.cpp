#include "pch.h"
#include "UI.h"

using namespace std;
using namespace imqs::train;

namespace imqs {
namespace anno {

void UI::LoadSaveThreadFunc(UI* ui) {
	auto        lastLoad = time::Now();
	std::string lastVideoFilename;
	auto        reloadInterval = 60 * time::Second;

	auto setError = [ui](std::string err) {
		std::lock_guard<std::mutex> lock(ui->LoadSaveErrorLock);
		ui->LoadSaveError = err;
	};

	while (!ui->IsExiting) {
		LoadSavePackage* package = ui->SaveQueue;
		if (package) {
			setError("");
			if (ui->SaveQueue.compare_exchange_strong(package, nullptr)) {
				// save last filename, so that loader knows about it
				lastVideoFilename = package->VideoFilename;
				for (const auto& frame : package->Labels.Frames) {
					auto err = SaveFrameLabels(package->VideoFilename, frame);
					if (!err.OK()) {
						setError(tsf::fmt("Error saving labels for %v: %v", package->VideoFilename, err.Message()));
						break; // don't try saving further frames
					}
				}
				delete package;
			}
		} else if (time::Now() - lastLoad > reloadInterval && lastVideoFilename != "") {
			setError("");
			package                = new LoadSavePackage();
			package->VideoFilename = lastVideoFilename;
			auto err               = LoadVideoLabels(lastVideoFilename, package->Labels);
			if (!err.OK()) {
				setError(tsf::fmt("Error loading labels for %v: %v", lastVideoFilename, err.Message()));
				delete package;
			} else {
				// Ideally we should remove a stale doc here, but that just makes us more complicated. I don't believe it's worth the tradeoff.
				LoadSavePackage* empty = nullptr;
				if (!ui->LoadQueue.compare_exchange_strong(empty, package))
					delete package;
				lastLoad = time::Now();
			}
		} else {
			os::Sleep(100 * time::Millisecond);
		}
	}
}

UI::UI(xo::DomNode* root) {
	IsExiting      = false;
	SaveQueue      = nullptr;
	LoadQueue      = nullptr;
	UserName       = os::UserName();
	Root           = root;
	LoadSaveThread = std::thread(LoadSaveThreadFunc, this);
	ModelLoadErr   = "Loading...";
	Render();
}

UI::~UI() {
	// wait for the load/save thread to process any pending writes
	while (SaveQueue.load())
		os::Sleep(50 * time::Millisecond);

	IsExiting = true;
	LoadSaveThread.join();

	// discard any pending load
	delete LoadQueue.load();
}

void UI::Render() {
	auto ui = this;
	Root->Clear();

	auto   videoArea = Root->ParseAppendNode("<div style='break:after; box-sizing: margin; width: 100%; margin: 2ep'></div>");
	double aspect    = 1920.0 / 1080.0;
	//double videoWidth = 1540;
	VideoCanvasWidth  = 1550;
	VideoCanvasHeight = (int) (VideoCanvasWidth / aspect);

	ToolsBox = videoArea->ParseAppendNode("<div>Tools...</div>");
	ToolsBox->StyleParse("width: 200ep");
	RenderToolsUI();

	VideoCanvas = videoArea->AddCanvas();
	VideoCanvas->StyleParse("border: 1px #888; border-radius: 1.5ep; break:after;");
	VideoCanvas->StyleParsef("width: %dpx; height: %dpx", VideoCanvasWidth, VideoCanvasHeight);
	VideoCanvas->OnMouseMove([ui](xo::Event& ev) {
		if (ev.IsPressed(xo::Button::KeyAlt))
			return;
		int  dragThreshold = ui->LabelMode == UI::LabelModes::FixedBoxes ? 15 : 0;
		auto vpos          = ui->CanvasToVideoPos((int) ev.PointsRel[0].x, (int) ev.PointsRel[0].y);
		if (ui->IsCreatingPolygon && ui->CurrentPolygon->Vertices.size() != 0) {
			ui->CurrentPolygon->Vertices.back() = train::Point(vpos.X, vpos.Y);
			ui->DrawLabels();
		} else if (ev.IsPressed(xo::Button::MouseLeft) && (ev.PointsRel[0].distance(ui->MouseDown) > dragThreshold || ui->IsMouseDragging)) {
			// only start drag after cursor has moved some distance, to prevent false twitchy positives
			auto vposMouseDown  = ui->CanvasToVideoPos((int) ui->MouseDown.x, (int) ui->MouseDown.y);
			ui->IsMouseDragging = true;
			if (ui->IsMouseDragging && ui->LabelMode == UI::LabelModes::FixedBoxes)
				ui->OnPaintLabel(ev);
			else if (ui->IsMouseDragging && ui->LabelMode == UI::LabelModes::Segmentation)
				ui->Segmentation_MouseDrag(ev, vposMouseDown, vpos);
		}
	});
	VideoCanvas->OnMouseDown([ui](xo::Event& ev) {
		auto vpos = ui->CanvasToVideoPos((int) ev.PointsRel[0].x, (int) ev.PointsRel[0].y);

		if (ev.Button == xo::Button::MouseLeft && ev.IsPressed(xo::Button::KeyAlt)) {
			ui->Segmentation_DeleteLabel(vpos);
			ui->DrawLabels();
		} else if (ev.Button == xo::Button::MouseLeft) {
			ui->MouseDown = ev.PointsRel[0];
			if (ui->IsCreatingPolygon) {
				ui->CurrentPolygon->Vertices.push_back(train::Point(vpos.X, vpos.Y));
				if (ui->CurrentPolygon->Vertices.size() == 1)
					ui->CurrentPolygon->Vertices.push_back(train::Point(vpos.X, vpos.Y));
				ui->DrawLabels();
			} else {
				ui->IsMouseDragging = false;
				ui->OnPaintLabel(ev);
			}
		}
	});
	VideoCanvas->OnMouseUp([ui](xo::Event& ev) {
		if (ui->CurrentLabel && ui->IsDraggingVertex) {
			ui->SetLabelDirty(ui->LabelsForCurrentFrame(false), ui->CurrentLabel, false);
		}
		ui->IsMouseDragging  = false;
		ui->IsDraggingVertex = false;
		//ui->CurrentLabel          = nullptr;
		ui->CurrentDraggingVertex = -1;
		if (ev.Button == xo::Button::MouseRight && ui->IsCreatingPolygon) {
			ui->CurrentPolygon->Vertices.pop_back();
			ui->SetLabelDirty(ui->LabelsForCurrentFrame(false), ui->CurrentLabel, false);
			ui->IsCreatingPolygon = false;
			ui->CurrentPolygon    = nullptr;
			ui->CurrentLabel      = nullptr;
			ui->DrawLabels();
		}
	});

	auto fileArea = videoArea->ParseAppendNode("<div style='break:after; margin: 4ep; margin-bottom: 0ep'></div>");
	fileArea->StyleParsef("width: %vpx", VideoCanvasWidth); // hcenter:hcenter
	auto fileName  = fileArea->ParseAppendNode("<div class='font-medium' style='cursor: hand'></div>");
	auto exportBtn = xo::controls::Button::NewText(fileArea, "Export");
	StatusLabel    = fileArea->ParseAppendNode("<div class='font-medium' style='margin-left: 2em'></div>");
	ErrorLabel     = fileArea->ParseAppendNode("<div class='font-medium' style='right: right'></div>");
	fileName->AddText(VideoFilename);
	fileName->OnClick([this] {
		std::vector<std::pair<std::string, std::string>> types = {
		    {"Video Files", "*.mp4;*.avi;*.mkv;*.mov"},
		};
		if (xo::osdialogs::BrowseForFileOpen(Root->GetDoc(), types, VideoFilename))
			OpenVideo();
	});

	ExportCallback = [this](size_t i, size_t total) -> bool {
		if (ExportMsgBox == nullptr)
			return false;
		// We can't update the DOM object directly, because that's a threading violation. The xo rendering
		// thread could be busy reading the DOM object while we're executing here.
		// Instead, we rely on a timer to take care of the DOM update.
		ExportProgMsg = tsf::fmt("%v/%d frames", i, total);
		return true;
	};

	ExportDlgClosed = [this]() {
		ExportMsgBox = nullptr;
		if (ExportThread.joinable())
			ExportThread.join();
	};

	Root->OnTimer([this] {
		if (ExportMsgBox)
			ExportMsgBox->SetText(ExportProgMsg.c_str());
	},
	              1000);

	exportBtn->OnClick([this] {
		ExportMsgBox = new xo::controls::MsgBox();
		ExportMsgBox->Create(Root->GetDoc(), "Exporting...", "Cancel");
		ExportMsgBox->OnClose = ExportDlgClosed;

		// super lazy thread use
		auto taxonomyCopy = Taxonomy;
		auto labelsCopy   = Labels;
		auto filenameCopy = VideoFilename;
		ExportThread      = std::thread([this, taxonomyCopy, labelsCopy, filenameCopy] {
            auto exportType = LabelMode == UI::LabelModes::FixedBoxes ? ExportTypes::Jpeg : ExportTypes::Segmentation;
            auto err        = ExportLabeledImagePatches_Video(exportType, filenameCopy, taxonomyCopy, labelsCopy, ExportCallback);
            if (ExportMsgBox) {
                if (err.OK())
                    ExportMsgBox->SetText("Done");
                else
                    ExportMsgBox->SetText(tsf::fmt("Error: %v", err.Message()).c_str());
            }
            ExportMsgBox = nullptr;
        });
		//auto err = ExportLabeledImagePatches_Video(ExportTypes::Png, VideoFilename, Labels, prog);
		//xo::controls::MsgBox::Show(Root->GetDoc(), tsf::fmt("Done: %v", err.Message()).c_str());
	});

	TimeSliderBox = Root->ParseAppendNode("<div></div>");
	RenderTimeSlider(true);

	auto bottomControlBoxes = Root->ParseAppendNode("<div style='padding: 4ep'></div>");
	auto mediaControlBox    = bottomControlBoxes->ParseAppendNode("<div></div>");
	auto modeControlBox     = bottomControlBoxes->ParseAppendNode("<div></div>");
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
	//btnTest->OnClick([ui] {
	//});

	auto modeSwitch = modeControlBox->ParseAppendNode(tsf::fmt("<div>Mode: %v</div>", IsModeLabel ? "Label" : "Inference"));
	modeSwitch->OnClick([ui] {
		ui->IsModeLabel = !ui->IsModeLabel;
		ui->Render();
	});

	LabelBox = bottomControlBoxes->ParseAppendNode("<div style='padding-left: 20ep; color: #333'></div>");
	RenderLabelUI();

	// Setup once-off things
	if (OnDestroyEv == 0) {
		OnDestroyEv = Root->OnDestroy([ui] {
			delete ui;
		});

		Root->OnKeyChar([ui](xo::Event& ev) { ui->OnKeyChar(ev); });
		Root->OnTimer([ui]() { ui->OnLoadSaveTimer(); }, 1000);
	}

	DrawCurrentFrame();
}

void UI::RenderToolsUI() {
	if (!ToolsBox)
		return;

	ToolsBox->Clear();
	/*
	auto newLabelBtn = xo::controls::Button::NewText(ToolsBox, "Polygon");
	newLabelBtn->StyleParse("break:after");

	newLabelBtn->OnClick([this] {
		CreateNewPolygonLabel("lane");
	});
	*/
}

void UI::RenderLabelUI() {
	if (!LabelBox)
		return;
	LabelBox->Clear();
	int labelRows = 3;

	for (auto group : Taxonomy.ClassGroups()) {
		if (!group[0].IsPolygon && LabelMode != LabelModes::FixedBoxes)
			continue;
		if (group[0].IsPolygon && LabelMode != LabelModes::Segmentation)
			continue;

		// compute number of columns necessary for this group, and then size the width
		// of the container so that the children will wrap around appropriately
		int  labelCols = (int) ceil(group.size() / (double) labelRows);
		auto makeLabel = [](string key, string title, bool isActive) -> string {
			// the label with (7.5) must be slightly less than the column multiplier, down below (8)
			const char* active = isActive ? "class='active-label'" : "";
			return tsf::fmt("<div style='width: 7.2em' %v><span class='shortcut'>%v</span>%v</div>", active, key, title);
		};
		string labelSrc = tsf::fmt("<div style='width: %vem' class='label-group'>", labelCols * 7.4);
		for (size_t i = 0; i < group.size(); i++) {
			bool active = CurrentAssignClass.Class == group[i].Class;
			labelSrc += makeLabel(group[i].KeyStr(), group[i].Class, active);
		}
		labelSrc += "</div>";
		LabelBox->ParseAppend(labelSrc);
	}
	string severity = tsf::fmt("<div style='margin-left: 10ep'>Severity <span class='severity'>%v</span></div>", CurrentAssignSeverity);
	LabelBox->ParseAppend(severity);
}

void UI::RenderTimeSlider(bool first) {
	if (!Video.IsOpen())
		return;

	if (first) {
		TimeSliderBox->Clear();
		TimeSliderBox->StyleParse("break:after; margin: 16ep 4ep; padding: 0ep; box-sizing: margin; width: 100%; height: 24ep; border-radius: 3ep; background: #eee");
		TimeSliderBox->ParseAppendNode("<div style='position: absolute; width: 100%; height: 100%'></div>");                                                         // tick container
		TimeSliderBox->ParseAppendNode("<div style='background: #ccc3; border: 1px #888; width: 5ep; height: 120%; vcenter: vcenter; border-radius: 2.5ep'></div>"); // caret

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
	auto caret         = TimeSliderBox->NodeByIndex(1);

	auto   info = Video.GetVideoStreamInfo();
	double dur  = info.DurationSeconds();
	double pos  = Video.LastFrameTimeSeconds();

	// lerp between an old and new color, to indicate when other people are actively
	// working on a segment of video.
	xo::Vec3f oldcolor(0, 0.95f, 0);
	xo::Vec3f newcolor(0.95f, 0, 0);

	auto now = time::Now();
	tickContainer->Clear();
	for (const auto& frame : Labels.Frames) {
		if (LabelMode == LabelModes::FixedBoxes && !frame.HasRects())
			continue;
		if (LabelMode == LabelModes::Segmentation && !frame.HasPolygons())
			continue;
		double p    = 100.0 * frame.TimeSeconds() / dur;
		auto   tick = tickContainer->ParseAppendNode("<div></div>");
		float  age  = (float) ((now - frame.MaxEditTime()).Seconds() / 300);
		age         = xo::Clamp(age, 0.0f, 1.0f);
		// lerping in sRGB is just so damn ugly. But I am so damn ready to ship this!
		xo::Vec3f fcolor = age * oldcolor + (1.0f - age) * newcolor;
		int       r      = xo::Clamp((int) (fcolor.x * 255), 0, 255);
		int       g      = xo::Clamp((int) (fcolor.y * 255), 0, 255);
		int       b      = xo::Clamp((int) (fcolor.z * 255), 0, 255);
		int       a      = 80;
		tick->StyleParsef("background: #%02x%02x%02x%02x; position: absolute; hcenter: %.1f%%; width: 1px; height: 100%%", r, g, b, a, p);
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
	// When seeking by fraction, we just seek to a keyframe, and the codec buffer still has data inside it. So that's
	// why we need to burn a frame.
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

train::ImageLabels* UI::LabelsForCurrentFrame(bool create) {
	if (create)
		return Labels.FindOrInsertFrame(Video.LastFrameTimeMicrosecond());
	else
		return Labels.FindFrame(Video.LastFrameTimeMicrosecond());
}

void UI::Segmentation_MouseDrag(xo::Event& ev, xo::Point vposMouseDown, xo::Point vpos) {
	if (!IsDraggingVertex) {
		Label* lab                = nullptr;
		size_t ivertex            = -1;
		bool   canCreateNewVertex = !IsDraggingVertex && VideoPosToCanvas(vposMouseDown.X, vposMouseDown.Y).distance(VideoPosToCanvas(vpos.X, vpos.Y)) > 10;
		Segmentation_FindCloseVertex(vposMouseDown, lab, ivertex, canCreateNewVertex);
		if (lab == nullptr)
			return;
		IsDraggingVertex      = true;
		CurrentLabel          = lab;
		CurrentDraggingVertex = ivertex;
	}

	if (IsDraggingVertex) {
		CurrentLabel->Polygon.Vertices[CurrentDraggingVertex] = train::Point(vpos.X, vpos.Y);
		DrawLabels();
	}
}

void UI::Segmentation_FindCloseVertex(xo::Point vpos, train::Label*& lab, size_t& ivertex, bool createVertexIfNone) {
	lab        = nullptr;
	ivertex    = -1;
	auto frame = LabelsForCurrentFrame(false);
	if (!frame)
		return;

	float bestDistVx   = 30;
	float bestDistLine = 30;

	train::Label* closestLabByLine    = nullptr;
	size_t        closestVertexByLine = -1;
	gfx::Vec2f    closestPosByLine;

	gfx::Vec2f vposf((float) vpos.X, (float) vpos.Y);

	for (auto& ilab : frame->Labels) {
		if (ilab.IsPolygon()) {
			for (size_t i = 0; i < ilab.Polygon.Vertices.size(); i++) {
				size_t j    = (i + 1) % ilab.Polygon.Vertices.size();
				float  dist = ilab.Polygon.Vertices[i].Distance(vpos.X, vpos.Y);
				if (dist < bestDistVx) {
					lab        = &ilab;
					ivertex    = i;
					bestDistVx = dist;
				}
				auto closestPt = geom::ClosestPtOnLineT(vposf, (gfx::Vec2f) ilab.Polygon.Vertices[i], (gfx::Vec2f) ilab.Polygon.Vertices[j], true);
				dist           = closestPt.distance(vposf);
				if (dist < bestDistLine) {
					bestDistLine        = dist;
					closestLabByLine    = &ilab;
					closestVertexByLine = i;
					closestPosByLine    = closestPt;
				}
			}
		}
	}

	if (lab)
		return;

	if (createVertexIfNone && closestLabByLine) {
		lab = closestLabByLine;
		lab->Polygon.Vertices.insert(lab->Polygon.Vertices.begin() + closestVertexByLine + 1, train::Point((int) closestPosByLine.x, (int) closestPosByLine.y));
		ivertex = closestVertexByLine + 1;
	}
}

train::Label* UI::Segmentation_FindCloseLabel(xo::Point vpos) {
	auto frame = LabelsForCurrentFrame(false);
	if (!frame)
		return nullptr;

	for (auto& lab : frame->Labels) {
		if (lab.IsPolygon()) {
			if (geom2d::PtInsidePoly(vpos.X, vpos.Y, lab.Polygon.Vertices.size(), &lab.Polygon.Vertices[0].X, sizeof(train::Point) / sizeof(int)))
				return &lab;
		}
	}

	return nullptr;
}

void UI::Segmentation_DeleteLabel(xo::Point vpos) {
	auto lab = Segmentation_FindCloseLabel(vpos);
	if (lab) {
		auto frame = LabelsForCurrentFrame(false);
		for (size_t i = 0; i < frame->Labels.size(); i++) {
			if (&frame->Labels[i] == lab) {
				SetLabelDirty(frame, lab, false);
				frame->Labels.erase(frame->Labels.begin() + i);
				break;
			}
		}
	}
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
	auto  p2 = GridPosToVideo(gridX + 1, gridY + 1);
	lab.Rect = Rect::Inverted();
	lab.Rect.ExpandToFit(p1.X, p1.Y);
	lab.Rect.ExpandToFit(p2.X, p2.Y);
	frame->Labels.push_back(lab);
	return &frame->Labels.back();
}

bool UI::RemoveLabel(ImageLabels* frame, int gridX, int gridY) {
	for (size_t i = 0; i < frame->Labels.size(); i++) {
		const auto& lab     = frame->Labels[i];
		auto        c       = lab.Rect.Center();
		auto        gridPos = VideoPosToGrid(c.X, c.Y);
		if (gridPos.X == gridX && gridPos.Y == gridY) {
			frame->SetDirty();
			frame->Labels.erase(frame->Labels.begin() + i);
			return true;
		}
	}
	return false;
}

// Using our trained model, evaluate at intervals, and overlay
void UI::DrawEvalOverlay() {
	if (ModelLoadErr != "")
		return;
#ifdef IMQS_AI_API
	auto cx = VideoCanvas->GetCanvas2D();
	cx->GetImage()->CopyFrom(&LastFrameImg);
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	xo::Color palette[] = {
	    {250, 0, 0, 200},
	    {250, 250, 0, 200},
	    {0, 250, 0, 200},
	    {0, 250, 250, 200},
	    {0, 0, 250, 200},
	    {250, 0, 250, 200},
	};
	int lwidth  = (vwidth / LabelGridSize) * LabelGridSize;
	int lheight = (vheight / LabelGridSize) * LabelGridSize;
	int stride  = 128;
	int gwidth  = 1 + (vwidth - LabelGridSize) / stride;
	int gheight = 1 + (vheight - LabelGridSize) / stride;
	gheight     = 4;
	auto start  = time::Now();
	for (int x = 0; x < gwidth; x++) {
		for (int y = 0; y < gheight; y++) {
			IMQS_ASSERT(x * stride + LabelGridSize <= (int) LastFrameImg.Width);
			IMQS_ASSERT(y * stride + LabelGridSize <= (int) LastFrameImg.Height);
			int     py = vheight - y * stride - LabelGridSize;
			int     c  = Model.EvalRGBAClassify(LastFrameImg.DataAt(x * stride, py), LabelGridSize, LabelGridSize, LastFrameImg.Stride);
			int     bs = 10;
			int     x1 = (int) (((float) x + 0.5f) * stride);
			int     y1 = vheight - (int) (((float) y + 0.5f) * stride);
			xo::Box r(x1, y1, x1 + bs, y1 + bs);
			cx->FillRect(r, palette[c]);
		}
	}
	auto   end     = time::Now();
	double totalMS = (end - start).Milliseconds();
	xo::Trace("Inference: %.1f ms (%.1f ms / sample)\n", totalMS, totalMS / (gwidth * gheight));
#endif
}

void UI::DrawLabels() {
	switch (LabelMode) {
	case LabelModes::FixedBoxes: DrawLabelBoxes(); break;
	case LabelModes::Segmentation: DrawSegmentationLabels(); break;
	default: IMQS_DIE();
	}
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
			float       fontSize  = 24;
			auto        color     = xo::Color(180, 0, 0, 180);
			const char* font      = "Segoe UI Bold";
			bool        isLabeled = true;
			string      labelStr;
			if (label.Classes.size() == 0) {
				labelStr  = "u";
				isLabeled = false;
				fontSize  = 14;
				color     = xo::Color(200, 0, 0, 150);
			} else {
				for (const auto& c : label.Classes) {
					auto k = Taxonomy.FindClass(c.Class);
					if (!k) {
						labelStr += "unknown class: " + c.Class;
					} else {
						if (k->HasSeverity)
							labelStr += tsf::fmt("%s%c ", ShortcutKeyForClass(c.Class), '0' + c.Severity);
						else
							labelStr += tsf::fmt("%s ", ShortcutKeyForClass(c.Class));
					}
				}
			}
			if (strings::EndsWith(labelStr, " "))
				labelStr = labelStr.substr(0, labelStr.size() - 1);
			float tx = (float) (centx - (float) labelStr.size() * (float) fontSize / 3.5f);
			float ty = (float) (centy + fontSize / 2);
			if (isLabeled) {
				for (float dx = -1; dx <= 1; dx++) {
					for (float dy = -1; dy <= 1; dy++) {
						cx->Text(tx + dx, ty + dy, 0, fontSize, xo::Color(255, 255, 255, 150), font, labelStr.c_str());
					}
				}
			}
			cx->Text(tx, ty, 0, fontSize, color, font, labelStr.c_str());
		}
	}

	VideoCanvas->ReleaseAndInvalidate(cx);
}

void UI::DrawSegmentationLabels() {
	auto frame = Labels.FindFrame(Video.LastFrameTimeMicrosecond());
	if (!frame)
		return;
	auto cx = VideoCanvas->GetCanvas2D();
	cx->GetImage()->CopyFrom(&LastFrameImg);
	vector<float> vx;
	for (const auto& lab : frame->Labels) {
		if (lab.IsPolygon())
			DrawPolygon(cx, lab.Polygon, &vx);
	}
	VideoCanvas->ReleaseAndInvalidate(cx);
}

void UI::DrawPolygon(xo::Canvas2D* cx, const train::Polygon& poly, std::vector<float>* tmpVx) {
	if (poly.Vertices.size() < 2)
		return;

	vector<float> tmp;
	if (!tmpVx)
		tmpVx = &tmp;

	size_t nv = poly.Vertices.size();
	if (nv * 2 > tmpVx->size())
		tmpVx->resize(max(tmpVx->size() * 2, nv * 2));
	for (size_t i = 0; i < nv; i++) {
		//auto p              = VideoPosToCanvas(poly.Vertices[i].X, poly.Vertices[i].Y);
		auto p              = xo::Vec2f((float) poly.Vertices[i].X, (float) poly.Vertices[i].Y);
		(*tmpVx)[i * 2]     = p.x;
		(*tmpVx)[i * 2 + 1] = p.y;
	}
	cx->StrokeLine(true, (int) nv, &tmpVx->at(0), sizeof(float) * 2, xo::Color(200, 50, 50, 210), 1.2f);
	cx->FillPoly((int) nv, &tmpVx->at(0), sizeof(float) * 2, xo::Color(150, 0, 0, 50));
}

void UI::OnKeyChar(xo::Event& ev) {
	switch (ev.KeyChar) {
	case ' ':
		FlipPlayState();
		break;
	case ',':
		PrevFrame();
		break;
	case '.':
		NextFrame();
		break;
	case '0':
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
		CurrentAssignSeverity = ev.KeyChar - '0';
		RenderLabelUI();
		break;
	}
	for (auto c : Taxonomy.Classes) {
		if (toupper(c.Key) == toupper(ev.KeyChar)) {
			if (LabelMode == LabelModes::FixedBoxes)
				SetCurrentLabel(c);
			else
				CreateNewPolygonLabel(c.Class, CurrentAssignSeverity);
			break;
		}
	}
}

// Find any dirty frames and add them to a save package
void UI::OnLoadSaveTimer() {
	LoadSaveErrorLock.lock();
	ErrorLabel->SetText(LoadSaveError);
	LoadSaveErrorLock.unlock();

	// Wait for previous package to be saved away
	if (SaveQueue.load() != nullptr)
		return;

	{
		// Merge potentially new data that the load/save thread has read from disk
		LoadSavePackage* package = LoadQueue.load();
		if (package && LoadQueue.compare_exchange_strong(package, nullptr)) {
			if (package->VideoFilename == VideoFilename) {
				int nnew = MergeVideoLabels(package->Labels, Labels);
				if (nnew)
					RenderTimeSlider();
			}
			delete package;
		}
	}

	{
		// Save dirty frames
		LoadSavePackage* package = new LoadSavePackage();
		package->VideoFilename   = VideoFilename;
		for (auto& frame : Labels.Frames) {
			if (frame.IsDirty) {
				// This assumes that the save will succeed. To manage failures is quite a bit more work, and
				// I'm not convinced it's worth the effort right now.
				frame.IsDirty = false;
				package->Labels.Frames.emplace_back(frame);
			}
		}
		// Always send a new package, even if the frames are empty. This has the advantage
		// of updating the load/save thread's state of our current video filename, so that
		// it can do background loads.
		SaveQueue = package;
	}

	auto        classToGroup = ClassToGroupMap(Taxonomy.Classes);
	auto        counts       = Labels.CategorizedLabelCount(&classToGroup);
	std::string status;
	int         total = 0;
	for (auto& p : counts) {
		status += tsf::fmt("%v:%d ", p.first, p.second);
		total += p.second;
	}

	if (LabelMode == LabelModes::FixedBoxes) {
		status += tsf::fmt("TOTAL:%d ", total);
		StatusLabel->SetText(status);
		//StatusLabel->SetText(tsf::fmt("%v labels", Labels.TotalLabelCount()));
	} else if (LabelMode == LabelModes::Segmentation) {
		status = tsf::fmt("Frames:%d ", Labels.TotalPolygonFrameCount());
		StatusLabel->SetText(status);
	}
}

void UI::SetCurrentLabel(LabelClass c) {
	ActionState        = ActionStates::AssignLabel;
	CurrentAssignClass = c;
	RenderLabelUI();
}

void UI::OnPaintLabel(xo::Event& ev) {
	if (!IsModeLabel)
		return;
	if (ActionState == ActionStates::AssignLabel && (ev.Type == xo::EventMouseDown || ev.IsPressed(xo::Button::MouseLeft))) {
		auto vpos   = CanvasToVideoPos((int) ev.PointsRel[0].x, (int) ev.PointsRel[0].y);
		auto gpos   = VideoPosToGrid(vpos.X, vpos.Y);
		auto time   = Video.LastFrameTimeMicrosecond();
		auto frame  = Labels.FindOrInsertFrame(time);
		auto label  = FindOrInsertLabel(frame, gpos.X, gpos.Y);
		auto _class = CurrentAssignClass.Class;
		if (label->Severity(_class) == CurrentAssignSeverity && ev.Type == xo::EventMouseDown) {
			// for a single click of the same class, with same severity, remove the class
			label->RemoveClass(_class);
			SetLabelDirty(frame, label);
			return;
		}
		if (CurrentAssignClass.Class == Taxonomy.Classes[UnlabeledClass].Class) {
			// remove all classes
			label->Classes.clear();
			SetLabelDirty(frame, label);
			return;
		}
		// remove all other classes from the same group
		for (const auto& c : Taxonomy.ClassesInGroup(CurrentAssignClass.Group))
			label->RemoveClass(c);
		label->SetClass(CurrentAssignClass.Class, CurrentAssignSeverity);
		SetLabelDirty(frame, label);
	}
}

void UI::CreateNewPolygonLabel(std::string klass, int severity) {
	IsCreatingPolygon = true;
	auto frame        = LabelsForCurrentFrame(true);
	frame->Labels.push_back(Label());
	frame->Labels.back().Classes.push_back(ClassSeverity(klass, severity));
	CurrentPolygon = &frame->Labels.back().Polygon;
	CurrentLabel   = &frame->Labels.back();
}

void UI::SetLabelDirty(train::ImageLabels* frame, train::Label* label, bool redrawOnCanvas) {
	label->Author   = UserName;
	label->EditTime = time::Now();
	// remove all patches that have no labels
	for (size_t i = frame->Labels.size() - 1; i != -1; i--) {
		if (frame->Labels[i].Classes.size() == 0) {
			frame->Labels.erase(frame->Labels.begin() + i);
		}
	}
	frame->SetDirty();
	if (redrawOnCanvas)
		DrawLabels();
}

void UI::GridDimensions(int& width, int& height) {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	width  = (int) floor((float) vwidth / (float) LabelGridSize);
	height = (int) floor((float) vheight / (float) LabelGridSize);
}

// If video is shown at half it's actual resolution, then scale = 0.5
xo::Vec2f UI::VideoScaleOnCanvas() {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	if (vwidth == 0)
		return xo::Vec2f(0, 0);
	int cwidth  = VideoCanvasWidth;
	int cheight = VideoCanvasHeight;
	/*
	// This doesn't work, because the DocUI system is busy using the layout, while it's processing events.
	// So long story short -- this stuff needs work, and we just have to hack around it.
	auto rd      = VideoCanvas->GetDoc()->Group->RenderDoc;
	auto layout  = rd->AcquireLatestLayout();
	if (layout) {
		auto node = layout->Node(VideoCanvas);
		if (node) {
			cwidth  = node->ContentWidthPx();
			cheight = node->ContentHeightPx();
		}
		rd->ReleaseLayout(layout);
	}
	*/
	/*
	if (cwidth == 1) {
		// legacy path, should never be used
		auto cx = VideoCanvas->GetCanvas2D();
		cwidth  = cx->Width();
		cheight = cx->Height();
		VideoCanvas->ReleaseCanvas(cx);
	}
	*/
	return xo::Vec2f(vwidth / (float) cwidth, vheight / (float) cheight);
}

xo::Point UI::CanvasToVideoPos(int x, int y) {
	auto scale = VideoScaleOnCanvas();
	return xo::Point((int) (x * scale.x), (int) (y * scale.y));
}

xo::Vec2f UI::VideoPosToCanvas(int x, int y) {
	auto scale = VideoScaleOnCanvas();
	return xo::Vec2f((float) x / scale.x, (float) y / scale.y);
}

xo::Point UI::GridPosOffset() {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	// horizontally: center the grid
	// vertically: we want to align it to the bottom of the screen
	xo::Point offset;
	offset.X = (vwidth % LabelGridSize) / 2;
	offset.Y = 0;
	return offset;
}

xo::Point UI::VideoPosToGrid(int x, int y) {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	auto offset = GridPosOffset();
	if (!GridTopDown)
		y = vheight - y;
	x -= offset.X;
	y -= offset.Y;
	int gx = (int) (x / LabelGridSize);
	int gy = (int) (y / LabelGridSize);
	int gwidth, gheight;
	GridDimensions(gwidth, gheight);
	gx = xo::Clamp(gx, 0, gwidth - 1);
	gy = xo::Clamp(gy, 0, gheight - 1);
	return xo::Point(gx, gy);
}

xo::Point UI::GridPosToVideo(int x, int y) {
	int vwidth, vheight;
	Video.Dimensions(vwidth, vheight);
	x *= LabelGridSize;
	y *= LabelGridSize;
	auto offset = GridPosOffset();
	x += offset.X;
	y += offset.Y;
	if (!GridTopDown)
		y = vheight - y;
	return xo::Point(x, y);
}

void UI::PrevFrame() {
	if (!Video.IsOpen())
		return;

	Video.SeekToPreviousFrame();
	NextFrame();
}

void UI::NextFrame() {
	if (!Video.IsOpen())
		return;

	LastFrameImg.Alloc(xo::TexFormatRGBA8, Video.Width(), Video.Height());
	Video.DecodeFrameRGBA(Video.Width(), Video.Height(), LastFrameImg.Data, LastFrameImg.Stride);

	DrawCurrentFrame();
}

void UI::DrawCurrentFrame() {
	if (!Video.IsOpen() || LastFrameImg.Width == 0)
		return;

	VideoCanvas->SetImageSizeOnly(LastFrameImg.Width, LastFrameImg.Height);
	auto cx = VideoCanvas->GetCanvas2D();
	cx->GetImage()->CopyFrom(&LastFrameImg);
	VideoCanvas->ReleaseAndInvalidate(cx);
	if (IsModeLabel)
		DrawLabels();
	else
		DrawEvalOverlay();
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
	LoadLabels();
	Render();
	NextFrame();
	return true;
}

std::string UI::ShortcutKeyForClass(const std::string& klass) {
	auto c = Taxonomy.FindClass(klass);
	if (!c)
		return "";
	return tsf::fmt("%c", (char) c->Key);
}

void UI::LoadLabels() {
	LoadVideoLabels(VideoFilename, Labels);
	// Perform any once-off fixups here

	//for (auto& f : Labels.Frames) {
	//	for (auto& lab : f.Labels) {
	//		if (lab.HasClass("edge")) {
	//			lab.RemoveClass("edge");
	//			f.SetDirty();
	//		}
	//	}
	//}

	//for (auto& f : Labels.Frames) {
	//	for (size_t i = f.Labels.size() - 1; i != -1; i--) {
	//		if (f.Labels[i].Class == "normal road") {
	//			f.Labels.erase(f.Labels.begin() + i);
	//		}
	//	}
	//	//f.SetDirty();
	//}
	//Labels.RemoveEmptyFrames();
}

} // namespace anno
} // namespace imqs
