#pragma once

#include "VideoDecode.h"
#include "LabelIO.h"

namespace imqs {
namespace anno {

// This captures our UI state, and maintains the DOM state
class UI {
public:
	struct SavePackage {
		std::string VideoFilename;
		VideoLabels Labels;
	};

	enum class PlayStates {
		Stop,
		Play
	} PlayState = PlayStates::Stop;

	enum class ActionStates {
		None,
		AssignLabel,
	} ActionState = ActionStates::None;

	xo::DomNode*      Root          = nullptr;
	xo::DomNode*      TimeSliderBox = nullptr;
	xo::DomCanvas*    VideoCanvas   = nullptr;
	xo::DomNode*      PlayBtn       = nullptr;
	int64_t           PlayTimer     = 0;
	int64_t           OnDestroyEv   = 0;
	std::string       VideoFilename;
	VideoFile         Video;
	xo::Image         LastFrameImg;
	std::thread       SaveThread;
	std::atomic<bool> IsExiting;

	size_t                  UnlabeledClass = 0; // First class must be unlabeled
	std::vector<LabelClass> Classes;
	VideoLabels             Labels;
	LabelClass              CurrentAssignClass;
	int                     LabelGridSize = 128;
	bool                    GridTopDown   = false; // For road markings, we prefer bottom up, because the interesting stuff is at the bottom of the frame

	// Saving
	std::atomic<SavePackage*> SaveQueue; // Dirty frames waiting to be saved by save thread
	std::atomic<const char*>  SaveError; // Last save message

	static void SaveThreadFunc(UI* ui);

	UI(xo::DomNode* root);
	~UI();

	void   Render();
	void   RenderTimeSlider(bool first = false);
	void   FlipPlayState();
	void   SeekFromUI(xo::Event& ev);
	void   Play();
	void   Stop();
	void   DrawLabelBoxes();
	void   AssignLabel(LabelClass c);
	void   GridDimensions(int& width, int& height);
	void   NextFrame();
	bool   OpenVideo();
	size_t FindClass(const std::string& klass);

private:
	xo::Vec2f VideoScaleOnCanvas();
	xo::Point CanvasToVideoPos(int x, int y);
	xo::Point VideoPosToGrid(int x, int y);
	xo::Point GridPosToVideo(int x, int y);

	void OnPaintLabel(xo::Event& ev);
	void OnKeyChar(xo::Event& ev);
	void OnSaveTimer();

	// The labels in our label store use raw video pixel coordinate rectangles.
	// Here we map those arbitrary rectangles back to our fixed grid.
	ohash::map<uint64_t, Label> ExtractLabelsForGrid();
	Label*                      FindOrInsertLabel(ImageLabels* frame, int gridX, int gridY);
	bool                        RemoveLabel(ImageLabels* frame, int gridX, int gridY); // Returns true if a label was found and removed

	uint64_t MakeGridCoord64(int x, int y) {
		return ((uint64_t) x << 32) | (uint64_t) y;
	}
};

} // namespace anno
} // namespace imqs
