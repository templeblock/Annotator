#pragma once

namespace imqs {
namespace anno {

// This captures our UI state, and maintains the DOM state
class UI {
public:
	struct LoadSavePackage {
		std::string        VideoFilename;
		train::VideoLabels Labels;
	};

	enum class LabelModes {
		FixedBoxes,
		Segmentation,
	} LabelMode = LabelModes::FixedBoxes;

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
	xo::DomNode*      StatusLabel   = nullptr;
	xo::DomNode*      ErrorLabel    = nullptr;
	xo::DomNode*      LabelBox      = nullptr;
	xo::DomNode*      ToolsBox      = nullptr;
	int64_t           PlayTimer     = 0;
	int64_t           OnDestroyEv   = 0;
	std::string       VideoFilename;
	video::VideoFile  Video;
	xo::Image         LastFrameImg;
	std::thread       LoadSaveThread;
	std::atomic<bool> IsExiting;

	std::thread             ExportThread;
	xo::controls::MsgBox*   ExportMsgBox = nullptr;
	std::string             ExportProgMsg;
	train::ProgressCallback ExportCallback;
	std::function<void()>   ExportDlgClosed;

	std::string          UserName;
	size_t               UnlabeledClass = 0; // First class must be unlabeled
	train::LabelTaxonomy Taxonomy;
	train::VideoLabels   Labels;
	train::LabelClass    CurrentAssignClass;
	int                  CurrentAssignSeverity = 1;
	int                  LabelGridSize         = 256;
	bool                 GridTopDown           = false; // For road markings, we prefer bottom up, because the interesting stuff is at the bottom of the frame

#ifdef IMQS_AI_API
	// Inference
	AI::Model Model;
#endif
	std::string ModelLoadErr;
	bool        IsModeLabel = true; // else inference

	// Loading/Saving labels
	std::atomic<LoadSavePackage*> SaveQueue;         // Dirty frames waiting to be saved by save thread
	std::atomic<LoadSavePackage*> LoadQueue;         // Frames loaded by load/save thread. Waiting for us to merge them into our doc
	std::string                   LoadSaveError;     // Last load/save message. Guarded by LoadSaveErrorLock
	std::mutex                    LoadSaveErrorLock; // Guards access to LoadSaveError
	static void                   LoadSaveThreadFunc(UI* ui);

	UI(xo::DomNode* root);
	~UI();

	bool OpenVideo();
	void Render();

private:
	int             VideoCanvasWidth  = 0;
	int             VideoCanvasHeight = 0;
	xo::Vec2f       MouseDown;
	bool            IsMouseDragging       = false;
	bool            IsCreatingPolygon     = false;
	bool            IsDraggingVertex      = false;
	train::Polygon* CurrentPolygon        = nullptr;
	train::Label*   CurrentLabel          = nullptr;
	size_t          CurrentDraggingVertex = -1;

	void RenderTimeSlider(bool first = false);
	void FlipPlayState();
	void SeekFromUI(xo::Event& ev);
	void Play();
	void Stop();
	void DrawLabels();
	void DrawLabelBoxes();
	void DrawSegmentationLabels();
	void DrawPolygon(xo::Canvas2D* cx, const train::Polygon& poly, std::vector<float>* tmpVx = nullptr);
	void DrawEvalOverlay();
	void SetCurrentLabel(train::LabelClass c);
	void GridDimensions(int& width, int& height);
	void PrevFrame();
	void NextFrame();
	void DrawCurrentFrame();
	void LoadLabels();
	void RenderToolsUI();
	void RenderLabelUI();
	void SetLabelDirty(train::ImageLabels* frame, train::Label* label, bool redrawOnCanvas = true);
	void CreateNewPolygonLabel(std::string klass, int severity);

	train::ImageLabels* LabelsForCurrentFrame(bool create);

	void          Segmentation_MouseDrag(xo::Event& ev, xo::Point vposMouseDown, xo::Point vpos);
	void          Segmentation_FindCloseVertex(xo::Point vpos, train::Label*& lab, size_t& ivertex, bool createVertexIfNone);
	train::Label* Segmentation_FindCloseLabel(xo::Point vpos);
	void          Segmentation_DeleteLabel(xo::Point vpos);

	std::string ShortcutKeyForClass(const std::string& klass);

	xo::Vec2f VideoScaleOnCanvas();
	xo::Point CanvasToVideoPos(int x, int y);
	xo::Vec2f VideoPosToCanvas(int x, int y);
	xo::Point VideoPosToGrid(int x, int y);
	xo::Point GridPosToVideo(int x, int y);
	xo::Point GridPosOffset();

	void OnPaintLabel(xo::Event& ev);
	void OnKeyChar(xo::Event& ev);
	void OnLoadSaveTimer();

	// The labels in our label store use raw video pixel coordinate rectangles.
	// Here we map those arbitrary rectangles back to our fixed grid.
	ohash::map<uint64_t, train::Label> ExtractLabelsForGrid();
	train::Label*                      FindOrInsertLabel(train::ImageLabels* frame, int gridX, int gridY);
	bool                               RemoveLabel(train::ImageLabels* frame, int gridX, int gridY); // Returns true if a label was found and removed

	uint64_t MakeGridCoord64(int x, int y) {
		return ((uint64_t) x << 32) | (uint64_t) y;
	}
};

} // namespace anno
} // namespace imqs
