#pragma once

namespace imqs {
namespace train {

struct IMQS_TRAIN_API Point {
	int X = 0;
	int Y = 0;

	Point() {}
	Point(int x, int y) : X(x), Y(y) {}
};

struct IMQS_TRAIN_API Rect {
	int X1 = 0;
	int Y1 = 0;
	int X2 = 0;
	int Y2 = 0;

	Rect() {}
	Rect(int x1, int y1, int x2, int y2) : X1(x1), Y1(y1), X2(x2), Y2(y2) {}

	static Rect Inverted() { return Rect(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN); }

	void Expand(int x, int y) {
		X1 -= x;
		Y1 -= y;
		X2 += x;
		Y2 += y;
	}

	void ExpandToFit(int x, int y) {
		X1 = std::min(X1, x);
		Y1 = std::min(Y1, y);
		X2 = std::max(X2, x);
		Y2 = std::max(Y2, y);
	}

	void  Expand(int xy) { Expand(xy, xy); }
	Point Center() const { return Point((X1 + X2) / 2, (Y1 + Y2) / 2); }
	int   Width() const { return X2 - X1; }
	int   Height() const { return Y2 - Y1; }

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;
};

// A single labeled region inside an image
class IMQS_TRAIN_API Label {
public:
	Rect        Rect;
	std::string Class;
	std::string Labeler; // Person who created this label

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;
};

// Set of labels for an image (image is usually a single frame from a video)
class IMQS_TRAIN_API ImageLabels {
public:
	int64_t            Time = 0; // Video time in microseconds (0 = start of video)
	std::vector<Label> Labels;
	time::Time         EditTime;        // Time when this frame's labels were last edited
	bool               IsDirty = false; // Needs to be saved to disk

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;
	void  SetDirty();

	double TimeSeconds() const { return (double) Time / 1000000.0; }

	bool operator<(const ImageLabels& b) const { return Time < b.Time; }
};

// Labels for a video
class IMQS_TRAIN_API VideoLabels {
public:
	std::vector<ImageLabels> Frames;

	ImageLabels*                 FindFrame(int64_t time);
	ImageLabels*                 FindOrInsertFrame(int64_t time);
	ImageLabels*                 InsertFrame(int64_t time);
	void                         RemoveEmptyFrames();
	ohash::map<std::string, int> CategorizedLabelCount() const;
	std::vector<std::string>     Classes() const; // Classes are sorted alphabetically
	ohash::map<std::string, int> ClassToIndex() const;
	int64_t                      TotalLabelCount() const;
	nlohmann::json               ToJson() const;
};

// label class and associated shortcut key
class IMQS_TRAIN_API LabelClass {
public:
	int         Key = 0; // Shortcut key (Unicode character)
	std::string Class;   // Class label

	LabelClass() {}
	LabelClass(int key, const std::string& _class) : Key(key), Class(_class) {}

	std::string KeyStr() const;
};

IMQS_TRAIN_API std::string LabelFileDir(std::string videoFilename);
IMQS_TRAIN_API std::string ImagePatchDir(std::string videoFilename);
IMQS_TRAIN_API Error LoadVideoLabels(std::string videoFilename, VideoLabels& labels);
IMQS_TRAIN_API Error SaveFrameLabels(std::string videoFilename, const ImageLabels& frame);
IMQS_TRAIN_API int   MergeVideoLabels(const VideoLabels& src, VideoLabels& dst); // Returns number of new frames

} // namespace train
} // namespace imqs
