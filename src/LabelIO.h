#pragma once

namespace imqs {
namespace anno {

struct Point {
	int X = 0;
	int Y = 0;

	Point() {}
	Point(int x, int y) : X(x), Y(y) {}
};

struct Rect {
	int X1 = 0;
	int Y1 = 0;
	int X2 = 0;
	int Y2 = 0;

	Rect() {}
	Rect(int x1, int y1, int x2, int y2) : X1(x1), Y1(y1), X2(x2), Y2(y2) {}

	void Expand(int x, int y) {
		X1 -= x;
		Y1 -= y;
		X2 += x;
		Y2 += y;
	}

	void  Expand(int xy) { Expand(xy, xy); }
	Point Center() const { return Point((X1 + X2) / 2, (Y1 + Y2) / 2); }

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;
};

// A single labeled region inside an image
class Label {
public:
	Rect        Rect;
	std::string Class;

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;
};

// Set of labels for an image (image is usually a single frame from a video)
class ImageLabels {
public:
	int64_t            Time = 0; // Time in ffmpeg units AV_TIME_BASE - which is millionths of a second
	std::vector<Label> Labels;
	bool               IsDirty = false; // Needs to be saved to disk

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;

	bool operator<(const ImageLabels& b) const { return Time < b.Time; }
};

// Labels for a video
class VideoLabels {
public:
	std::vector<ImageLabels> Frames;

	ImageLabels* FindFrame(int64_t time);
	ImageLabels* FindOrInsertFrame(int64_t time);
	ImageLabels* InsertFrame(int64_t time);
};

// label class and associated shortcut key
class LabelClass {
public:
	int         Key = 0; // Shortcut key (Unicode character)
	std::string Class;   // Class label

	LabelClass() {}
	LabelClass(int key, const std::string& _class) : Key(key), Class(_class) {}

	std::string KeyStr() const;
};

std::string LabelFileDir(std::string videoFilename);
Error       LoadVideoLabels(std::string videoFilename, VideoLabels& labels);
Error       SaveFrameLabels(std::string videoFilename, const ImageLabels& frame);

} // namespace anno
} // namespace imqs
