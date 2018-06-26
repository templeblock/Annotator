#pragma once

namespace imqs {
namespace train {

struct IMQS_TRAIN_API Point {
	int X = 0;
	int Y = 0;

	Point() {}
	Point(int x, int y) : X(x), Y(y) {}

	float Distance(int x, int y) const { return sqrt((float) (x - X) * (x - X) + (float) (y - Y) * (y - Y)); }

	operator gfx::Vec2f() const {
		return gfx::Vec2f((float) X, (float) Y);
	}
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

// A polygon has 3 or more vertices, and is implicitly closed. The final vertex is not repeated.
class IMQS_TRAIN_API Polygon {
public:
	std::vector<Point> Vertices;

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;
};

// A class and severity pair
class IMQS_TRAIN_API ClassSeverity {
public:
	std::string Class;
	int         Severity = 0; // 1..5, for classes that have severity defined

	ClassSeverity() {}
	ClassSeverity(const std::string& c, int s) : Class(c), Severity(s) {}

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;
};

// A single labeled region inside an image
class IMQS_TRAIN_API Label {
public:
	// Either Rect or Polygon is populated
	Rect                       Rect;
	Polygon                    Polygon;
	std::vector<ClassSeverity> Classes;
	std::string                Author;   // Person who last edited this label
	time::Time                 EditTime; // Time when this label was created

	Error FromJson(const nlohmann::json& j);
	void  ToJson(nlohmann::json& j) const;
	bool  HasClass(const std::string& _class) const;
	int   Severity(const std::string& _class) const; // Returns severity, or -1 if class does not exist
	void  RemoveClass(const std::string& _class);
	void  SetClass(const std::string& _class, int severity);
	bool  IsRect() const { return Rect.Width() != 0; }
	bool  IsPolygon() const { return Polygon.Vertices.size() != 0; }
};

// Set of labels for an image (image is usually a single frame from a video)
class IMQS_TRAIN_API ImageLabels {
public:
	int64_t            Time = 0;        // Frame time in microseconds
	std::vector<Label> Labels;          // Labels
	bool               IsDirty = false; // Needs to be saved to disk

	Error      FromJson(const nlohmann::json& j);
	void       ToJson(nlohmann::json& j) const;
	void       SetDirty();
	time::Time MaxEditTime() const; // Max edit time of labels inside this frame
	bool       HasRects() const;
	bool       HasPolygons() const;

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
	ohash::map<std::string, int> CategorizedLabelCount(ohash::map<std::string, std::string>* groups = nullptr) const;
	std::vector<std::string>     Classes() const; // Classes are sorted alphabetically
	ohash::map<std::string, int> ClassToIndex() const;
	int64_t                      TotalLabelCount() const;
	int64_t                      TotalPolygonFrameCount() const;
	nlohmann::json               ToJson() const;
};

// label class and associated shortcut key
class IMQS_TRAIN_API LabelClass {
public:
	int         Key = 0; // Shortcut key (Unicode character)
	std::string Group;   // Only one label allowed per group
	std::string Class;   // Class label
	bool        HasSeverity = false;
	bool        IsPolygon   = false;

	LabelClass() {}
	LabelClass(bool isPolygon, bool hasSeverity, int key, std::string group, std::string _class) : IsPolygon(isPolygon), HasSeverity(hasSeverity), Key(key), Group(group), Class(_class) {}

	std::string KeyStr() const;
};

class IMQS_TRAIN_API LabelTaxonomy {
public:
	std::vector<LabelClass> Classes;

	ohash::map<std::string, int>                PatchClassToIndex() const;
	ohash::map<std::string, int>                SegmentationClassToIndex() const;
	size_t                                      FindClassIndex(const std::string& klass) const;
	const LabelClass*                           FindClass(const std::string& klass) const;
	std::vector<std::string>                    ClassesInGroup(std::string group) const;
	std::vector<std::vector<train::LabelClass>> ClassGroups() const;
};

IMQS_TRAIN_API std::string LabelFileDir(std::string videoFilename);
IMQS_TRAIN_API std::string ImagePatchDir(std::string videoFilename);
IMQS_TRAIN_API Error LoadVideoLabels(std::string videoFilename, VideoLabels& labels);
IMQS_TRAIN_API Error SaveVideoLabels(std::string videoFilename, const VideoLabels& labels);
IMQS_TRAIN_API Error SaveFrameLabels(std::string videoFilename, const ImageLabels& frame);
IMQS_TRAIN_API int   MergeVideoLabels(const VideoLabels& src, VideoLabels& dst); // Returns number of new frames
IMQS_TRAIN_API Error ExportClassTaxonomy(std::string filename, std::vector<LabelClass> classes);
IMQS_TRAIN_API ohash::map<std::string, std::string> ClassToGroupMap(std::vector<LabelClass> classes);

} // namespace train
} // namespace imqs
