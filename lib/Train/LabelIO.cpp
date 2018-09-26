#include "pch.h"
#include "LabelIO.h"

using namespace std;
using json = nlohmann::json;

// This seems kinda neat, but I don't see a benefit here beyond plain old To/From methods
/*
namespace nlohmann {

template <>
struct adl_serializer<imqs::anno::Rect> {
	static void to_json(json& j, const imqs::anno::Rect& rect) {
		j["x1"] = rect.X1;
		j["y1"] = rect.Y1;
		j["x2"] = rect.X2;
		j["y2"] = rect.Y2;
	}

	static void from_json(const json& j, imqs::anno::Rect& rect) {
		if (j.is_null())
			return;
		rect.X1 = j["x1"];
		rect.Y1 = j["y1"];
		rect.X2 = j["x2"];
		rect.Y2 = j["y2"];
	}
};

template <>
struct adl_serializer<imqs::anno::Label> {
	static void to_json(json& j, const imqs::anno::Label& lab) {
		json jr;
		adl_serializer<imqs::anno::Rect>::to_json(jr, lab.Rect);
		j["rect"] = jr;
	}

	static void from_json(const json& j, imqs::anno::Label& lab) {
		if (j.is_null())
			return;
	}
};

} // namespace nlohmann
*/

namespace imqs {
namespace train {

Error Rect::FromJson(const nlohmann::json& j) {
	X1 = j["x1"];
	Y1 = j["y1"];
	X2 = j["x2"];
	Y2 = j["y2"];
	return Error();
}

void Rect::ToJson(nlohmann::json& j) const {
	j["x1"] = X1;
	j["y1"] = Y1;
	j["x2"] = X2;
	j["y2"] = Y2;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Error Polygon::FromJson(const nlohmann::json& j) {
	if (!j.is_array() || j.size() < 6 || j.size() % 2 != 0)
		return Error("Polygon must be an array of X,Y pairs, and have at least 3 vertices");
	for (size_t i = 0; i < j.size(); i += 2) {
		int x = j[i].get<int>();
		int y = j[i + 1].get<int>();
		Vertices.push_back(Point(x, y));
	}
	return Error();
}

void Polygon::ToJson(nlohmann::json& j) const {
	for (const auto& v : Vertices) {
		j.push_back((int) v.X);
		j.push_back((int) v.Y);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Error ClassSeverity::FromJson(const nlohmann::json& j) {
	if (!j.is_array() || j.size() != 2)
		return Error("ClassSeverity pair is not an array of two elements");
	Class    = j[0];
	Severity = j[1];
	return Error();
}

void ClassSeverity::ToJson(nlohmann::json& j) const {
	j.push_back(Class);
	j.push_back(Severity);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Error Label::FromJson(const nlohmann::json& j) {
	Classes.clear();
	for (const auto& jc : j["classes"]) {
		Classes.push_back({});
		auto err = Classes.back().FromJson(jc);
		if (!err.OK())
			return err;
	}
	Author = j["labeler"];
	if (j.find("rect") != j.end())
		Rect.FromJson(j["rect"]);
	if (j.find("poly") != j.end())
		Polygon.FromJson(j["poly"]);
	return Error();
}

void Label::ToJson(nlohmann::json& j) const {
	nlohmann::json jclasses;
	for (const auto& c : Classes) {
		nlohmann::json jc;
		c.ToJson(jc);
		jclasses.push_back(std::move(jc));
	}
	j["classes"]  = std::move(jclasses);
	j["labeler"]  = Author;
	j["edittime"] = EditTime.Unix();
	if (Rect.Width() != 0)
		Rect.ToJson(j["rect"]);
	if (Polygon.Vertices.size() >= 3)
		Polygon.ToJson(j["poly"]);
}

bool Label::HasClass(const std::string& _class) const {
	for (const auto& c : Classes) {
		if (c.Class == _class)
			return true;
	}
	return false;
}

int Label::Severity(const std::string& _class) const {
	for (const auto& c : Classes) {
		if (c.Class == _class)
			return c.Severity;
	}
	return -1;
}

void Label::RemoveClass(const std::string& _class) {
	for (size_t i = 0; i < Classes.size(); i++) {
		if (Classes[i].Class == _class) {
			Classes.erase(Classes.begin() + i);
			return;
		}
	}
}

void Label::SetClass(const std::string& _class, int severity) {
	RemoveClass(_class);
	Classes.emplace_back(_class, severity);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Error ImageLabels::FromJson(const nlohmann::json& j) {
	// Originally we stored edit time for an entire frame.
	// Later we switched to storing edit time for individual labels.
	auto       frameEditTime = j.find("edittime");
	time::Time editTime;
	if (frameEditTime != j.end())
		editTime = time::Time::FromUnix(*frameEditTime, 0);

	for (const auto& jlab : j["labels"]) {
		Label lab;
		auto  err = lab.FromJson(jlab);
		if (!err.OK())
			return err;
		if (lab.EditTime.IsNull())
			lab.EditTime = editTime;
		Labels.emplace_back(std::move(lab));
	}
	return Error();
}

void ImageLabels::ToJson(nlohmann::json& j) const {
	auto& jlabels = j["labels"];
	for (const auto& lab : Labels) {
		json jlab;
		lab.ToJson(jlab);
		jlabels.emplace_back(std::move(jlab));
	}
}

void ImageLabels::SetDirty() {
	IsDirty = true;
}

time::Time ImageLabels::MaxEditTime() const {
	if (Labels.size() == 0)
		return time::Time(2000, time::Month::January, 1, 0, 0, 0, 0);

	auto maxT = Labels[0].EditTime;
	for (const auto& lab : Labels) {
		if (lab.EditTime > maxT)
			maxT = lab.EditTime;
	}
	return maxT;
}

bool ImageLabels::HasRects() const {
	for (const auto& lab : Labels) {
		if (lab.IsRect())
			return true;
	}
	return false;
}

bool ImageLabels::HasPolygons() const {
	for (const auto& lab : Labels) {
		if (lab.IsPolygon())
			return true;
	}
	return false;
}

Label* ImageLabels::FindLabel(Rect rect, bool createIfNotExist) {
	for (auto& lab : Labels) {
		if (lab.IsRect() && lab.Rect == rect)
			return &lab;
	}
	if (createIfNotExist) {
		Labels.push_back(Label());
		Labels.back().Rect = rect;
		return &Labels.back();
	}
	return nullptr;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static int CompareFrame(const ImageLabels& f, const int64_t& time) {
	if (f.Time < time)
		return -1;
	if (f.Time > time)
		return 1;
	return 0;
}

ImageLabels* VideoLabels::FindFrame(int64_t time) {
	if (Frames.size() == 0)
		return nullptr;
	auto i = algo::BinarySearch(Frames.size(), &Frames[0], time, CompareFrame);
	if (i == -1)
		return nullptr;
	return &Frames[i];
}

ImageLabels* VideoLabels::FindOrInsertFrame(int64_t time) {
	auto f = FindFrame(time);
	if (f)
		return f;
	return InsertFrame(time);
}

ImageLabels* VideoLabels::InsertFrame(int64_t time) {
	size_t i = 0;
	if (Frames.size() != 0)
		i = algo::BinarySearchTry(Frames.size(), &Frames[0], time, CompareFrame);
	ImageLabels frame;
	frame.Time = time;
	Frames.insert(Frames.begin() + i, frame);
	return &Frames[i];
}

void VideoLabels::RemoveEmptyFrames() {
	for (size_t i = Frames.size() - 1; i != -1; i--) {
		if (Frames[i].Labels.size() == 0)
			Frames.erase(Frames.begin() + i);
	}
}

ohash::map<std::string, int> VideoLabels::CategorizedLabelCount(ohash::map<std::string, std::string>* groups) const {
	ohash::map<std::string, int> counts;
	for (const auto& f : Frames) {
		for (const auto& lab : f.Labels) {
			for (const auto& c : lab.Classes) {
				if (groups) {
					auto group = groups->get(c.Class);
					if (group == "")
						counts["UNGROUPED:" + c.Class]++;
					else
						counts[group]++;
				} else {
					counts[c.Class]++;
				}
			}
		}
	}
	return counts;
}

std::vector<std::string> VideoLabels::Classes() const {
	ohash::set<std::string> cset;
	for (const auto& f : Frames) {
		for (const auto& lab : f.Labels) {
			for (const auto& c : lab.Classes)
				cset.insert(c.Class);
		}
	}
	std::vector<std::string> classes;
	for (const auto& c : cset)
		classes.push_back(c);

	std::sort(classes.begin(), classes.end());
	return classes;
}

ohash::map<std::string, int> VideoLabels::ClassToIndex() const {
	auto                         classes = Classes();
	ohash::map<std::string, int> map;
	for (size_t i = 0; i < classes.size(); i++)
		map.insert(classes[i], (int) i);
	return map;
}

int64_t VideoLabels::TotalLabelCount() const {
	int64_t c = 0;
	for (const auto& f : Frames)
		c += f.Labels.size();
	return c;
}

int64_t VideoLabels::TotalPolygonFrameCount() const {
	int64_t c = 0;
	for (const auto& f : Frames)
		c += f.HasPolygons() ? 1 : 0;
	return c;
}

nlohmann::json VideoLabels::ToJson() const {
	nlohmann::json jframes;
	for (const auto& f : Frames) {
		nlohmann::json jframe;
		jframe["time"] = f.Time;
		nlohmann::json jlabels;
		for (const auto& lab : f.Labels) {
			nlohmann::json jlab;
			lab.ToJson(jlab);
			jlabels.push_back(std::move(jlab));
		}
		jframe["labels"] = std::move(jlabels);
		jframes.push_back(std::move(jframe));
	}
	return {{"frames", jframes}};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string LabelClass::KeyStr() const {
	char buf[2] = {(char) Key, 0};
	return buf;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ohash::map<std::string, int> LabelTaxonomy::PatchClassToIndex() const {
	ohash::map<std::string, int> map;
	int                          ilab = 0;
	for (const auto& c : Classes) {
		if (!c.IsPolygon)
			map.insert(c.Class, ilab++);
	}
	return map;
}

ohash::map<std::string, int> LabelTaxonomy::SegmentationClassToIndex() const {
	// for segmentation, we always need a 'null' label, which is label 0
	ohash::map<std::string, int> map;
	map.insert("unlabeled", 0);
	int ilab = 1;
	for (const auto& c : Classes) {
		if (c.IsPolygon)
			map.insert(c.Class, ilab++);
	}
	return map;
}

size_t LabelTaxonomy::FindClassIndex(const std::string& klass) const {
	for (size_t i = 0; i < Classes.size(); i++) {
		if (Classes[i].Class == klass)
			return i;
	}
	return -1;
}

const LabelClass* LabelTaxonomy::FindClass(const std::string& klass) const {
	size_t i = FindClassIndex(klass);
	return i == -1 ? nullptr : &Classes[i];
}

std::vector<std::string> LabelTaxonomy::ClassesInGroup(std::string group) const {
	std::vector<std::string> cg;
	for (const auto& c : Classes) {
		if (c.Group == group)
			cg.push_back(c.Class);
	}
	return cg;
}

std::vector<std::vector<train::LabelClass>> LabelTaxonomy::ClassGroups() const {
	std::vector<std::vector<train::LabelClass>> groups;
	string                                      last = "not a group";
	for (size_t i = 0; i < Classes.size(); i++) {
		if (Classes[i].Group != last) {
			groups.push_back({});
		}
		groups.back().push_back(Classes[i]);
		last = Classes[i].Group;
	}
	return groups;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IMQS_TRAIN_API std::string LabelFileDir(std::string videoFilename, std::string modelName) {
	return path::Dir(videoFilename) + "/labels-" + modelName + "/" + path::Filename(videoFilename);
}

IMQS_TRAIN_API std::string ImagePatchDir(std::string videoFilename) {
	return path::Dir(videoFilename) + "/patches/" + path::Filename(videoFilename);
}

IMQS_TRAIN_API bool VideoFileHasLabels(std::string videoFilename, std::string modelName) {
	os::FileAttributes at;
	return os::Stat(LabelFileDir(videoFilename, modelName), at).OK();
}

IMQS_TRAIN_API Error LoadVideoLabels(std::string videoFilename, std::string modelName, VideoLabels& labels) {
	labels.Frames.clear();
	auto  dir = LabelFileDir(videoFilename, modelName);
	Error err;
	auto  errFind = os::FindFiles(dir, [&err, &labels](const os::FindFileItem& item) -> bool {
        if (item.IsDir)
            return false;
        // 000001.json
        if (strings::EndsWith(item.Name, ".json")) {
            std::string buf;
            err = os::ReadWholeFile(item.FullPath(), buf);
            if (!err.OK())
                return false;
            auto        j = json::parse(buf.c_str());
            ImageLabels frame;
            frame.Time = AtoI64(item.Name.c_str());
            err        = frame.FromJson(j);
            if (!err.OK())
                return false;
            labels.Frames.emplace_back(std::move(frame));
        }
        return true;
    });
	std::sort(labels.Frames.begin(), labels.Frames.end());

	if (!errFind.OK())
		return errFind;
	return err;
}

IMQS_TRAIN_API Error SaveVideoLabels(std::string videoFilename, std::string modelName, const VideoLabels& labels) {
	for (const auto& f : labels.Frames) {
		auto err = SaveFrameLabels(videoFilename, modelName, f);
		if (!err.OK())
			return err;
	}
	return Error();
}

// Save one labeled frame. By saving at image granularity instead of video granularity, we allow concurrent labeling by many people
IMQS_TRAIN_API Error SaveFrameLabels(std::string videoFilename, std::string modelName, const ImageLabels& frame) {
	auto dir = LabelFileDir(videoFilename, modelName);
	auto err = os::MkDirAll(dir);
	if (!err.OK())
		return err;
	json j;
	frame.ToJson(j);
	auto fn = tsf::fmt("%v.json", frame.Time);
	return os::WriteWholeFile(dir + "/" + fn, j.dump(4));
}

IMQS_TRAIN_API int MergeVideoLabels(const VideoLabels& src, VideoLabels& dst) {
	int nnew = 0;
	for (const auto& sframe : src.Frames) {
		auto dframe = dst.FindOrInsertFrame(sframe.Time);
		if (sframe.MaxEditTime() > dframe->MaxEditTime()) {
			nnew++;
			*dframe = sframe;
		}
	}
	return nnew;
}

IMQS_TRAIN_API Error ExportClassTaxonomy(std::string filename, std::vector<LabelClass> classes) {
	json   root;
	string lastGroup = "not a group";
	for (auto c : classes) {
		if (c.Group == "")
			continue;
		if (c.Group != lastGroup) {
			root[c.Group] = json();
			lastGroup     = c.Group;
		}
		json item = {
		    {"name", c.Class},
		    {"hasSeverity", c.HasSeverity},
		};
		root[c.Group].push_back(item);
	}
	return os::WriteWholeFile(filename, root.dump(4));
}

IMQS_TRAIN_API ohash::map<std::string, std::string> ClassToGroupMap(std::vector<LabelClass> classes) {
	ohash::map<std::string, std::string> map;
	for (auto c : classes)
		map.insert(c.Class, c.Group);
	return map;
}

} // namespace train
} // namespace imqs
