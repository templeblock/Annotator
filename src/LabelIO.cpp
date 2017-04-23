#include "pch.h"
#include "LabelIO.h"

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
namespace anno {

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

Error Label::FromJson(const nlohmann::json& j) {
	Class = j["class"];
	Rect.FromJson(j["rect"]);
	return Error();
}

void Label::ToJson(nlohmann::json& j) const {
	j["class"] = Class;
	Rect.ToJson(j["rect"]);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Error ImageLabels::FromJson(const nlohmann::json& j) {
	for (const auto& jlab : j["labels"]) {
		Label lab;
		auto  err = lab.FromJson(jlab);
		if (!err.OK())
			return err;
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
	auto frame = ImageLabels();
	frame.Time = time;
	Frames.insert(Frames.begin() + i, frame);
	return &Frames[i];
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

std::string LabelFileDir(std::string videoFilename) {
	return path::Dir(videoFilename) + "/labels/" + path::Filename(videoFilename);
}

Error LoadVideoLabels(std::string videoFilename, VideoLabels& labels) {
	labels.Frames.clear();
	auto  dir = LabelFileDir(videoFilename);
	Error err;
	auto  errFind = os::FindFiles(dir, [&err, &labels](const os::FindFileItem& item) -> bool {
		// 000001.json
		if (strings::EndsWith(item.Name, ".json")) {
			std::string buf;
			err = os::ReadWholeFile(item.FullPath(), buf);
			if (!err.OK())
				return false;
			auto        j = json::parse(buf.c_str());
			ImageLabels frame;
			frame.Time = AtoI64(item.Name.c_str());
			//nlohmann::adl_serializer<Rect>::from_json(j, r);
			err = frame.FromJson(j);
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

// Save one labeled frame. By saving at image granularity instead of video granularity, we allow concurrent labeling by many people
Error SaveFrameLabels(std::string videoFilename, const ImageLabels& frame) {
	auto dir = LabelFileDir(videoFilename);
	auto err = os::MkDirAll(dir);
	if (!err.OK())
		return err;
	json j;
	frame.ToJson(j);
	auto fn = tsf::fmt("%v.json", frame.Time);
	return os::WriteWholeFile(dir + "/" + fn, j.dump(4));
}

} // namespace anno
} // namespace imqs
