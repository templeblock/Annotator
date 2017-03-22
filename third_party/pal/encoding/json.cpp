#include "pch.h"
#include "json.h"
#include "../os/os.h"

namespace imqs {
namespace rj {

IMQS_PAL_API Error ParseString(const char* str, size_t len, rapidjson::Document& doc) {
	if (len == -1)
		doc.Parse(str);
	else
		doc.Parse(str, len);

	if (doc.HasParseError())
		return Error(tsf::fmt("JSON parse error '%s'", rapidjson::GetParseError_En(doc.GetParseError())));

	return Error();
}

IMQS_PAL_API Error ParseString(const std::string& str, rapidjson::Document& doc) {
	return ParseString(str.c_str(), str.length(), doc);
}

IMQS_PAL_API Error ParseFile(const std::string& filename, rapidjson::Document& doc) {
	std::string buf;
	auto        err = os::ReadWholeFile(filename, buf);
	if (!err.OK())
		return err;
	return ParseString(buf, doc);
}

IMQS_PAL_API Error WriteFile(const rapidjson::Document& doc, const std::string& filename) {
	rapidjson::StringBuffer                    buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	doc.Accept(writer);
	return os::WriteWholeFile(filename, buffer.GetString(), buffer.GetSize());
}

IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, const char* value, rapidjson::MemoryPoolAllocator<>* allocator) {
	obj.AddMember(rapidjson::Value(key, *allocator).Move(), rapidjson::Value(value, *allocator).Move(), *allocator);
}

IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, const std::string& value, rapidjson::MemoryPoolAllocator<>* allocator) {
	Set(obj, key, value.c_str(), allocator);
}

IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, bool value, rapidjson::MemoryPoolAllocator<>* allocator) {
	obj.AddMember(rapidjson::Value(key, *allocator).Move(), value, *allocator);
}

IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const char* value) {
	Set(doc, key, value, &doc.GetAllocator());
}

IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const std::string& value) {
	Set(doc, key, value.c_str(), &doc.GetAllocator());
}

IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, bool value) {
	Set(doc, key, value, &doc.GetAllocator());
}

IMQS_PAL_API bool GetBool(const rapidjson::Value& v, const char* key, bool defaultValue) {
	auto iter = v.FindMember(key);
	if (iter != v.MemberEnd())
		return iter->value.GetBool();
	else
		return defaultValue;
}

IMQS_PAL_API int GetInt(const rapidjson::Value& v, const char* key, int defaultValue) {
	auto iter = v.FindMember(key);
	if (iter != v.MemberEnd())
		return iter->value.GetInt();
	else
		return defaultValue;
}

IMQS_PAL_API std::vector<std::string> Keys(const rapidjson::Value& v) {
	std::vector<std::string> vals;
	for (const auto& m : v.GetObject())
		vals.push_back(m.name.GetString());
	return vals;
}

}
}