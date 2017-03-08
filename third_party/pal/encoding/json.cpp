#include "pch.h"
#include "json.h"
#include "../os/os.h"

namespace imqs {
namespace json {

IMQS_PAL_API Error RapidJsonFromString(const char* str, size_t len, rapidjson::Document& doc) {
	if (len == -1)
		doc.Parse(str);
	else
		doc.Parse(str, len);

	if (doc.HasParseError())
		return Error(tsf::fmt("JSON parse error '%s'", rapidjson::GetParseError_En(doc.GetParseError())));

	return Error();
}

IMQS_PAL_API Error RapidJsonFromString(const std::string& str, rapidjson::Document& doc) {
	return RapidJsonFromString(str.c_str(), str.length(), doc);
}

IMQS_PAL_API Error RapidJsonFromFile(const std::string& filename, rapidjson::Document& doc) {
	std::string buf;
	auto        err = os::ReadWholeFile(filename, buf);
	if (!err.OK())
		return err;
	return RapidJsonFromString(buf, doc);
}

IMQS_PAL_API Error RapidJsonToFile(const rapidjson::Document& doc, const std::string& filename) {
	rapidjson::StringBuffer                    buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	doc.Accept(writer);
	return os::WriteWholeFile(filename, buffer.GetString(), buffer.GetSize());
}

IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const char* value) {
	doc.AddMember(rapidjson::Value(key, doc.GetAllocator()).Move(), rapidjson::Value(value, doc.GetAllocator()).Move(), doc.GetAllocator());
}

IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const std::string& value) {
	Set(doc, key, value.c_str());
}

}
}