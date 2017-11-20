#pragma once

namespace imqs {
namespace xml {

class IMQS_PAL_API Serializer {
public:
	static bool GetAttrib(tinyxml2::XMLElement* el, const char* name, std::string& val);
};

IMQS_PAL_API Error ToError(tinyxml2::XMLError err);
IMQS_PAL_API Error ParseFile(const std::string& filename, tinyxml2::XMLDocument& doc);
IMQS_PAL_API Error ParseString(const std::string& str, tinyxml2::XMLDocument& doc);
IMQS_PAL_API Error ParseString(const char* str, tinyxml2::XMLDocument& doc, size_t strLen = -1);

} // namespace xml
} // namespace imqs