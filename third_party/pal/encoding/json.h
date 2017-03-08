#pragma once

#include "../Error.h"

namespace imqs {
namespace json {
// if len is -1, then 'str' must be null terminated
IMQS_PAL_API Error RapidJsonFromString(const char* str, size_t len, rapidjson::Document& doc);
IMQS_PAL_API Error RapidJsonFromString(const std::string& str, rapidjson::Document& doc);
IMQS_PAL_API Error RapidJsonFromFile(const std::string& filename, rapidjson::Document& doc);
IMQS_PAL_API Error RapidJsonToFile(const rapidjson::Document& doc, const std::string& filename);

// Helpers to make building up Json documents less painful
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const char* value);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const std::string& value);
}
}