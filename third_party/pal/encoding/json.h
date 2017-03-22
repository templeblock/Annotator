#pragma once

#include "../Error.h"

namespace imqs {

// Rapidjson helper functions
namespace rj {
// if len is -1, then 'str' must be null terminated
IMQS_PAL_API Error ParseString(const char* str, size_t len, rapidjson::Document& doc);
IMQS_PAL_API Error ParseString(const std::string& str, rapidjson::Document& doc);
IMQS_PAL_API Error ParseFile(const std::string& filename, rapidjson::Document& doc);
IMQS_PAL_API Error WriteFile(const rapidjson::Document& doc, const std::string& filename);

IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, const char* value, rapidjson::MemoryPoolAllocator<>* allocator);
IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, const std::string& value, rapidjson::MemoryPoolAllocator<>* allocator);
IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, bool value, rapidjson::MemoryPoolAllocator<>* allocator);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const char* value);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const std::string& value);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, bool value);
IMQS_PAL_API bool GetBool(const rapidjson::Value& v, const char* key, bool defaultValue = false);
IMQS_PAL_API int  GetInt(const rapidjson::Value& v, const char* key, int defaultValue = 0);
IMQS_PAL_API std::vector<std::string> Keys(const rapidjson::Value& v);
}
}