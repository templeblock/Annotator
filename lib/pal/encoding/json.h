#pragma once

namespace imqs {

#ifndef IMQS_PAL_DISABLE_RAPIDJSON
// Rapidjson helper functions
namespace rj {
// if len is -1, then 'str' must be null terminated
IMQS_PAL_API Error ParseString(const char* str, size_t len, rapidjson::Document& doc);
IMQS_PAL_API Error ParseString(const std::string& str, rapidjson::Document& doc);
IMQS_PAL_API Error ParseFile(const std::string& filename, rapidjson::Document& doc);
IMQS_PAL_API std::string WriteString(const rapidjson::Document& doc);
IMQS_PAL_API Error WriteFile(const rapidjson::Document& doc, const std::string& filename);

IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, const char* value, rapidjson::MemoryPoolAllocator<>* allocator);
IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, const std::string& value, rapidjson::MemoryPoolAllocator<>* allocator);
IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, bool value, rapidjson::MemoryPoolAllocator<>* allocator);
IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, int64_t value, rapidjson::MemoryPoolAllocator<>* allocator);
IMQS_PAL_API void Set(rapidjson::Value& obj, const char* key, double value, rapidjson::MemoryPoolAllocator<>* allocator);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const char* value);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, const std::string& value);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, bool value);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, int64_t value);
IMQS_PAL_API void Set(rapidjson::Document& doc, const char* key, double value);

IMQS_PAL_API bool GetBool(const rapidjson::Value& v, const char* key, bool defaultValue = false);
IMQS_PAL_API int  GetInt(const rapidjson::Value& v, const char* key, int defaultValue = 0);
IMQS_PAL_API int64_t GetInt64(const rapidjson::Value& v, const char* key, int64_t defaultValue = 0);
IMQS_PAL_API std::string GetString(const rapidjson::Value& v, const char* key, std::string defaultValue = "");

IMQS_PAL_API std::vector<std::string> Keys(const rapidjson::Value& v);

IMQS_PAL_API bool InOut(bool out, rapidjson::Document& doc, rapidjson::Value& v, const char* key, int64_t& value);
IMQS_PAL_API bool InOut(bool out, rapidjson::Document& doc, rapidjson::Value& v, const char* key, std::string& value);
} // namespace rj
#endif

// nlohmann:json helpers
namespace nj {
IMQS_PAL_API Error ParseString(const std::string& raw, nlohmann::json& doc);
IMQS_PAL_API Error ParseFile(const std::string& filename, nlohmann::json& doc);

IMQS_PAL_API bool GetBool(const nlohmann::json& v, const char* key, bool defaultValue = false);
IMQS_PAL_API std::string GetString(const nlohmann::json& v, const char* key, std::string defaultValue = "");
IMQS_PAL_API int         GetInt(const nlohmann::json& v, const char* key, int defaultValue = 0);
IMQS_PAL_API int64_t GetInt64(const nlohmann::json& v, const char* key, int64_t defaultValue = 0);
IMQS_PAL_API double  GetDouble(const nlohmann::json& v, const char* key, double defaultValue = 0);
IMQS_PAL_API std::vector<std::string> GetStringList(const nlohmann::json& v, const char* key);
IMQS_PAL_API bool                     IsArray(const nlohmann::json& v, const char* key);
IMQS_PAL_API bool                     IsObject(const nlohmann::json& v, const char* key);
IMQS_PAL_API bool                     IsString(const nlohmann::json& v, const char* key);
IMQS_PAL_API bool                     IsNumber(const nlohmann::json& v, const char* key);
IMQS_PAL_API bool                     IsBool(const nlohmann::json& v, const char* key);
} // namespace nj
} // namespace imqs
