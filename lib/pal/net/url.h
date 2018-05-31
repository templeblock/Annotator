#pragma once

namespace imqs {
namespace url {

// A URL, broken down into it's individual unescaped components
class IMQS_PAL_API URL {
public:
	std::string                                      Proto;    // "http", "https", etc
	std::string                                      Host;     // "example.com"
	std::string                                      Path;     // "/path/with/leading/slash"
	std::vector<std::pair<std::string, std::string>> Query;    // {{"x", "y"}}
	int                                              Port = 0; // 0 = unspecified

	URL();
	URL(const char* proto, const char* host, int port, const char* path, const std::vector<std::pair<std::string, std::string>>& query);
	~URL();

	std::string Encode() const;
	Error       Decode(const char* url);
};

IMQS_PAL_API std::string Encode(const std::string& s);
IMQS_PAL_API std::string Encode(const ohash::map<std::string, std::string>& queryParams);
IMQS_PAL_API std::string Decode(const std::string& s);
} // namespace url
} // namespace imqs
