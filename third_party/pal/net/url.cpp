#include "pch.h"
#include "url.h"
#include "../alloc.h"
#include "../modp/modp_burl.h"
#include "../modp/modp_qsiter.h"

namespace imqs {
namespace url {

static bool IsDigit(int c) {
	return c >= '0' && c <= '9';
}

static void EncodeAndAppendTo(std::string& dst, const char* src, size_t srcLen, char*& buf, size_t& bufSize, bool& isStaticBuf) {
	size_t size = modp_burl_min_encode_strlen(src, srcLen);
	if (size > bufSize) {
		while (size > bufSize)
			bufSize *= 2;
		if (!isStaticBuf)
			free(buf);
		buf = (char*) imqs_malloc_or_die(bufSize);
	}

	size = modp_burl_min_encode(buf, src, srcLen);
	dst.append(buf, size);
}

static void DecodeAndAppendTo(std::string& dst, const char* src, size_t srcLen, char*& buf, size_t& bufSize, bool& isStaticBuf) {
	size_t size = modp_burl_decode_len(srcLen);
	if (size > bufSize) {
		while (size > bufSize)
			bufSize *= 2;
		if (!isStaticBuf)
			free(buf);
		buf = (char*) imqs_malloc_or_die(bufSize);
	}

	size = modp_burl_decode(buf, src, srcLen);
	dst.append(buf, size);
}

URL::URL() {
}

URL::URL(const char* proto, const char* host, int port, const char* path, const std::vector<std::pair<std::string, std::string>>& query) {
	Proto = proto;
	Host  = host;
	Port  = port;
	Path  = path;
	Query = query;
}

URL::~URL() {
}

std::string URL::Encode() const {
	char   static_buf[200];
	char*  buf           = static_buf;
	size_t buf_size      = sizeof(static_buf);
	bool   buf_is_static = true;

	std::string s;
	if (Port != 0)
		s = tsf::fmt("%s://%s:%d", Proto, Host, Port);
	else
		s = tsf::fmt("%s://%s", Proto, Host);

	// add initial slash, if not present in Path
	if (Path.length() > 0 && Path[0] != '/')
		s += '/';

	EncodeAndAppendTo(s, Path.c_str(), Path.length(), buf, buf_size, buf_is_static);

	// add ? separator before query, if not present in path
	if (Path.length() > 0 && Path[Path.length() - 1] != '?' && Query.size() != 0)
		s += '?';

	for (size_t i = 0; i < Query.size(); i++) {
		const auto& p = Query[i];
		EncodeAndAppendTo(s, p.first.c_str(), p.first.length(), buf, buf_size, buf_is_static);
		s += '=';
		EncodeAndAppendTo(s, p.second.c_str(), p.second.length(), buf, buf_size, buf_is_static);
		if (i != Query.size() - 1)
			s += '&';
	}

	if (buf != static_buf)
		free(buf);

	return s;
}

Error URL::Decode(const char* url) {
	char   static_buf[200];
	char*  buf           = static_buf;
	size_t buf_size      = sizeof(static_buf);
	bool   buf_is_static = true;

	// proto
	size_t i = 0;
	for (; url[i]; i++) {
		if (url[i] == ':' && url[i + 1] == '/' && url[i + 2] == '/') {
			Proto.assign(url, i);
			i += 3;
			break;
		}
	}
	if (url[i] == 0)
		return Error("No protocol in URL");

	// hostname & port
	Port                   = 0;
	size_t startOfHostname = i;
	size_t endOfHostname   = -1;
	for (; url[i]; i++) {
		if (url[i - 1] != ':' && url[i] == ':' && IsDigit(url[i + 1])) {
			endOfHostname = i;
			int    port   = 0;
			size_t j      = i + 1;
			while (IsDigit(url[j])) {
				port = port * 10 + url[j] - '0';
				j++;
			}
			Port = port;
			i    = j;
			break;
		} else if (url[i] == '/') {
			endOfHostname = i;
			break;
		}
	}
	if (endOfHostname == -1)
		return Error("No hostname in URL");

	Host.assign(url + startOfHostname, endOfHostname - startOfHostname);

	// path
	size_t startPath = i;
	while (url[i] && url[i] != '?')
		i++;
	DecodeAndAppendTo(Path, url + startPath, i - startPath, buf, buf_size, buf_is_static);

	if (url[i] == '?') {
		i++;
		qsiter_t it;
		qsiter_reset(&it, url + i, strlen(url + i));
		while (qsiter_next(&it)) {
			Query.push_back({});
			DecodeAndAppendTo(Query[Query.size() - 1].first, it.key, it.keylen, buf, buf_size, buf_is_static);
			DecodeAndAppendTo(Query[Query.size() - 1].second, it.val, it.vallen, buf, buf_size, buf_is_static);
		}
	}

	return Error();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IMQS_PAL_API std::string Encode(const std::string& s) {
	return modp::url_encode(s);
}

IMQS_PAL_API std::string Encode(const ohash::map<std::string, std::string>& queryParams) {
	std::string enc;
	for (const auto& pair : queryParams) {
		enc += modp::url_encode(pair.first);
		enc += '=';
		enc += modp::url_encode(pair.second);
		enc += '&';
	}
	if (enc.length() != 0)
		enc.erase(enc.end() - 1, enc.end());
	return enc;
}
}
}
