#include "pch.h"
#include "Session.h"

using namespace std;
using namespace imqs;
using namespace imqs::frameserver;

const char* CookieName = "frameserver_session";

// Use cookie or "id" query parameter
static string GetSessionID(phttp::Request& r) {
	for (const auto& h : r.Headers) {
		if (h.first == "Cookie") {
			auto pos = h.second.find(CookieName);
			if (pos != -1) {
				pos += strlen(CookieName) + 1;
				auto end = h.second.find(';', pos);
				if (end == -1)
					end = h.second.size();
				return h.second.substr(pos, end - pos);
			}
		}
	}
	return r.QueryVal("frameserver_session");
}

struct ImqsLz4ImageHeader {
	uint16_t Magic     = 0xf789;
	uint16_t Format    = 0; // 0 = RGBA
	uint16_t Width     = 0;
	uint16_t Height    = 0;
	uint32_t Reserved1 = 0;
	uint32_t Reserved2 = 0;
};

static void EncodeFrame(string codec, const void* buf, int width, int height, int stride, int quality, phttp::Response& w) {
	gfx::ImageIO     io;
	Error            err;
	void*            encbuf  = nullptr;
	size_t           encsize = 0;
	gfx::ImageFormat iformat = gfx::ImageFormat::Null;
	if (codec == "png") {
		iformat = gfx::ImageFormat::Png;
		w.SetHeader("Content-Type", "image/png");
		err = io.SavePng(false, width, height, stride, buf, quality != -1 ? quality : 5, encbuf, encsize);
	} else if (codec == "jpeg") {
		iformat = gfx::ImageFormat::Jpeg;
		w.SetHeader("Content-Type", "image/jpeg");
		err = io.SaveJpeg(width, height, stride, buf, quality != -1 ? quality : 95, encbuf, encsize);
	} else if (codec == "lz4") {
		w.SetHeader("Content-Type", "image/x-imqs-lz4");
		IMQS_ASSERT(stride == width * 4);
		size_t enccap = LZ4F_compressBound(height * stride, nullptr);
		encbuf        = imqs_malloc_or_die(enccap);
		encsize       = LZ4F_compressFrame(encbuf, enccap, buf, height * stride, nullptr);
	} else {
		w.SetStatusAndBody(400, "Invalid format. Valid formats are png, jpeg, lz4");
		return;
	}

	if (!err.OK()) {
		w.SetStatusAndBody(500, err.Message());
		return;
	}

	w.Status = 200;
	w.Body.assign((const char*) encbuf, encsize);
	if (iformat != gfx::ImageFormat::Null)
		io.FreeEncodedBuffer(iformat, encbuf);
	else
		free(encbuf);
}

static void HandleHttp(SessionStore& sessions, phttp::Response& w, phttp::Request& r) {
	auto parts = strings::Split(r.Path, '/');
	if (parts.size() == 0) {
		w.SetStatusAndBody(200, "URLS: ping, create_session, info, get_frame");
		return;
	}

	switch (imqs::hash::crc32(parts[1])) {
	case "ping"_crc32:
		w.Body = "Hello from frameserver";
		break;
		// create_session/:videofile
	case "create_session"_crc32: {
		const size_t vstart = 2;
		if (parts.size() <= vstart) {
			w.SetStatusAndBody(phttp::Status400_Bad_Request, "No videofile specified");
			return;
		}
		string videofile = path::SafeJoin(parts.size() - vstart, &parts[vstart]);
		string id;
		auto   err = sessions.CreateSession(videofile, id);
		if (!err.OK()) {
			w.SetStatusAndBody(phttp::Status400_Bad_Request, err.Message());
			return;
		}
		w.SetHeader("Set-Cookie", tsf::fmt("%v=%v; Path=/", CookieName, id));
		w.SetStatusAndBody(200, id);
		break;
	}
		// info -> returns JSON with video metadata such as width, height, duration
	case "info"_crc32: {
		auto ses = sessions.GetSession(GetSessionID(r));
		if (!ses) {
			w.SetStatusAndBody(phttp::Status410_Gone, "Session expired or invalid");
			return;
		}
		nlohmann::json j;
		j["Width"]           = ses->Video.Width();
		j["Height"]          = ses->Video.Height();
		j["Seconds"]         = ses->Video.GetVideoStreamInfo().DurationSeconds();
		j["FramesPerSecond"] = ses->Video.GetVideoStreamInfo().FrameRateSeconds();
		w.SetHeader("Content-Type", "application/json");
		w.SetStatusAndBody(200, j.dump());
		break;
	}
		// frame? width=1280 height=720 frame=1234 format=png -> returns frame
		// frame is optional. If omitted, returns the next frame
		// quality is optional. For jpeg, quality is between 0 and 100. For png, quality is between 0 and 9 (zlib)
	case "frame"_crc32: {
		auto ses = sessions.GetSession(GetSessionID(r));
		if (!ses) {
			w.SetStatusAndBody(phttp::Status410_Gone, "Session expired or invalid");
			return;
		}
		auto width   = r.QueryInt("width");
		auto height  = r.QueryInt("height");
		auto format  = r.QueryVal("format");
		auto quality = r.QueryInt("quality");
		if (r.QueryVal("quality") == "")
			quality = -1;
		int   stride = width * 4;
		void* buf    = imqs_malloc_or_die(height * stride);
		ses->Video.DecodeFrameRGBA(width, height, buf, stride);
		EncodeFrame(format, buf, width, height, stride, quality, w);
		free(buf);
		break;
	}
	default:
		w.Status = 404;
		break;
	}
}

void ShowHelp() {
	tsf::print("FrameServer video_dir\n");
	tsf::print("  video_dir The directory containing the video files to be served\n");
}

int main(int argc, char** argv) {
	vector<string> args;
	for (int i = 1; i < argc; i++)
		args.push_back(argv[i]);
	if (args.size() == 0) {
		ShowHelp();
		return 1;
	}

	phttp::Initialize();
	anno::VideoFile::Initialize();

	uberlog::Logger log;
	logging::CreateLogger("FrameServer", log);
#ifdef _DEBUG
	log.TeeStdOut = true;
#endif

	SessionStore sessions(args[0], &log);

	phttp::Server server;
	bool          ok = server.ListenAndRun("localhost", 8080, [&](phttp::Response& w, phttp::Request& r) {
        // We can't place HandleHttp inside a lambda here, because of a bug in clang-format 6.0.
        // It's anyway cleaner like this.
        HandleHttp(sessions, w, r);
    });

	if (!ok)
		tsf::print("Failed to start server\n");

	phttp::Shutdown();

	return 0;
}
