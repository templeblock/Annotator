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
		// get_frame? width=1280 height=720 frame=1234 -> returns frame as raw 24-bit RGB.
		// frame is optional. If omitted, returns the next frame
	case "get_frame"_crc32: {
		auto ses = sessions.GetSession(GetSessionID(r));
		if (!ses) {
			w.SetStatusAndBody(phttp::Status410_Gone, "Session expired or invalid");
			return;
		}
		w.SetHeader("Content-Type", "application/octet-stream");
		auto width  = r.QueryInt("width");
		auto height = r.QueryInt("height");
		//ses->Video.DecodeFrameRGBA(width, height, )
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
		// We can't place HandleHttp inside a lambda here, because of a bug in clang-format 6.0
		HandleHttp(sessions, w, r);
	});

	if (!ok)
		tsf::print("Failed to start server\n");

	phttp::Shutdown();

	return 0;
}
