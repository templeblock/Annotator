#pragma once

namespace imqs {
namespace video {

enum SeekFlags {
	SeekFlagNone = 0,               // seek to nearest keyframe
	SeekFlagAny  = AVSEEK_FLAG_ANY, // seek to any frame, even non-keyframes

	// Note: ffmpeg has these:
	// AVSEEK_FLAG_BACKWARD 1 ///< seek backward
	// AVSEEK_FLAG_BYTE     2 ///< seeking based on position in bytes
	// AVSEEK_FLAG_ANY      4 ///< seek to any frame, even non-keyframes
	// AVSEEK_FLAG_FRAME    8 ///< seeking based on frame number
	// We can add support for them if/when we figure out their exact behaviour
};

extern StaticError ErrNeedMoreData; // Codec needs more data before it can deliver a frame/audio

class IMQS_VIDEO_API IVideo {
public:
	virtual ~IVideo() {}
	virtual Error OpenFile(std::string filename)                                                               = 0;
	virtual void  Info(int& width, int& height, int64_t& durationMicroseconds)                                 = 0;
	virtual Error DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds = nullptr) = 0;
	virtual Error SeekToMicrosecond(int64_t microsecond, unsigned flags = SeekFlagNone)                        = 0;

	static Error TranslateAvErr(int ret, const char* whileBusyWith = nullptr);
};

} // namespace video
} // namespace imqs