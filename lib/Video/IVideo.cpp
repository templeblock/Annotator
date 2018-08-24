#include "pch.h"
#include "IVideo.h"

namespace imqs {
namespace video {

StaticError ErrNeedMoreData("Codec needs more data");

Error IVideo::TranslateAvErr(int ret, const char* whileBusyWith) {
	char errBuf[AV_ERROR_MAX_STRING_SIZE + 1];

	switch (ret) {
	case AVERROR_EOF: return ErrEOF;
	case AVERROR(EAGAIN): return ErrNeedMoreData;
	default:
		av_strerror(ret, errBuf, sizeof(errBuf));
		if (whileBusyWith)
			return Error::Fmt("%v: %v", whileBusyWith, errBuf);
		else
			return Error::Fmt("AVERROR %v", errBuf);
	}
}

} // namespace video
} // namespace imqs