#include "pch.h"
#include "SvgIcons.h"

namespace imqs {
namespace anno {
namespace svg {

const char* media_pause = R"-(<svg width="8" height="8" viewBox="0 0 8 8">
  <path d="M0 0v6h2v-6h-2zm4 0v6h2v-6h-2z" transform="translate(1 1)" />
</svg>)-";

const char* media_play = R"-(<svg width="8" height="8" viewBox="0 0 8 8">
  <path d="M0 0v6l6-3-6-3z" transform="translate(1 1)" />
</svg>)-";

const char* media_record = R"-(<svg width="8" height="8" viewBox="0 0 8 8">
  <path d="M3 0c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z" transform="translate(1 1)" />
</svg>)-";

const char* media_skip_backward = R"-(<svg width="8" height="8" viewBox="0 0 8 8">
  <path d="M4 0l-4 3 4 3v-6zm0 3l4 3v-6l-4 3z" transform="translate(0 1)" />
</svg>)-";

const char* media_skip_forward = R"-(<svg width="8" height="8" viewBox="0 0 8 8">
  <path d="M0 0v6l4-3-4-3zm4 3v3l4-3-4-3v3z" transform="translate(0 1)" />
</svg>)-";

const char* media_step_backward = R"-(<svg width="8" height="8" viewBox="0 0 8 8">
  <path d="M0 0v6h2v-6h-2zm2 3l5 3v-6l-5 3z" transform="translate(0 1)" />
</svg>)-";

const char* media_step_forward = R"-(<svg width="8" height="8" viewBox="0 0 8 8">
  <path d="M0 0v6l5-3-5-3zm5 3v3h2v-6h-2v3z" transform="translate(0 1)" />
</svg>)-";

const char* media_stop = R"-(<svg width="8" height="8" viewBox="0 0 8 8">
  <path d="M0 0v6h6v-6h-6z" transform="translate(1 1)" />
</svg>)-";

const char* All[] = {
	"media-pause", media_pause,
	"media-play", media_play,
	"media-record", media_record,
	"media-skip-backward", media_skip_backward,
	"media-skip-forward", media_skip_forward,
	"media-step-backward", media_step_backward,
	"media-step-forward", media_step_forward,
	"media-stop", media_stop,
	nullptr, nullptr
};

void LoadAll(xo::Doc* doc) {
	for (size_t i = 0; All[i]; i += 2)
		doc->SetSvg(All[i], All[i + 1]);
}

}
}
}
