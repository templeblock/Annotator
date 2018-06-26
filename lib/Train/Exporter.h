#pragma once

#include "LabelIO.h"

namespace imqs {
namespace train {

enum class ExportTypes {
	Png,          // One directory per class, with all images for that class inside the directory.
	Jpeg,         // Same as png, but jpeg at 95% quality
	Segmentation, // Segmentation. JPEG for video frame, and PNG for segmentation class
};

typedef std::function<bool(size_t pos, size_t total)> ProgressCallback;

IMQS_TRAIN_API Error ExportLabeledImagePatches_Video(ExportTypes type, std::string videoFilename, const LabelTaxonomy& taxonomy, const VideoLabels& labels, ProgressCallback prog);
IMQS_TRAIN_API Error ExportLabeledBatch(bool channelsFirst, bool compress, const std::vector<std::pair<int, int>>& batch, video::VideoFile& video, const VideoLabels& labels, std::string& encoded);

} // namespace train
} // namespace imqs