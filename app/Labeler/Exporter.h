#pragma once

#include "LabelIO.h"

namespace imqs {
namespace anno {

enum class ExportTypes {
	Cifar10, // One .bin file for every frame. Format inside .bin is cifar10 format, which is one byte for label, followed by raw RGB planes.
	Png,     // One directory per class, with all images for that class inside the directory.
};

typedef std::function<bool(size_t pos, size_t total)> ProgressCallback;

Error ExportLabeledImagePatches_Frame(ExportTypes type, std::string dir, int64_t frameTime, const ImageLabels& labels, const ohash::map<std::string, int>& labelToIndex, const xo::Image& frameImg);
Error ExportLabeledImagePatches_Video(ExportTypes type, std::string videoFilename, const VideoLabels& labels, ProgressCallback prog);

} // namespace anno
} // namespace imqs