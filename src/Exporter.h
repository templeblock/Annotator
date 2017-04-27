#pragma once

#include "LabelIO.h"

namespace imqs {
namespace anno {

Error ExportLabeledImagePatches_Frame(std::string dir, std::string frameName, const ImageLabels& labels, const ohash::map<std::string, int>& labelToIndex, const xo::Image& frameImg);
Error ExportLabeledImagePatches_Video(std::string videoFilename, const VideoLabels& labels);

}
} // namespace imqs