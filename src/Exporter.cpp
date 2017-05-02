#include "pch.h"
#include "Exporter.h"
#include "VideoDecode.h"

namespace imqs {
namespace anno {

static void WriteRaw(const xo::Texture& tex, uint8_t label, uint8_t* buf) {
	buf[0] = label;
	buf++;
	bool planar = true;
	if (planar) {
		// RRR GGG BBB
		for (int chan = 0; chan < 3; chan++) {
			for (uint32_t y = 0; y < tex.Height; y++) {
				auto src = (uint8_t*) tex.DataAtLine(y);
				src += chan;
				auto dst    = buf + y * tex.Width;
				int  srcBPP = (int) tex.BytesPerPixel();
				for (uint32_t x = 0; x < tex.Width; x++) {
					*dst = *src;
					dst++;
					src += srcBPP;
				}
			}
			buf += tex.Width * tex.Height;
		}
	} else {
		// RGB RGB RGB
		for (uint32_t y = 0; y < tex.Height; y++) {
			auto src    = (uint8_t*) tex.DataAtLine(y);
			auto dst    = buf + y * tex.Width * 3;
			int  srcBPP = (int) tex.BytesPerPixel();
			for (uint32_t x = 0; x < tex.Width; x++) {
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
				dst += 3;
				src += srcBPP;
			}
		}
	}
}

Error ExportLabeledImagePatches_Frame(std::string dir, std::string frameName, const ImageLabels& labels, const ohash::map<std::string, int>& labelToIndex, const xo::Image& frameImg) {
	if (labels.Labels.size() == 0)
		return Error();

	os::File f;
	auto     err = f.Create(tsf::fmt("%v/%v.bin", dir, frameName));
	if (!err.OK())
		return err;

	int      dim     = labels.Labels[0].Rect.Width(); // assume all rectangles are the same size, and that width == height
	size_t   bufSize = 1 + dim * dim * 3;
	uint8_t* buf     = (uint8_t*) imqs_malloc_or_die(bufSize);

	for (const auto& patch : labels.Labels) {
		IMQS_ASSERT(patch.Rect.Width() == dim && patch.Rect.Height() == dim);
		auto    patchTex = frameImg.Window(patch.Rect.X1, patch.Rect.Y1, patch.Rect.Width(), patch.Rect.Height());
		uint8_t klass    = labelToIndex.get(patch.Class);
		WriteRaw(patchTex, klass, buf);
		err = f.Write(buf, bufSize);
		if (!err.OK())
			return err;
	}

	free(buf);

	return Error();
}

Error ExportLabeledImagePatches_Video(std::string videoFilename, const VideoLabels& labels) {
	auto dir = ImagePatchDir(videoFilename);
	auto err = os::MkDirAll(dir);
	if (!err.OK())
		return err;

	VideoFile video;
	err = video.OpenFile(videoFilename);
	if (!err.OK())
		return err;

	xo::Image img;
	if (!img.Alloc(xo::TexFormatRGBA8, video.Width(), video.Height()))
		return Error("Out of memory allocating image buffer");

	// Establish a mapping from label (a string) to an integer class.
	// The ML libraries just want an integer for the class, not a string.
	ohash::map<std::string, int> labelToIndex;
	for (const auto& frame : labels.Frames) {
		for (const auto& lab : frame.Labels)
			labelToIndex.insert(lab.Class, labelToIndex.size());
	}

	int64_t lastFrameTime = 0;
	int64_t micro         = 1000000;

	for (const auto& frame : labels.Frames) {
		// Only seek if frame is more than 5 seconds into the future
		if (frame.Time - lastFrameTime > 5 * micro) {
			int64_t buffer = 3 * micro; // seek 3 seconds behind frame target
			err            = video.SeekToMicrosecond(frame.Time - buffer);
			if (!err.OK())
				return err;
		}

		while (true) {
			err = video.DecodeFrameRGBA(img.Width, img.Height, img.Data, img.Stride);
			if (!err.OK())
				return err;
			int64_t pts   = video.LastFrameTimeMicrosecond();
			lastFrameTime = pts;
			if (pts == frame.Time) {
				// found our frame
				ExportLabeledImagePatches_Frame(dir, tsf::fmt("%v", frame.Time), frame, labelToIndex, img);
				break;
			} else if (pts > frame.Time) {
				tsf::print("Fail to find frame %v\n", frame.Time);
				break;
			}
		}
	}

	return Error();
}

} // namespace anno
} // namespace imqs