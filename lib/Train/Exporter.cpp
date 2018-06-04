#include "pch.h"
#include "Exporter.h"

using namespace std;

namespace imqs {
namespace train {

static void WriteRawCifar10(const gfx::Image& img, uint8_t label, uint8_t* buf) {
	// 1st byte is label
	buf[0] = label;
	buf++;

	// after label follows RGB, either planar, or packed

	bool planar = true;
	if (planar) {
		// RRR GGG BBB
		for (int chan = 0; chan < 3; chan++) {
			for (uint32_t y = 0; y < (uint32_t) img.Height; y++) {
				auto src = (uint8_t*) img.Line(y);
				src += chan;
				auto dst    = buf + y * img.Width;
				int  srcBPP = (int) img.BytesPerPixel();
				for (uint32_t x = 0; x < (uint32_t) img.Width; x++) {
					*dst = *src;
					dst++;
					src += srcBPP;
				}
			}
			buf += img.Width * img.Height;
		}
	} else {
		// RGB RGB RGB
		for (uint32_t y = 0; y < (uint32_t) img.Height; y++) {
			auto src    = (uint8_t*) img.Line(y);
			auto dst    = buf + y * img.Width * 3;
			int  srcBPP = (int) img.BytesPerPixel();
			for (uint32_t x = 0; x < (uint32_t) img.Width; x++) {
				dst[0] = src[0];
				dst[1] = src[1];
				dst[2] = src[2];
				dst += 3;
				src += srcBPP;
			}
		}
	}
}

Error SaveImageFile(gfx::ImageIO& imgIO, const gfx::Image& img, gfx::ImageType filetype, std::string filename) {
	void*  encbuf  = nullptr;
	size_t encsize = 0;
	auto   err     = imgIO.Save(img.Width, img.Height, img.Stride, img.Data, filetype, false, 95, 1, encbuf, encsize);
	if (!err.OK())
		return err;
	err = os::WriteWholeFile(filename, encbuf, encsize);
	imgIO.FreeEncodedBuffer(filetype, encbuf);
	return err;
}

Error ExportLabeledImagePatches_Frame(ExportTypes type, std::string dir, int64_t frameTime, const ImageLabels& labels, const ohash::map<std::string, int>& labelToIndex, const gfx::Image& frameImg) {
	if (labels.Labels.size() == 0)
		return Error();

	os::File f;
	if (type == ExportTypes::Cifar10) {
		auto err = f.Create(tsf::fmt("%v/%v.bin", dir, frameTime));
		if (!err.OK())
			return err;
	}

	int      dim     = labels.Labels[0].Rect.Width(); // assume all rectangles are the same size, and that width == height
	size_t   bufSize = 1 + dim * dim * 3;
	uint8_t* buf     = (uint8_t*) imqs_malloc_or_die(bufSize);

	gfx::ImageIO imgIO;

	for (const auto& patch : labels.Labels) {
		IMQS_ASSERT(patch.Rect.Width() == dim && patch.Rect.Height() == dim);
		auto patchTex = frameImg.Window(patch.Rect.X1, patch.Rect.Y1, patch.Rect.Width(), patch.Rect.Height());
		if (type == ExportTypes::Cifar10) {
			uint8_t klass = labelToIndex.get(patch.Class);
			WriteRawCifar10(patchTex, klass, buf);
			auto err = f.Write(buf, bufSize);
			if (!err.OK())
				return err;
		} else if (type == ExportTypes::Png || type == ExportTypes::Jpeg) {
			gfx::ImageType filetype;
			string         ext;
			if (type == ExportTypes::Png) {
				filetype = gfx::ImageType::Png;
				ext      = ".png";
			} else {
				filetype = gfx::ImageType::Jpeg;
				ext      = ".jpeg";
			}
			auto classDir = dir + "/" + strings::Replace(patch.Class, " ", "_");
			auto err      = os::MkDirAll(classDir);
			if (!err.OK())
				return err;
			auto filename = classDir + "/" + tsf::fmt("%09d-%04d-%04d-%04d-%04d.%v", frameTime, patch.Rect.X1, patch.Rect.Y1, patch.Rect.Width(), patch.Rect.Height(), ext);
			err           = SaveImageFile(imgIO, patchTex, filetype, filename);
			if (!err.OK())
				return err;
		}
	}

	free(buf);

	return Error();
}

Error ExportLabeledImagePatches_Video(ExportTypes type, std::string videoFilename, const VideoLabels& labels, ProgressCallback prog) {
	auto dir = ImagePatchDir(videoFilename);
	auto err = os::MkDirAll(dir);
	if (!err.OK())
		return err;

	video::VideoFile video;
	err = video.OpenFile(videoFilename);
	if (!err.OK())
		return err;

	gfx::Image img(gfx::ImageFormat::RGBA, video.Width(), video.Height());

	// Establish a mapping from label (a string) to an integer class.
	// The ML libraries just want an integer for the class, not a string.
	ohash::map<std::string, int> labelToIndex = labels.ClassToIndex();

	int64_t lastFrameTime = 0;
	int64_t micro         = 1000000;
	bool    abort         = false;

	for (size_t i = 0; i < labels.Frames.size() && !abort; i++) {
		const auto& frame = labels.Frames[i];
		// Only seek if frame is more than 5 seconds into the future. Haven't measured optimal metric to use here.
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
				ExportLabeledImagePatches_Frame(type, dir, frame.Time, frame, labelToIndex, img);
				if (prog != nullptr) {
					if (!prog(i, labels.Frames.size())) {
						abort = true;
						break;
					}
				}
				break;
			} else if (pts > frame.Time) {
				tsf::print("Fail to find frame %v\n", frame.Time);
				break;
			}
		}
	}

	return Error();
}

static void ConvertRGBAtoRGB(bool channelsFirst, int srcStride, const uint8_t* src, int dstStride, uint8_t* dst, int width, int height) {
	// convert RGBA to RGB
	if (channelsFirst) {
		for (int y = 0; y < height; y++) {
			auto* srcLine  = src + srcStride * y;
			auto* dstLineR = dst + dstStride * y;
			auto* dstLineG = dst + dstStride * y + height * dstStride;
			auto* dstLineB = dst + dstStride * y + height * dstStride * 2;
			for (int x = 0; x < width; x++) {
				*dstLineR = srcLine[0];
				*dstLineG = srcLine[1];
				*dstLineB = srcLine[2];
				dstLineR++;
				dstLineG++;
				dstLineB++;
				srcLine += 4;
			}
		}
	} else {
		for (int y = 0; y < height; y++) {
			auto* srcLine = src + srcStride * y;
			auto* dstLine = dst + dstStride * y;
			for (int x = 0; x < width; x++) {
				dstLine[0] = srcLine[0];
				dstLine[1] = srcLine[1];
				dstLine[2] = srcLine[2];
				dstLine += 3;
				srcLine += 4;
			}
		}
	}
}

#pragma pack(push)
#pragma pack(4)
struct BatchHeader {
	int32_t Version     = 0;
	int32_t BatchSize   = 0;
	int32_t ImgWidth    = 0;
	int32_t ImgHeight   = 0;
	int32_t ImgChannels = 0;
	int32_t RawSize     = 0; // If zero, then uncompressed
};
#pragma pack(pop)

// Export a labeled batch into a data format that can be easily turned into a numpy array, or pytorch tensor.
// 'batch' contains a list of pairs, where the first part of the pair is the frame index (not the frame time),
// and the second part is the label index within that frame.
IMQS_TRAIN_API Error ExportLabeledBatch(bool channelsFirst, bool compress, const std::vector<std::pair<int, int>>& batch, video::VideoFile& video, const VideoLabels& labels, std::string& encoded) {
	int    sampleWidth  = 0;
	int    sampleHeight = 0;
	size_t dstImageSize = 0;
	int    dstStride    = 0;
	int    lastFrame    = -1;

	gfx::Image frameBuf(gfx::ImageFormat::RGBA, video.Width(), video.Height());

	// store labels separately, and add them in at the end
	string dstLabels;
	dstLabels.resize(sizeof(int) * batch.size());
	int* dstLabelsPtr = (int*) dstLabels.data();

	string dstImage;
	auto   classToIndex = labels.ClassToIndex();

	for (size_t i = 0; i < batch.size(); i++) {
		const auto& p = batch[i];
		if ((unsigned) p.first >= labels.Frames.size())
			return Error::Fmt("Invalid frame number %v (number of frames is %v)", p.first, labels.Frames.size());

		const auto& frame = labels.Frames[p.first];

		if ((unsigned) p.second >= frame.Labels.size())
			return Error::Fmt("Invalid label number %v, in frame %v (number of labels is %v)", p.second, p.first, frame.Labels.size());

		const auto& label = frame.Labels[p.second];
		if (sampleWidth == 0) {
			sampleWidth  = label.Rect.Width();
			sampleHeight = label.Rect.Height();
			// now that we know the sample size, we can allocate our entire buffer up front
			dstStride    = channelsFirst ? sampleWidth : sampleWidth * 3;
			dstImageSize = sampleWidth * sampleHeight * 3;
			dstImage.resize(batch.size() * dstImageSize);
		} else if (sampleWidth != label.Rect.Width() || sampleHeight != label.Rect.Height()) {
			return Error::Fmt("Label %v.%v has different dimensions (%v, %v) to the other labels (%v, %v)", label.Rect.Width(), label.Rect.Height(), sampleWidth, sampleHeight);
		}

		if (p.first != lastFrame) {
			auto err = video.SeekToMicrosecond(frame.Time);
			if (!err.OK())
				return Error::Fmt("Error seeking to frame %v: %v", p.first, err.Message());
			err = video.DecodeFrameRGBA(video.Width(), video.Height(), frameBuf.Data, frameBuf.Stride);
			if (!err.OK())
				return Error::Fmt("Error decoding frame %v: %v", p.first, err.Message());
			lastFrame = p.first;
		}

		auto     wnd    = frameBuf.Window(label.Rect.X1, label.Rect.Y1, label.Rect.Width(), label.Rect.Height());
		uint8_t* dstBuf = (uint8_t*) dstImage.data() + i * dstImageSize;
		ConvertRGBAtoRGB(channelsFirst, wnd.Stride, wnd.Data, dstStride, dstBuf, wnd.Width, wnd.Height);

		dstLabelsPtr[i] = classToIndex.get(label.Class);
	}

	// append labels to images, so it's one contiguous block of bytes
	dstImage += dstLabels;

	BatchHeader head;
	head.Version     = 1;
	head.ImgWidth    = sampleWidth;
	head.ImgHeight   = sampleHeight;
	head.ImgChannels = 3;
	head.BatchSize   = (int) batch.size();

	// f = final buffer, which can be either compressed or uncompressed
	size_t fsize = dstImage.size();
	void*  fbuf  = (void*) dstImage.data();

	if (compress) {
		head.RawSize = (int) dstImage.size();
		fsize        = LZ4F_compressFrameBound(dstImage.size(), nullptr);
		fbuf         = imqs_malloc_or_die(fsize);
		fsize        = LZ4F_compressFrame(fbuf, fsize, dstImage.data(), dstImage.size(), nullptr);
		IMQS_ASSERT(!LZ4F_isError(fsize));
	} else {
		head.RawSize = 0;
	}

	encoded.resize(sizeof(head) + fsize);
	uint8_t* enc = (uint8_t*) encoded.c_str();

	memcpy(enc, &head, sizeof(head));
	memcpy(enc + sizeof(head), fbuf, fsize);

	if (compress)
		free(fbuf);

	return Error();
}

} // namespace train
} // namespace imqs