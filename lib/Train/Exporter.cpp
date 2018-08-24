#include "pch.h"
#include "Exporter.h"

using namespace std;

namespace imqs {
namespace train {

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

static Error MakeClassDir(string baseDir, string className) {
	auto classDir = baseDir + "/" + strings::Replace(className, " ", "_");
	return os::MkDirAll(classDir);
}

Error ExportLabeledImagePatches_Frame_Rect(ExportTypes type, std::string dir, int64_t frameTime, const ImageLabels& labels, const gfx::Image& frameImg) {
	if (labels.Labels.size() == 0)
		return Error();

	int dim = labels.Labels[0].Rect.Width(); // assume all rectangles are the same size, and that width == height

	atomic<bool> haveErr;
	haveErr = false;
	Error firstErr;
#pragma omp parallel for
	for (int i = 0; i < (int) labels.Labels.size(); i++) {
		if (haveErr)
			continue;
		const auto& patch = labels.Labels[i];
		IMQS_ASSERT(patch.Rect.Width() == dim && patch.Rect.Height() == dim);
		auto           patchTex = frameImg.Window(patch.Rect.X1, patch.Rect.Y1, patch.Rect.Width(), patch.Rect.Height());
		gfx::ImageIO   imgIO;
		gfx::ImageType filetype;
		string         ext;
		if (type == ExportTypes::Png) {
			filetype = gfx::ImageType::Png;
			ext      = "png";
		} else {
			filetype = gfx::ImageType::Jpeg;
			ext      = "jpeg";
		}
		auto filename = dir + "/" + tsf::fmt("%09d-%04d-%04d-%04d-%04d.%v", frameTime, patch.Rect.X1, patch.Rect.Y1, patch.Rect.Width(), patch.Rect.Height(), ext);
		if (os::IsFile(filename)) {
			// since our images are just extracts of the video, and labels are stored separately, we never have to re-export an image patch
			continue;
		}
		auto err = SaveImageFile(imgIO, patchTex, filetype, filename);
		if (!err.OK()) {
#pragma omp critical(firstError)
			firstErr = err;
			haveErr  = true;
		}
	}
	return firstErr;
}

Error ExportLabeledImagePatches_Frame_Polygons(std::string dir, int64_t frameTime, const ImageLabels& labels, const gfx::Image& frameImg) {
	IMQS_ASSERT(labels.HasPolygons());

	gfx::ImageIO imgIO;

	auto srcFilename = dir + "/" + tsf::fmt("%09d-whole.jpeg", frameTime);
	auto segFilename = dir + "/" + tsf::fmt("%09d-class.png", frameTime);
	if (!os::IsFile(srcFilename)) {
		// since our images are just video frames, we never need to re-export a frame
		auto err = SaveImageFile(imgIO, frameImg, gfx::ImageType::Jpeg, srcFilename);
		if (!err.OK())
			return err;
	}

	gfx::Canvas   canvas(frameImg.Width, frameImg.Height, gfx::Color8(0, 0, 0, 255));
	vector<float> vx;
	for (auto& lab : labels.Labels) {
		if (!lab.IsPolygon())
			continue;
		vx.clear();
		for (auto& v : lab.Polygon.Vertices) {
			vx.push_back((float) v.X);
			vx.push_back((float) v.Y);
		}
		// Note we use antialiased rendering, intended for MSE loss.
		// We'll need to binaries the output if we're going to be doing BCE loss, or multiple categories.
		canvas.FillPoly((int) lab.Polygon.Vertices.size(), &vx[0], 2 * sizeof(float), gfx::Color8(255, 255, 255, 255));
	}

	auto err = SaveImageFile(imgIO, *canvas.GetImage(), gfx::ImageType::Png, segFilename);
	if (!err.OK())
		return err;

	return Error();
}

Error ExportLabeledImagePatches_Video_Bulk(ExportTypes type, std::string rootDir, const LabelTaxonomy& taxonomy) {
	auto           labelsDir = path::Join(rootDir, "labels");
	vector<string> videoFiles;
	auto           err = os::FindFiles(labelsDir, [&](const os::FindFileItem& item) -> bool {
        if (item.IsDir) {
            videoFiles.push_back(path::Join(rootDir, item.Name));
            return false;
        }
        return true;
    });
	if (!err.OK())
		return err;

	for (size_t i = 0; i < videoFiles.size(); i++) {
		auto progress = [&](size_t pos, size_t total) -> bool {
			tsf::print("Video %v. File %v/%v\r", videoFiles[i], pos + 1, total);
			fflush(stdout);
			return true;
		};
		VideoLabels labels;
		auto        err = LoadVideoLabels(videoFiles[i], labels);
		if (!err.OK())
			return err;
		err = ExportLabeledImagePatches_Video(type, videoFiles[i], taxonomy, labels, progress);
		if (!err.OK())
			return err;
		tsf::print("\n");
	}
	tsf::print("\n");
	return Error();
}

Error ExportLabeledImagePatches_Video(ExportTypes type, std::string videoFilename, const LabelTaxonomy& taxonomy, const VideoLabels& labels, ProgressCallback prog) {
	auto dir = ImagePatchDir(videoFilename);
	auto err = os::MkDirAll(dir);
	if (!err.OK())
		return err;

#ifdef _WIN32
	video::VideoFile video;
	bool             enableSeek = true;
#else
	video::NVVideo video;
	bool           enableSeek = false;
#endif
	err = video.OpenFile(videoFilename);
	if (!err.OK())
		return err;

	gfx::Image img(gfx::ImageFormat::RGBA, video.Width(), video.Height());

	int64_t lastFrameTime = 0;
	int64_t micro         = 1000000;

	for (size_t i = 0; i < labels.Frames.size(); i++) {
		const auto& frame = labels.Frames[i];
		if (type == ExportTypes::Segmentation && !frame.HasPolygons())
			continue;
		if (type != ExportTypes::Segmentation && !frame.HasRects())
			continue;

		// Only seek if frame is more than 20 seconds into the future. Haven't measured optimal metric to use here.
		if (enableSeek && frame.Time - lastFrameTime > 20 * micro) {
			int64_t buffer = 5 * micro; // seek 5 seconds behind frame target
			err            = video.SeekToMicrosecond(frame.Time - buffer);
			if (!err.OK())
				return err;
		}

		while (true) {
			double frameSeconds = 0;
			err                 = video.DecodeFrameRGBA(img.Width, img.Height, img.Data, img.Stride, &frameSeconds);
			if (!err.OK())
				break;
			//int64_t pts   = video.LastFrameTimeMicrosecond();
			int64_t pts   = frameSeconds * 1000000;
			lastFrameTime = pts;
			if (pts == frame.Time) {
				// found our frame
				if (type == ExportTypes::Segmentation)
					err = ExportLabeledImagePatches_Frame_Polygons(dir, frame.Time, frame, img);
				else
					err = ExportLabeledImagePatches_Frame_Rect(type, dir, frame.Time, frame, img);
				if (!err.OK())
					break;

				if (prog != nullptr) {
					if (!prog(i, labels.Frames.size()))
						err = Error("Cancelled");
				}
				break;
			} else if (pts > frame.Time) {
				err = Error::Fmt("Fail to find frame %v", frame.Time);
				break;
			}
		}
		if (!err.OK()) {
			return Error::Fmt("Error while finding frame %v: %v", frame.Time, err.Message());
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

		// BROKEN by move to multiple classes per patch
		IMQS_ASSERT(false);
		//dstLabelsPtr[i] = classToIndex.get(label.Class);
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