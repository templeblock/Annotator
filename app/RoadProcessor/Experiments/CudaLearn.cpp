#include "pch.h"
#include "../Globals.h"
#include "../LensCorrection.h"
#include "FrameExtractor.h"
#include "CudaHelpers.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

void TestCuda() {
	Error err;
	Image vignette, distortion, lensCombined;
	err = global::Lens->ComputeCorrection(64, 36, vignette, distortion, &lensCombined);
	// combined lens correction texture size considerations:
	//  32 * 18 * 4 * sizeof(float) =   9216 bytes
	//  64 * 36 * 4 * sizeof(float) =  36864 bytes
	// 128 * 72 * 4 * sizeof(float) = 147456 bytes

	video::NVVideo video;
	video.OutputMode = video::NVVideo::OutputGPU;
	err              = video.OpenFile("/home/ben/mldata/train/ORT Day1 (4).MOV");

	CudaFrameExtractor ex;
	ex.Initialize(ImageToTexData(lensCombined));

	for (size_t i = 0; i < 1; i++) {
		video::NVVideo::CudaFrame frame;
		err = video.DecodeFrameRGBA_GPU(frame);
		if (!err.OK())
			break;
		Image copy;
		copy.Alloc(ImageFormat::RGBA, video.Width(), video.Height());
		err = cuErr(cudaMemcpy2D(copy.Data, copy.Stride, frame.Frame, frame.Stride, video.Width() * 4, video.Height(), cudaMemcpyDeviceToHost));
		copy.SaveFile("/home/ben/hello-cuda-1.png");

		ex.Frame(frame.Frame, frame.Stride, video.Width(), video.Height());
		//LensCorrectCuda(frame.Frame, frame.Stride, video.Width(), video.Height(), ImageToTexData(lensCombined));
		err = cuErr(cudaMemcpy2D(copy.Data, copy.Stride, frame.Frame, frame.Stride, video.Width() * 4, video.Height(), cudaMemcpyDeviceToHost));
		copy.SaveFile("/home/ben/hello-cuda-2.png");

		cudaFree(frame.Frame);
	}
}

} // namespace roadproc
} // namespace imqs