#include "pch.h"
#include "FrameExtractor.h"

namespace imqs {
namespace roadproc {

CudaFrameExtractor::~CudaFrameExtractor() {
	cudaDestroyTextureObject(LensTexObj);
	cudaFreeArray(LensArray);
}

void CudaFrameExtractor::Initialize(const TexData& lensCorrectImg) {
	/*
    // Set texture parameters
    LensCorrectTex.addressMode[0] = cudaAddressModeClamp;
    LensCorrectTex.addressMode[1] = cudaAddressModeClamp;
    LensCorrectTex.filterMode = cudaFilterModeLinear;
	LensCorrectTex.normalized = true;    // access with normalized texture coordinates
	*/

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	cuDieOnError(cudaMallocArray(&LensArray, &channelDesc, lensCorrectImg.Width, lensCorrectImg.Height));
	cuDieOnError(cudaMemcpy2DToArray(LensArray, 0, 0, lensCorrectImg.Data, lensCorrectImg.Stride, lensCorrectImg.BytesPerLine(), lensCorrectImg.Height, cudaMemcpyHostToDevice));

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType         = cudaResourceTypeArray;
	resDesc.res.array.array = LensArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeClamp;
	texDesc.addressMode[1]   = cudaAddressModeClamp;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cuDieOnError(cudaCreateTextureObject(&LensTexObj, &resDesc, &texDesc, nullptr));
}

} // namespace roadproc
} // namespace imqs