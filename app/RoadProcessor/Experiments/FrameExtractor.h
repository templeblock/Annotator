#pragma once

#include "CudaHelpers.h"

namespace imqs {
namespace roadproc {
void HelloWorldCuda(void* frame, int stride, int width, int height);

class CudaFrameExtractor {
public:
	cudaArray*          LensArray  = nullptr;
	cudaTextureObject_t LensTexObj = 0;

	~CudaFrameExtractor();

	void Initialize(const TexData& lensCorrectImg);
	void Frame(void* frame, int stride, int width, int height);
};

} // namespace roadproc
} // namespace imqs