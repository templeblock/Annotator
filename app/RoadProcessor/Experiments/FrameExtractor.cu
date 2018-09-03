#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "CudaHelpers.h"
#include "FrameExtractor.h"

// Texture reference for 2D float texture
//texture<float, cudaTextureType2D, cudaReadModeElementType> LensCorrectTex;

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

// Increases the redness of the image
static __global__ void HelloWorld(uint8_t* frame, int stride, int width, int height) {
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);
	if (x >= width || y >= height) {
		return;
	}

	uint8_t* px = frame + y * stride + x * 4;
	float r = px[0];
	float g = px[1];
	float b = px[2];
	r = Clamp(r * 1.2f, 0.0f, 255.0f);
	g = Clamp(g * 1.1f, 0.0f, 255.0f);
	b = Clamp(b * 0.9f, 0.0f, 255.0f);
	px[0] = (uint8_t) r;
	px[1] = (uint8_t) g;
	px[2] = (uint8_t) b;
}

__device__ static float sRGBToLinear(uint8_t v) {
	const float a = 0.055f;
	float vf = (float) v / 255.0f;
	return vf <= 0.04045f ? vf / 12.92f : pow((vf + a) / (1.0f + a), 2.4f);
}

__device__ static uint8_t LinearTosRGB(float v) {
	const float a = 0.055f;
	v = v <= 0.0031308f ? 12.92f * v : (1.0f + a) * pow(v, (1.0f / 2.4f));
	return (uint8_t) Clamp(v * 255.0f, 0.0f, 255.0f);
}

static __global__ void LensCorrect(uint8_t* frame, int stride, int width, int height, cudaTextureObject_t lensTex) {
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);
	if (x >= width || y >= height) {
		return;
	}

	uint8_t* px = frame + y * stride + x * 4;
	float r = sRGBToLinear(px[0]);
	float g = sRGBToLinear(px[1]);
	float b = sRGBToLinear(px[2]);
	float nx = ((float) x + 0.5f) / (float) width;
	float ny = ((float) y + 0.5f) / (float) height;
	float4 lc = tex2D<float4>(lensTex, nx, ny);
	lc.x *= 0.75f; // without this, we get more highlight clipping
	r *= lc.x;
	g *= lc.x;
	b *= lc.x;
	px[0] = LinearTosRGB(r);
	px[1] = LinearTosRGB(g);
	px[2] = LinearTosRGB(b);
}

namespace imqs {
namespace roadproc {

void HelloWorldCuda(void* frame, int stride, int width, int height) {
	dim3 blockSize(16, 16);
	dim3 gridSize((width + 15) / 16, (height + 15) / 16);
	HelloWorld<<<gridSize, blockSize>>>((uint8_t*) frame, stride, width, height);
}


void CudaFrameExtractor::Frame(void* frame, int stride, int width, int height) {
	dim3 blockSize(16, 16);
	dim3 gridSize((width + 15) / 16, (height + 15) / 16);
	LensCorrect<<<gridSize, blockSize>>>((uint8_t*) frame, stride, width, height, LensTexObj);
}

}
}