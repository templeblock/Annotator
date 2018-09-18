/*
* Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <cuda_runtime.h>
#include "NvCodecUtils.h"

// [BMH 2018-09-18]
// I added sRGB resizing for the Luma channel. If you don't do this, then the image gets darkened.
// HOWEVER, I don't know what the right thing to do is, for 16-bit data.
// This code naively assumes that the 16-bit data is also sRGB encoded, which I think is unlikely
// to be true.
#define sRGB_RESIZE

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<class TOut>
__device__ static TOut LinearTosRGB(float v) {
    const float a = 0.055f;
    const float scale =  (1 << (sizeof(TOut) * 8)) - 1;
	v = v <= 0.0031308f ? 12.92f * v : (1.0f + a) * pow(v, (1.0f / 2.4f));
	return (TOut) Clamp(v * scale, 0.0f, scale);
}

template<typename YuvUnitx2>
static __global__ void Resize(cudaTextureObject_t texY, cudaTextureObject_t texUv,
        uint8_t *pDst, uint8_t *pDstUV, int nPitch, int nWidth, int nHeight,
        float fxScale, float fyScale)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= nWidth / 2 || iy >= nHeight / 2) {
        return;
    }

    int x = ix * 2;
    int y = iy * 2;
    typedef decltype(YuvUnitx2::x) YuvUnit;
    const int MAX = (1 << (sizeof(YuvUnit) * 8)) - 1;
    
#ifdef sRGB_RESIZE
    YuvUnit* dstY = (YuvUnit*) (pDst + y * nPitch + x * sizeof(YuvUnit));
    dstY[0] = LinearTosRGB<YuvUnit>(tex2D<float>(texY, x / fxScale, y / fyScale));
    dstY[1] = LinearTosRGB<YuvUnit>(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale));
    
    y++;
    dstY += nPitch / sizeof(YuvUnit);
    dstY[0] = LinearTosRGB<YuvUnit>(tex2D<float>(texY, x / fxScale, y / fyScale));
    dstY[1] = LinearTosRGB<YuvUnit>(tex2D<float>(texY, (x + 1) / fxScale, y / fyScale));
#else
    YuvUnit* dstY = (YuvUnit*) (pDst + y * nPitch + x * sizeof(YuvUnit));
    dstY[0] = (YuvUnit) (tex2D<float>(texY, x / fxScale, y / fyScale) * MAX);
    dstY[1] = (YuvUnit) (tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX);

    y++;
    dstY += nPitch / sizeof(YuvUnit);
    dstY[0] = (YuvUnit) (tex2D<float>(texY, x / fxScale, y / fyScale) * MAX);
    dstY[1] = (YuvUnit) (tex2D<float>(texY, (x + 1) / fxScale, y / fyScale) * MAX);
#endif

    float2 uv = tex2D<float2>(texUv, ix / fxScale, iy / fyScale);
    *(YuvUnitx2 *)(pDstUV + iy * nPitch + ix * sizeof(YuvUnitx2)) = YuvUnitx2{ (YuvUnit)(uv.x * MAX), (YuvUnit)(uv.y * MAX) };
}

template <typename YuvUnitx2>
static void Resize(unsigned char *dpDst, unsigned char* dpDstUV, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dpSrc;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(YuvUnitx2::x)>();
    resDesc.res.pitch2D.width = nSrcWidth;
    resDesc.res.pitch2D.height = nSrcHeight;
    resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
#ifdef sRGB_RESIZE
    texDesc.sRGB = 1;
#endif

    cudaTextureObject_t texY = 0;
    ck(cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL));

    resDesc.res.pitch2D.devPtr = dpSrc + nSrcPitch * nSrcHeight;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<YuvUnitx2>();
    resDesc.res.pitch2D.width = nSrcWidth / 2;
    resDesc.res.pitch2D.height = nSrcHeight / 2;
    texDesc.sRGB = 0;

    cudaTextureObject_t texUv = 0;
    ck(cudaCreateTextureObject(&texUv, &resDesc, &texDesc, NULL));

    Resize<YuvUnitx2> << <dim3((nDstWidth + 31) / 32, (nDstHeight + 31) / 32), dim3(16, 16) >> >(texY, texUv, dpDst, dpDstUV,
        nDstPitch, nDstWidth, nDstHeight, 1.0f * nDstWidth / nSrcWidth, 1.0f * nDstHeight / nSrcHeight);

    ck(cudaDestroyTextureObject(texY));
    ck(cudaDestroyTextureObject(texUv));
}

void ResizeNv12(unsigned char *dpDstNv12, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrcNv12, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char* dpDstNv12UV)
{
    unsigned char* dpDstUV = dpDstNv12UV ? dpDstNv12UV : dpDstNv12 + (nDstPitch*nDstHeight);
    return Resize<uchar2>(dpDstNv12, dpDstUV, nDstPitch, nDstWidth, nDstHeight, dpSrcNv12, nSrcPitch, nSrcWidth, nSrcHeight);
}


void ResizeP016(unsigned char *dpDstP016, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrcP016, int nSrcPitch, int nSrcWidth, int nSrcHeight, unsigned char* dpDstP016UV)
{
    unsigned char* dpDstUV = dpDstP016UV ? dpDstP016UV : dpDstP016 + (nDstPitch*nDstHeight);
    return Resize<ushort2>(dpDstP016, dpDstUV, nDstPitch, nDstWidth, nDstHeight, dpSrcP016, nSrcPitch, nSrcWidth, nSrcHeight);
}

static __global__ void Scale(cudaTextureObject_t texSrc,
    uint8_t *pDst, int nPitch, int nWidth, int nHeight,
    float fxScale, float fyScale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nWidth || y >= nHeight)
    {
        return;
    }

    *(unsigned char*)(pDst + (y * nPitch) + x) = (unsigned char)(fminf((tex2D<float>(texSrc, x * fxScale, y * fyScale)) * 255.0f, 255.0f));
}

static __global__ void Scale_uv(cudaTextureObject_t texSrc,
    uint8_t *pDst, int nPitch, int nWidth, int nHeight,
    float fxScale, float fyScale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nWidth || y >= nHeight)
    {
        return;
    }

    float2 uv = tex2D<float2>(texSrc, x * fxScale, y * fyScale);
    uchar2 uvOut = uchar2{ (unsigned char)(fminf(uv.x * 255.0f, 255.0f)), (unsigned char)(fminf(uv.y * 255.0f, 255.0f)) };

    *(uchar2*)(pDst + (y * nPitch) + 2 * x) = uvOut;
}

void ScaleKernelLaunch(unsigned char *dpDst, int nDstPitch, int nDstWidth, int nDstHeight, unsigned char *dpSrc, int nSrcPitch, int nSrcWidth, int nSrcHeight, bool bUVPlane = false) 
{
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dpSrc;
    resDesc.res.pitch2D.desc = bUVPlane ? cudaCreateChannelDesc<uchar2>() : cudaCreateChannelDesc<unsigned char>();
    resDesc.res.pitch2D.width = nSrcWidth;
    resDesc.res.pitch2D.height = nSrcHeight;
    resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;

    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

    cudaTextureObject_t texSrc = 0;
    ck(cudaCreateTextureObject(&texSrc, &resDesc, &texDesc, NULL));

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(((uint32_t)nDstWidth + blockSize.x - 1) / blockSize.x, ((uint32_t)nDstHeight + blockSize.y - 1) / blockSize.y, 1);

    if (bUVPlane)
    {
        Scale_uv << <gridSize, blockSize >> >(texSrc, dpDst,
            nDstPitch, nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight);
    }
    else
    {
        Scale << <gridSize, blockSize >> >(texSrc, dpDst,
            nDstPitch, nDstWidth, nDstHeight, 1.0f * nSrcWidth / nDstWidth, 1.0f * nSrcHeight / nDstHeight);
    }

    ck(cudaGetLastError());
    ck(cudaDestroyTextureObject(texSrc));
}

void ScaleYUV420(unsigned char *dpDstY,
                 unsigned char* dpDstU,
                unsigned char* dpDstV,
                int nDstPitch,
                int nDstChromaPitch,
                int nDstWidth,
                int nDstHeight,
                unsigned char *dpSrcY,
                unsigned char* dpSrcU,
                unsigned char* dpSrcV, 
                int nSrcPitch,
                int nSrcChromaPitch,
                int nSrcWidth,
                int nSrcHeight,
                bool bSemiplanar)
{
    int chromaWidthDst = (nDstWidth + 1) / 2;
    int chromaHeightDst = (nDstHeight + 1) / 2;

    int chromaWidthSrc = (nSrcWidth + 1) / 2;
    int chromaHeightSrc = (nSrcHeight + 1) / 2;

    ScaleKernelLaunch(dpDstY, nDstPitch, nDstWidth, nDstHeight, dpSrcY, nSrcPitch, nSrcWidth, nSrcHeight);

    if (bSemiplanar)
    {
        ScaleKernelLaunch(dpDstU, nDstChromaPitch, chromaWidthDst, chromaHeightDst, dpSrcU, nSrcChromaPitch, chromaWidthSrc, chromaHeightSrc, true);
    }
    else
    {
        ScaleKernelLaunch(dpDstU, nDstChromaPitch, chromaWidthDst, chromaHeightDst, dpSrcU, nSrcChromaPitch, chromaWidthSrc, chromaHeightSrc);
        ScaleKernelLaunch(dpDstV, nDstChromaPitch, chromaWidthDst, chromaHeightDst, dpSrcV, nSrcChromaPitch, chromaWidthSrc, chromaHeightSrc);
    }
}
