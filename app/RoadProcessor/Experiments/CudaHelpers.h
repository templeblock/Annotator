#pragma once

#include <stdio.h>

namespace imqs {
namespace roadproc {

// We keep the CUDA code's dependencies small, so this is basically a copy
// of gfx::Image, but for defining a texture for a CUDA kernel

// Why bother keeping the CUDA file dependencies small?
// If we don't then we need to worry about getting the precompiled headers
// working for the .cu compilation. Honestly, I haven't even tried.

struct TexData {
	void* Data            = nullptr;
	int   Stride          = 0;
	int   Width           = 0;
	int   Height          = 0;
	int   BytesPerChannel = 0;
	int   NumChannels     = 0;

	int BytesPerLine() const { return BytesPerChannel * NumChannels * Width; }
};

#ifdef IMQS_ROADPROC_PCH_INCLUDED
inline TexData ImageToTexData(const gfx::Image& img) {
	TexData t;
	t.Data            = img.Data;
	t.Stride          = img.Stride;
	t.Width           = img.Width;
	t.Height          = img.Height;
	t.BytesPerChannel = img.BytesPerPixel() / img.NumChannels();
	t.NumChannels     = img.NumChannels();
	return t;
}
#endif

inline void cuDieOnErrorFunc(int e, const char* file, int line) {
	if (e < 0) {
		printf("CUDA error %d at %s:%d\n", e, file, line);
		*((int*) 1) = 1;
	}
}

#define cuDieOnError(op) cuDieOnErrorFunc(op, __FILE__, __LINE__)

} // namespace roadproc
} // namespace imqs