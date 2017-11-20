#include "pch.h"
#include "aligned_alloc.h"
#include <malloc.h>
#include <stdlib.h>

// uncomment this once all posix systems support it (including Android)
//#define IMQS_HAVE_POSIX_MEMALIGN 1

namespace imqs {

IMQS_PAL_API void* aligned_alloc(size_t bytes, size_t alignment) {
#ifdef _WIN32
	return _aligned_malloc(bytes, alignment);
#elif defined(IMQS_HAVE_POSIX_MEMALIGN)
	void* p = nullptr;
	if (0 != posix_memalign(&p, alignment, bytes))
		return nullptr;
	return p;
#else
	// Since our offset is a single byte, we don't support more than 128-byte alignment.
	if (alignment > 128)
		return nullptr;

	size_t alignment_mask = alignment - 1;

	// Ensure that alignment is a power of 2
	if ((alignment_mask & alignment) != 0)
		return nullptr;

	size_t raw    = (size_t) malloc(bytes + alignment);
	size_t usable = 0;
	if (raw) {
		usable                           = (raw + alignment) & ~alignment_mask;
		*((unsigned char*) (usable - 1)) = (unsigned char) (usable - raw);
	}
	return (void*) usable;
#endif
}

// alignment must be the same as the original block
IMQS_PAL_API void* aligned_realloc(size_t original_block_bytes, void* block, size_t bytes, size_t alignment) {
#ifdef _WIN32
	return _aligned_realloc(block, bytes, alignment);
#else
	void* p = aligned_alloc(bytes, alignment);
	if (!p)
		return nullptr;
	memcpy(p, block, original_block_bytes);
	aligned_free(block);
	return p;
#endif
}

IMQS_PAL_API void aligned_free(void* block) {
#ifdef _WIN32
	_aligned_free(block);
#elif defined(IMQS_HAVE_POSIX_MEMALIGN)
	free(block);
#else
	if (block != nullptr) {
		unsigned char* usable = (unsigned char*) block;
		unsigned char* raw    = usable - usable[-1];
		free(raw);
	}
#endif
}

} // namespace imqs