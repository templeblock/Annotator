#pragma once

namespace imqs {

/*
How do we implement this manually?

Alignment must be a power of 2. If it is not, then aligned_alloc returns NULL.

Some examples illustrating 4 byte alignment, allocating 8 bytes of usable memory:

. Wasted byte at start
# Our byte that tells us how many bytes of wasted space before the usable memory
- Usable, aligned memory
* Wasted byte at end

Bytes before alignment point
			1					#--------***			(# = 1) There are 3 bytes extra at the end of the usable space
			2					.#--------**			(# = 2) There are 2 bytes extra at the end of the usable space
			3					..#--------*			(# = 3) There is 1 byte extra at the end of the usable space
			4					...#--------			(# = 4) Original malloc result was perfect. We had to burn 4 bytes. Zero bytes extra at the end of the usable space.

We always allocate (bytes + alignment), we always waste "alignment" bytes.
*/

IMQS_PAL_API void* aligned_alloc(size_t bytes, size_t alignment);
IMQS_PAL_API void* aligned_realloc(size_t original_block_bytes, void* block, size_t bytes, size_t alignment);
IMQS_PAL_API void  aligned_free(void* block);

} // namespace imqs