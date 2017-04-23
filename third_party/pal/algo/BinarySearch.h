#pragma once

namespace imqs {
namespace algo {

// Binary Search
// * This will walk to the first in a series of matches
// * There is only one branch in the inner loop
template <typename TData, typename TKey>
size_t BinarySearch(size_t n, const TData* items, const TKey& key, int (*compare)(const TData& item, const TKey& key)) {
	if (n == 0)
		return -1;
	size_t imin = 0;
	size_t imax = n;
	while (imax > imin) {
		size_t imid = (imin + imax) >> 1;
		if (compare(items[imid], key) < 0)
			imin = imid + 1;
		else
			imax = imid;
	}
	if (imin == imax && 0 == compare(items[imin], key))
		return imin;
	else
		return -1;
}

// Binary Search, but always return stopping position, regardless of match
// * This will walk to the first in a series of matches
// * There is only one branch in the inner loop
template <typename TData, typename TKey>
size_t BinarySearchTry(size_t n, const TData* items, const TKey& key, int (*compare)(const TData& item, const TKey& key)) {
	if (n == 0)
		return -1;
	size_t imin = 0;
	size_t imax = n;
	while (imax > imin) {
		size_t imid = (imin + imax) >> 1;
		if (compare(items[imid], key) < 0)
			imin = imid + 1;
		else
			imax = imid;
	}
	return imin;
}

} // namespace algo
} // namespace imqs