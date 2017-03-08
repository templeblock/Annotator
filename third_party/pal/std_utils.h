#pragma once

#include <vector>
#include <algorithm>

// Helper functions that augment the C++ standard library
// There are similar string-specific functions inside strings/strings.h

namespace imqs {
namespace stdutils {

template <typename Iter, typename T>
size_t indexof(Iter& begin, Iter& end, const T& val) {
	auto pos = std::find(begin, end, val);
	if (pos == end)
		return -1;
	return pos - begin;
}

template <typename T>
size_t indexof(std::vector<T>& v, const T& val) {
	auto pos = std::find(v.begin(), v.end(), val);
	if (pos == v.end())
		return -1;
	return pos - v.begin();
}

// Test whether container has at least one member equal to val
template<typename C, typename V>
bool contains(const C& container, const V& val) {
	for (const auto& v : container) {
		if (v == val)
			return true;
	}
	return false;
}

}
}
