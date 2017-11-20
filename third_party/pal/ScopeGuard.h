#pragma once

#include <functional>

namespace imqs {

// Runs a lambda function when it goes out of scope.
// This is intended to be similar to Go's "defer" mechanism.
class ScopeGuard {
public:
	std::function<void()> Func;

	ScopeGuard(std::function<void()> f) : Func(f) {
	}

	~ScopeGuard() {
		Func();
	}
};

} // namespace imqs
