#pragma once

namespace imqs {

inline Error cuMakeErr(int e, const char* file, int line) {
	if (e < 0)
		return Error::Fmt("CUDA error %v at %v:%v", e, file, line);
	else
		return Error();
}

	//inline Error cuMakeErr(cudaError_t e, const char* file, int line) {
	//	if (e < 0)
	//		return Error::Fmt("CUDA error %v at %v:%v", e, file, line);
	//	else
	//		return Error();
	//}

#define cuErr(op) cuMakeErr(op, __FILE__, __LINE__)

} // namespace imqs