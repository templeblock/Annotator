#pragma once

namespace imqs {
namespace roadproc {

class Proj {
public:
	projCtx Ctx   = nullptr;
	projPJ  SrcPJ = nullptr;
	projPJ  DstPJ = nullptr;

	~Proj();
	void       Reset();
	void       Init(const char* src, const char* dst);
	bool       Convert(size_t n, double* x, double* y, double* z) const; // z can be null, in which case stride is assumed to be 2 (with z, stride is 3)
	bool       Convert(size_t n, gfx::Vec3d* p) const;
	gfx::Vec3d Convert(const gfx::Vec3d& p) const;
};

} // namespace roadproc
} // namespace imqs