#include "pch.h"
#include "Proj.h"

namespace imqs {
namespace roadproc {

Proj::~Proj() {
	Reset();
}

void Proj::Reset() {
	if (SrcPJ)
		pj_free(SrcPJ);
	if (DstPJ)
		pj_free(DstPJ);
	pj_ctx_free(Ctx);
}

void Proj::Init(const char* src, const char* dst) {
	Reset();
	Ctx   = pj_ctx_alloc();
	SrcPJ = pj_init_plus_ctx(Ctx, src);
	DstPJ = pj_init_plus_ctx(Ctx, dst);
	IMQS_ASSERT(SrcPJ);
	IMQS_ASSERT(DstPJ);
}

gfx::Vec3d Proj::Convert(const gfx::Vec3d& p) const {
	auto copy = p;
	Convert(1, &copy);
	return copy;
}

bool Proj::Convert(size_t n, gfx::Vec3d* p) const {
	return Convert(n, &p[0].x, &p[0].y, nullptr);
}

bool Proj::Convert(size_t n, double* x, double* y, double* z) const {
	int stride = z ? 3 : 2;

	if (pj_is_latlong(SrcPJ)) {
		double* dx = x;
		double* dy = y;
		for (size_t i = 0; i < n; i++, dx += stride, dy += stride) {
			*dx *= IMQS_PI / 180.0;
			*dy *= IMQS_PI / 180.0;
		}
	}

	if (pj_transform(SrcPJ, DstPJ, (long) n, (int) stride, x, y, z) != 0)
		return false;

	if (pj_is_latlong(DstPJ)) {
		double* dx = x;
		double* dy = y;
		for (size_t i = 0; i < n; i++, dx += stride, dy += stride) {
			*dx *= 180.0 / IMQS_PI;
			*dy *= 180.0 / IMQS_PI;
		}
	}

	return true;
}

} // namespace roadproc
} // namespace imqs