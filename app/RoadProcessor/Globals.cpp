#include "pch.h"
#include "Globals.h"

namespace imqs {
namespace roadproc {
namespace global {
LensCorrector* Lens;
uint16_t*      LensFixedtoRaw;

Error Initialize() {
	return Error();
}

void Shutdown() {
}

gfx::Vec3d ConvertLLToMerc(const gfx::Vec3d& p) {
	return p;
}

} // namespace global
} // namespace roadproc
} // namespace imqs