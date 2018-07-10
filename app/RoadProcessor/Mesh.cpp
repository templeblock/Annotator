#include "pch.h"
#include "Mesh.h"

using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

Mesh::Mesh() {
}

Mesh::Mesh(int width, int height) {
	Initialize(width, height);
}

Mesh::~Mesh() {
	free(Pos);
}

void Mesh::Initialize(int width, int height) {
	free(Pos);
	Width  = width;
	Height = height;
	Pos    = (Vec2f*) imqs_malloc_or_die(width * height * sizeof(Vec2f));
}

void Mesh::ResetUniformRectangular(gfx::Vec2f topLeft, gfx::Vec2f topRight, gfx::Vec2f bottomLeft) {
	auto right = topRight - topLeft;
	auto down  = bottomLeft - topLeft;
	for (int x = 0; x < Width; x++) {
		for (int y = 0; y < Height; y++) {
			At(x, y) = topLeft + ((float) x / (float) (Width - 1)) * right + ((float) y / (float) (Height - 1)) * down;
		}
	}
}

} // namespace roadproc
} // namespace imqs