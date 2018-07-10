#pragma once

namespace imqs {
namespace roadproc {

// Mesh is a 2D distortion mesh
class Mesh {
public:
	int         Width  = 0;
	int         Height = 0;
	gfx::Vec2f* Pos    = nullptr; // The distortion position of this grid cell

	Mesh();
	Mesh(int width, int height);
	~Mesh();
	void Initialize(int width, int height);
	void ResetUniformRectangular(gfx::Vec2f topLeft, gfx::Vec2f topRight, gfx::Vec2f bottomLeft);

	const gfx::Vec2f& At(int x, int y) const { return Pos[y * Width + x]; }
	gfx::Vec2f&       At(int x, int y) { return Pos[y * Width + x]; }

	gfx::Vec2f UVAt(int x, int y) const { return gfx::Vec2f((float) x / (float) (Width - 1), (float) y / (float) (Height - 1)); }
};

} // namespace roadproc
} // namespace imqs