#pragma once

namespace imqs {
namespace roadproc {

// Mesh is a 2D distortion mesh
class Mesh {
public:
	struct VX {
		gfx::Vec2f  Pos;                                       // Pixel coordinates in destination image
		gfx::Vec2f  UV;                                        // Pixel coordinates (not normalized) in source image
		bool        IsValid = true;                            // Used by OpticalFlow to mark cells as being invalid, because partially outside of frustum
		gfx::Color8 Color   = gfx::Color8(255, 255, 255, 255); // Alpha can be used to lighten parts of the mesh, or tint parts, for debug viz
	};

	int Width    = 0;
	int Height   = 0;
	int Count    = 0;
	VX* Vertices = nullptr;

	Mesh();
	Mesh(int width, int height);
	~Mesh();
	void Initialize(int width, int height);
	void ResetUniformRectangular(gfx::Vec2f topLeft, gfx::Vec2f topRight, gfx::Vec2f bottomLeft, int imgWidth, int imgHeight);
	void TransformTargets(gfx::Vec2f translate);

	// Snap each mesh vertex so that it lies in the crack between the four nearest pixels (or on the
	// edge, if it is an edge vertex). We do this so that when we run the optional flow algorithm,
	// we can pick out regions of say 16x16 pixels that precisely surround a vertex. In other words,
	// we fetch 8x8 blocks from the four quadrants, and each of those 8x8 blocks falls precisely on
	// top of 8x8 pixels from the source image.
	//void SnapToPixelEdges(int imgWidth, int imgHeight);
	void SnapToUVPixelEdges();

	const VX& At(int x, int y) const {
		//IMQS_ASSERT(x >= 0 && x < Width && y >= 0 && y < Height);
		return Vertices[y * Width + x];
	}
	VX& At(int x, int y) {
		//IMQS_ASSERT(x >= 0 && x < Width && y >= 0 && y < Height);
		return Vertices[y * Width + x];
	}

	//gfx::Vec2f UVimg(int imgWidth, int imgHeight, int x, int y) const {
	//	const auto& p = At(x, y);
	//	return gfx::Vec2f(p.UV.x * (float) imgWidth, p.UV.y * (float) imgHeight);
	//}

	//void Print(gfx::Rect32 rect, int imgWidth, int imgHeight) const;
	void PrintSample(int x, int y) const;
	void Print(gfx::Rect32 rect) const;
	void PrintValid() const;
	void PrintDeltaPos(gfx::Rect32 rect) const;
	void DrawFlowImage(std::string filename) const;
	void DrawFlowImage(gfx::Rect32 rect, std::string filename) const;
};

} // namespace roadproc
} // namespace imqs