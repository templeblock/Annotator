#pragma once

namespace imqs {
namespace roadproc {

// Mesh is a 2D distortion mesh
class Mesh {
public:
	struct VX {
		gfx::Vec2f  Pos;                                       // Pixel coordinates in destination image
		gfx::Vec2f  UV;                                        // Pixel coordinates (not normalized) in source image
		gfx::Vec4f  Extra   = gfx::Vec4f(0, 0, 0, 0);          // Interpretation depends on shader
		gfx::Color8 Color   = gfx::Color8(255, 255, 255, 255); // Alpha can be used to lighten parts of the mesh, or tint parts, for debug viz. Colors are sRGB, non-premultiplied
		bool        IsValid = true;                            // Used by OpticalFlow to mark cells as being invalid, because partially outside of frustum

		// float       DeltaStrength = 0; // Not used - concept didn't work
	};

	int Width    = 0;
	int Height   = 0;
	int Count    = 0;
	VX* Vertices = nullptr;

	Mesh();
	Mesh(int width, int height);
	Mesh(const Mesh& m);
	Mesh(Mesh&& m);
	~Mesh();
	Mesh& operator=(const Mesh& m);
	Mesh& operator=(Mesh&& m);

	void       Initialize(int width, int height);
	void       ResetUniformRectangular(gfx::Vec2f topLeft, gfx::Vec2f topRight, gfx::Vec2f bottomLeft, int imgWidth, int imgHeight);
	void       ResetIdentityForWarpMesh(int imgWidth, int imgHeight, int matchRadius, bool moveBottomRowToBottomOfImage);
	void       TransformTargets(gfx::Vec2f translate);
	Error      SaveCompact(std::string filename);
	Error      LoadCompact(std::string filename);
	gfx::Vec2f AvgValidDisplacement() const;
	bool       FirstValid(int& x, int& y) const; // Returns false if there are no valid vertices
	gfx::RectF PosBounds() const;

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

	void SetPosAndUV(int x, int y, const gfx::Vec2f& pos, const gfx::Vec2f& uv) {
		At(x, y).Pos = pos;
		At(x, y).UV  = uv;
	}

	// Return interpolated Pos
	gfx::Vec2f PosAtFractionalUV(float u, float v) const;
	gfx::Vec2f PosAtFractionalUV(gfx::Vec2f uv) const { return PosAtFractionalUV(uv.x, uv.y); }

	//gfx::Vec2f UVimg(int imgWidth, int imgHeight, int x, int y) const {
	//	const auto& p = At(x, y);
	//	return gfx::Vec2f(p.UV.x * (float) imgWidth, p.UV.y * (float) imgHeight);
	//}

	//void Print(gfx::Rect32 rect, int imgWidth, int imgHeight) const;
	void PrintSample(int x, int y) const;
	void Print(gfx::Rect32 rect) const;
	void PrintValid() const;
	void PrintDeltaPos(gfx::Rect32 rect, gfx::Vec2f norm = gfx::Vec2f(FLT_MAX, FLT_MAX)) const;
	void PrintDeltaStrength(gfx::Rect32 rect) const;
	void DrawFlowImage(std::string filename) const;
	void DrawFlowImage(gfx::Rect32 rect, std::string filename) const;
};

} // namespace roadproc
} // namespace imqs