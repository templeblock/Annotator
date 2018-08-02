#pragma once

namespace imqs {
namespace roadproc {

// Perspective correction parameters.
struct PerspectiveParams {
	float Z1 = 0; // Scaling Constant
	float ZX = 0; // X coefficient - aka how much is the plane horizontally tilted in front of us (typically very small for roads)
	float ZY = 0; // Y coefficient - aka what is vertical angle of the camera. typically much larger than the ZX

	PerspectiveParams() {}
	PerspectiveParams(float z1, float zx, float zy) : Z1(z1), ZX(zx), ZY(zy) {}
};

// A frustum that is oriented like an upside-down triangle. The top edge of the frustum
// touches the top-left and top-right corners of the image. The bottom edge of the frustum
// touches the bottom edge of the image, and it's X coordinates are X1 and X2.
struct Frustum {
	int   Width  = 0;
	int   Height = 0;
	float X1     = 0; // In normalized coordinates (ie X1 is typically negative)
	float X2     = 0; // In normalized coordinates (ie X2 is typically positive)

	void DebugPrintParams(float z1, float zx, float zy, int frameWidth, int frameHeight) const;
	void Polygon(gfx::Vec2f poly[4], float expandX = 0, float expandY = 0);
	int  X1FullFrame() const { return Width / 2 + X1; }
	int  X2FullFrame() const { return Width / 2 + X2; }
};

float      FindZ1ForIdentityScaleAtBottom(int frameWidth, int frameHeight, float zx, float zy);
Frustum    ComputeFrustum(int frameWidth, int frameHeight, float z1, float zx, float zy);
Frustum    ComputeFrustum(int frameWidth, int frameHeight, PerspectiveParams pp);
gfx::Vec2f FlatToCamera(float x, float y, float z1, float zx, float zy);
void       FlatToCameraInt256(float z1, float zx, float zy, float x, float y, int32_t& u, int32_t& v);
gfx::Vec2f FlatToCamera(int frameWidth, int frameHeight, float x, float y, float z1, float zx, float zy);
gfx::Vec2f CameraToFlat(gfx::Vec2f cam, float z1, float zx, float zy);
gfx::Vec2f CameraToFlat(int frameWidth, int frameHeight, gfx::Vec2f cam, float z1, float zx, float zy);
gfx::Vec2f CameraToFlat(int frameWidth, int frameHeight, gfx::Vec2f cam, PerspectiveParams pp);
void       RemovePerspective(const gfx::Image& camera, gfx::Image& flat, float z1, float zx, float zy, float originX, float originY);
void       RemovePerspective(const gfx::Image& camera, gfx::Image& flat, PerspectiveParams pp, float originX, float originY);
void       FitQuadratic(const std::vector<std::pair<double, double>>& xy, double& a, double& b, double& c);
int        Perspective(argparse::Args& args);

} // namespace roadproc
} // namespace imqs