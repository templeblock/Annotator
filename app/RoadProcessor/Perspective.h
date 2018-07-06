#pragma once

namespace imqs {
namespace roadproc {

// Perspective correction parameters.
struct ZP {
	float Z1 = 0; // Constant
	float ZX = 0; // X coefficient - aka how much is the plane horizontally tilted in front of us (typically very small for roads)
	float ZY = 0; // Y coefficient - aka what is vertical angle of the camera. typically much larger than the ZX
};

// A frustum that is oriented like an upside-down triangle. The top edge of the frustum
// touches the top-left and top-right corners of the image. The bottom edge of the frustum
// touches the bottom edge of the image, and it's X coordinates are X1 and X2.
struct Frustum {
	int   Width;
	int   Height;
	float X1;
	float X2;

	void DebugPrintParams(float z1, float zx, float zy, int frameWidth, int frameHeight) const;
};

float      FindZ1ForIdentityScaleAtBottom(int frameWidth, int frameHeight, float zx, float zy);
Frustum    ComputeFrustum(int frameWidth, int frameHeight, float z1, float zx, float zy);
gfx::Vec2f FlatToCamera(float x, float y, float z1, float zx, float zy);
void       FlatToCameraInt256(float z1, float zx, float zy, float x, float y, int32_t& u, int32_t& v);
gfx::Vec2f FlatToCamera(int frameWidth, int frameHeight, float x, float y, float z1, float zx, float zy);
gfx::Vec2f CameraToFlat(gfx::Vec2f cam, float z1, float zx, float zy);
gfx::Vec2f CameraToFlat(int frameWidth, int frameHeight, gfx::Vec2f cam, float z1, float zx, float zy);
void       RemovePerspective(const gfx::Image& camera, gfx::Image& flat, float z1, float zx, float zy, float originX, float originY);
void       FitQuadratic(const std::vector<std::pair<double, double>>& xy, double& a, double& b, double& c);
int        Perspective(argparse::Args& args);

} // namespace roadproc
} // namespace imqs