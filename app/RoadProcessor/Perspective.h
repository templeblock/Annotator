#pragma once

namespace imqs {
namespace roadproc {

// A frustum that is oriented like an upside-down triangle. The top edge of the frustum
// touches the top-left and top-right corners of the image. The bottom edge of the frustum
// touches the bottom edge of the image, and it's X coordinates are X1 and X2.
struct Frustum {
	int   Width;
	int   Height;
	float X1;
	float X2;

	void DebugPrintParams(float z1, float z2, int frameWidth, int frameHeight) const;
};

float      FindZ1ForIdentityScaleAtBottom(int frameWidth, int frameHeight, float z2);
Frustum    ComputeFrustum(int frameWidth, int frameHeight, float z1, float z2);
gfx::Vec2f FlatToCamera(float x, float y, float z1, float z2);
void       FlatToCameraInt256(float z1, float z2, float x, float y, int32_t& u, int32_t& v);
gfx::Vec2f FlatToCamera(int frameWidth, int frameHeight, float x, float y, float z1, float z2);
gfx::Vec2f CameraToFlat(gfx::Vec2f cam, float z1, float z2);
gfx::Vec2f CameraToFlat(int frameWidth, int frameHeight, gfx::Vec2f cam, float z1, float z2);
void       RemovePerspective(const gfx::Image& camera, gfx::Image& flat, float z1, float z2, float originX, float originY);
void       FitQuadratic(const std::vector<std::pair<double, double>>& xy, double& a, double& b, double& c);
int        Perspective(argparse::Args& args);

} // namespace roadproc
} // namespace imqs