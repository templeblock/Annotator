#pragma once

#include "Mesh.h"
#include "Perspective.h"

namespace imqs {
namespace roadproc {

// GPU based mesh renderer
// This thing takes an unprojected image, with a distortion mesh, and renders that mesh
// onto the gigantic flat earth canvas.
class MeshRenderer {
public:
	int    FBWidth                 = 0; // Framebuffer width
	int    FBHeight                = 0; // Framebuffer height
	GLuint CopyShader              = -1;
	GLuint RemovePerspectiveShader = -1;

	~MeshRenderer();

	// Create a GPU rendering context with the given width and height
	Error Initialize(int fbWidth, int fbHeight);
	void  Destroy(); // Called by destructor

	void Clear(gfx::Color8 color);
	void CopyDeviceToImage(gfx::Rect32 srcRect, int dstX, int dstY, gfx::Image& img);
	void CopyImageToDevice(const gfx::Image& img, int dstX, int dstY);
	void DrawMesh(const Mesh& m, const gfx::Image& img, gfx::Rect32 meshRenderRect = gfx::Rect32::Inverted());
	void DrawMeshWithShader(GLuint shader, const Mesh& m, const gfx::Image& img1, const gfx::Image* img2 = nullptr, gfx::Rect32 meshRenderRect = gfx::Rect32::Inverted());
	void RemovePerspective(const gfx::Image& camera, const gfx::Image* adjuster, PerspectiveParams pp);
	void SaveToFile(std::string filename);

	Error DrawHelloWorldTriangle();

private:
	bool        IsInitialized = false;
	GLFWwindow* Window        = nullptr;
	GLuint      FBO           = -1;
	GLuint      FBTex         = -1;

	void  MakeCurrent();
	Error CompileShader(std::string vertexSrc, std::string fragSrc, GLuint& shader);
	void  SetTextureLinearFilter();
};

} // namespace roadproc
} // namespace imqs