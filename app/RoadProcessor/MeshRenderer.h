#pragma once

namespace imqs {
namespace roadproc {

// GPU based mesh renderer
// This thing takes an unprojected image, with a distortion mesh, and renders that mesh
// onto the gigantic flat earth canvas.
// I've tested creating a framebuffer up to 8192x8192 on a Geforce 1080.
class MeshRenderer {
public:
	~MeshRenderer();

	// Create a GPU rendering context with the given width and height
	Error Initialize(int fbWidth, int fbHeight);
	void  Destroy(); // Called by destructor

	Error DrawHelloWorldTriangle();

private:
	bool        IsInitialized = false;
	GLFWwindow* Window        = nullptr;
	GLuint      FBO           = -1;
	GLuint      FBTex         = -1;
	int         FBWidth       = 0;
	int         FBHeight      = 0;
};

} // namespace roadproc
} // namespace imqs