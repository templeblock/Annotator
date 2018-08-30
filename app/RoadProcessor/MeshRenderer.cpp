#include "pch.h"
#include "MeshRenderer.h"
#include "LensCorrection.h"

//#include <glfw/deps/linmath.h>

using namespace imqs::gfx;
using namespace std;

namespace imqs {
namespace roadproc {

static void myErrorCallback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

struct VxGen {
	gfx::Vec3f  Pos;
	gfx::Vec2f  UV;
	gfx::Vec4f  Extra;
	gfx::Color8 Color;

	static VxGen Make(const gfx::Vec3f& pos, const gfx::Vec2f& uv, gfx::Color8 color = gfx::Color8(255, 255, 255, 255), gfx::Vec4f extra = gfx::Vec4f(0, 0, 0, 0)) {
		VxGen v;
		v.Pos   = pos;
		v.UV    = uv;
		v.Extra = extra;
		v.Color = color;
		return v;
	}
};

static const struct
{
	float x, y;
	float r, g, b;
} vertices[3] =
    {
        {-0.6f, -0.4f, 1.f, 0.f, 0.f},
        {0.6f, -0.4f, 0.f, 1.f, 0.f},
        {0.f, 0.6f, 0.f, 0.f, 1.f}};

static const char* vertexShaderSrc = R"(
#version 110
uniform mat4 MVP;
attribute vec3 vCol;
attribute vec2 vPos;
varying vec3 color;
void main()
{
    gl_Position = MVP * vec4(vPos, 0.0, 1.0);
    color = vCol;
}
)";

static const char* fragShaderSrc = R"(
#version 110
varying vec3 color;
void main()
{
    gl_FragColor = vec4(color, 1.0);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* GLSLPrefix = R"(
#version 110
float fromSRGB_Component(float srgb)
{
	float sRGB_Low	= 0.0031308;
	float sRGB_a	= 0.055;

	if (srgb <= 0.04045)
		return srgb / 12.92;
	else
		return pow((srgb + sRGB_a) / (1.0 + sRGB_a), 2.4);
}

vec4 fromSRGB(vec4 c)
{
	vec4 linear_c;
	linear_c.r = fromSRGB_Component(c.r);
	linear_c.g = fromSRGB_Component(c.g);
	linear_c.b = fromSRGB_Component(c.b);
	linear_c.a = c.a;
	return linear_c;
}

vec4 premultiply(vec4 c)
{
	return vec4(c.r * c.a, c.g * c.a, c.b * c.a, c.a);
}
)";
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* CopyShaderVertex = R"(
uniform mat4 MVP;
attribute vec3 vPos;
attribute vec2 vUV;
attribute vec4 vColor;
varying vec2 uv;
varying vec4 color;
void main()
{
    gl_Position = MVP * vec4(vPos, 1.0);
	color = premultiply(fromSRGB(vColor));
	uv = vUV;
}
)";

static const char* CopyShaderFrag = R"(
uniform sampler2D tex;
varying vec2 uv;
varying vec4 color;
void main()
{
	vec4 c = color;
	//if (c.a < 0.5)
	//	c.a = 0.0;
	//else
	//	c.a = 1.0;
    gl_FragColor = texture2D(tex, uv) * c;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* RemovePerspectiveShaderVertex = R"(
uniform mat4 MVP;
attribute vec3 vPos;
attribute vec2 vUV;
attribute vec4 vExtra;
attribute vec4 vColor;
varying vec2 uv;
varying vec4 color;
varying vec4 extra;
void main()
{
    gl_Position = MVP * vec4(vPos, 1.0);
	color = vColor;
	extra = vExtra;
	uv = vUV;
}
)";

static const char* RemovePerspectiveShaderFrag = R"(
uniform sampler2D tex;
uniform sampler2D tex2;
varying vec2 uv;
varying vec4 extra;
varying vec4 color;
void main()
{
	// float z  = z1 + zx * x + zy * y;
	// float fx = x / z;
	// float fy = y / z;
	vec4 c = color;
	vec2 uvnorm = uv - vec2(0.5, 0.5);
	float z = extra.x * uvnorm.x + extra.y * uvnorm.y + extra.w;
	uvnorm = (1.0 / z) * uvnorm;
	vec2 uvr = uvnorm + vec2(0.5, 0.5);
	if (uvr.x < 0.0 || uvr.x > 1.0) {
		c *= 0.0;
	}
	vec4 adjust = (255.0/50.0) * texture2D(tex2, uvr).rrra;
    gl_FragColor = texture2D(tex, uvr) * adjust * c;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* LineShaderVertex = R"(
uniform mat4 MVP;
attribute vec3 vPos;
attribute vec2 vUV;
attribute vec4 vExtra;
attribute vec4 vColor;
varying vec2 uv;
varying vec4 extra;
varying vec4 color;
void main()
{
	// with buffer:
	// float buffer = 0.0;
	// pos += vec3(vUV * (buffer + vExtra.x), 0);
	// uv = vUV * (1.0 + buffer/vExtra.x);
	// however, it looks like that buffer is never necessary

	vec3 pos = vPos;
	pos += vec3(vUV * vExtra.x, 0);
    gl_Position = MVP * vec4(pos, 1.0);
	color = premultiply(fromSRGB(vColor));
	uv = vUV;
	extra = vExtra;
}
)";

static const char* LineShaderFrag = R"(
varying vec2 uv;
varying vec4 extra;
varying vec4 color;
void main()
{
	vec4 c = color;
	float distance_from_edge = extra.x - (extra.x * length(uv));
	float a = distance_from_edge;
	a = clamp(a, 0.0, 1.0);
	//c.g = length(uv);
	//c.b = length(uv);
    gl_FragColor = a * c;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MeshRenderer::~MeshRenderer() {
	Destroy();
}

Error MeshRenderer::Initialize(int fbWidth, int fbHeight) {
	/*
	glfwSetErrorCallback(myErrorCallback);

	if (!glfwInit())
		return Error("Failed to initialize glfw");
	IsInitialized = true;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	Window = glfwCreateWindow(100, 100, "Offscreen Renderer", nullptr, nullptr);
	if (!Window) {
		glfwTerminate();
		return Error("Failed to create offscreen glfw window");
	}

	glfwMakeContextCurrent(Window);
	gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
	*/
	const EGLint configAttribs[] = {
	    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
	    EGL_BLUE_SIZE, 8,
	    EGL_GREEN_SIZE, 8,
	    EGL_RED_SIZE, 8,
	    //EGL_DEPTH_SIZE, 8,
	    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
	    EGL_NONE};

	/*
	const int pbufferWidth  = 9;
	const int pbufferHeight = 9;
	const EGLint pbufferAttribs[] = {
	    EGL_WIDTH,
	    pbufferWidth,
	    EGL_HEIGHT,
	    pbufferHeight,
	    EGL_NONE,
	};
	*/

	// 1. Initialize EGL
	Display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (Display == nullptr)
		return Error("eglGetDisplay failed");

	EGLint major = 0, minor = 0;
	eglInitialize(Display, &major, &minor);
	const char* vendor  = eglQueryString(Display, EGL_VENDOR);
	const char* version = eglQueryString(Display, EGL_VERSION);
	//tsf::print("EGL %v.%v, %v %v\n", major, minor, vendor, version);

	// 2. Select an appropriate configuration
	EGLint    numConfigs = 0;
	EGLConfig cfg;

	eglChooseConfig(Display, configAttribs, &cfg, 1, &numConfigs);
	eglBindAPI(EGL_OPENGL_API);

	// 5. Create a context and make it current
	Ctx = eglCreateContext(Display, cfg, EGL_NO_CONTEXT, nullptr);

	eglMakeCurrent(Display, EGL_NO_SURFACE, EGL_NO_SURFACE, Ctx);

	gladLoadGLLoader((GLADloadproc) eglGetProcAddress);

	// The GLFW docs recommend that you use a framebuffer object instead of the Window, when rendering offscreen,
	// so we follow that advice here.
	auto err = ResizeFrameBuffer(fbWidth, fbHeight);
	if (!err.OK())
		return err;

	err = CompileShader(CopyShaderVertex, CopyShaderFrag, CopyShader);
	if (!err.OK())
		return err;
	err = CompileShader(RemovePerspectiveShaderVertex, RemovePerspectiveShaderFrag, RemovePerspectiveShader);
	if (!err.OK())
		return err;
	err = CompileShader(LineShaderVertex, LineShaderFrag, LineShader);
	if (!err.OK())
		return err;

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	//DrawHelloWorldTriangle();

	return Error();
}

Error MeshRenderer::ResizeFrameBuffer(int fbWidth, int fbHeight) {
	MakeCurrent();
	if (FBO != -1) {
		glDeleteFramebuffers(1, &FBO);
		glDeleteTextures(1, &FBTex);
		FBO   = -1;
		FBTex = -1;
	}
	glGenFramebuffers(1, &FBO);
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glGenTextures(1, &FBTex);
	glBindTexture(GL_TEXTURE_2D, FBTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, fbWidth, fbHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FBTex, 0);
	auto fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		return Error("Framebuffer not complete");
	FBWidth  = fbWidth;
	FBHeight = fbHeight;
	glViewport(0, 0, FBWidth, FBHeight);

	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_FRAMEBUFFER_SRGB);
	return Error();
}

void MeshRenderer::Destroy() {
	//if (Window) {
	//	//MakeCurrent();
	//	glfwDestroyWindow(Window);
	//}
	if (Ctx) {
		eglDestroyContext(Display, Ctx);
	}
	if (IsInitialized) {
		eglTerminate(Display);
		//glfwTerminate();
		IsInitialized = false;
	}
}

void MeshRenderer::Clear(gfx::Color8 color) {
	MakeCurrent();
	glClearColor(color.RLinear(), color.GLinear(), color.BLinear(), color.Af());
	glClear(GL_COLOR_BUFFER_BIT);
}

void MeshRenderer::CopyDeviceToImage(gfx::Rect32 srcRect, int dstX, int dstY, Image& img) {
	MakeCurrent();
	if (img.Width == 0) {
		img.Alloc(ImageFormat::RGBAP, srcRect.Width(), srcRect.Height());
	} else {
		IMQS_ASSERT(img.Width >= srcRect.Width());
		IMQS_ASSERT(img.Height >= srcRect.Height());
	}
	glPixelStorei(GL_PACK_ROW_LENGTH, img.Stride / 4);
	glReadPixels(srcRect.x1, srcRect.y1, srcRect.Width(), srcRect.Height(), GL_RGBA, GL_UNSIGNED_BYTE, img.At(dstX, dstY));
	glPixelStorei(GL_PACK_ROW_LENGTH, 0);
	IMQS_ASSERT(glGetError() == GL_NO_ERROR);
}

void MeshRenderer::CopyImageToDevice(const gfx::Image& img, int dstX, int dstY) {
	MakeCurrent();
	Vec2f topLeft  = Vec2f((float) dstX, (float) dstY);
	Vec2f topRight = topLeft + Vec2f(img.Width, 0);
	Vec2f botLeft  = topLeft + Vec2f(0, img.Height);
	Mesh  m(2, 2);
	m.ResetUniformRectangular(topLeft, topRight, botLeft, img.Width, img.Height);
	DrawMesh(m, img);
}

void MeshRenderer::DrawMesh(const Mesh& m, const gfx::Image& img, gfx::Rect32 meshRenderRect) {
	MakeCurrent();
	DrawMeshWithShader(CopyShader, m, img, nullptr, meshRenderRect);
}

void MeshRenderer::DrawMeshWithShader(GLuint shader, const Mesh& m, const gfx::Image& img1, const gfx::Image* img2, gfx::Rect32 meshRenderRect) {
	MakeCurrent();
	if (meshRenderRect.IsInverted())
		meshRenderRect = Rect32(0, 0, m.Width, m.Height);
	auto mr = meshRenderRect;

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	glUseProgram(shader);

	// mandatory attribs
	auto locMVP    = glGetUniformLocation(shader, "MVP");
	auto locvPos   = glGetAttribLocation(shader, "vPos");
	auto locvUV    = glGetAttribLocation(shader, "vUV");
	auto locvColor = glGetAttribLocation(shader, "vColor");
	auto locTex    = glGetUniformLocation(shader, "tex");
	IMQS_ASSERT(locMVP != -1 && locvPos != -1 && locvUV != -1 && locvColor != -1 && locTex != -1);
	//IMQS_ASSERT(locMVP != -1 && locvPos != -1 && locvColor != -1);

	// optional attribs
	auto locvExtra = glGetAttribLocation(shader, "vExtra");
	auto locTex2   = glGetUniformLocation(shader, "tex2");
	//if (shader == RemovePerspectiveShader)
	//	IMQS_ASSERT(locvExtra != -1 && locTex2 != -1);

	GLuint tex[2];
	glGenTextures(2, tex);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex[0]);
	// this texture is always real world colors, so it's sRGB
	SetTexture2D(img1, true);
	SetTextureLinearFilter();

	if (img2) {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, tex[1]);
		// this is used as an adjustment texture (lens correction), so it's linear, not sRGB
		SetTexture2D(*img2, false);
		SetTextureLinearFilter();
	}

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	IMQS_ASSERT(mr.Width() * mr.Height() <= 65535); // If you blow this limit, then just switch to uint32 indices.

	VxGen*    vx       = new VxGen[mr.Width() * mr.Height()];
	int       nindices = (mr.Width() - 1) * (mr.Height() - 1) * 6;
	uint16_t* indices  = new uint16_t[nindices];

	VxGen* vxout = vx;
	for (int y = mr.y1; y < mr.y2; y++) {
		for (int x = mr.x1; x < mr.x2; x++) {
			const auto& mv = m.At(x, y);
			Vec2f       uv = mv.UV;
			uv.x           = uv.x / (float) img1.Width;
			uv.y           = uv.y / (float) img1.Height;
			*vxout++       = VxGen::Make(Vec3f(mv.Pos, 0), uv, mv.Color, mv.Extra);
		}
	}

	uint16_t* indout = indices;
	for (int y = 0; y < mr.Height() - 1; y++) {
		size_t row1 = y * mr.Width();
		size_t row2 = (y + 1) * mr.Width();
		for (int x = 0; x < mr.Width() - 1; x++) {
			// triangle 1 (top right)
			*indout++ = row1 + x;
			*indout++ = row2 + x + 1;
			*indout++ = row1 + x + 1;
			// triangle 2 (bottom left)
			*indout++ = row1 + x;
			*indout++ = row2 + x;
			*indout++ = row2 + x + 1;
		}
	}
	IMQS_ASSERT(indout == indices + nindices);

	const uint8_t* vptr = (const uint8_t*) vx;

	glEnableVertexAttribArray(locvPos);
	glVertexAttribPointer(locvPos, 3, GL_FLOAT, GL_FALSE, sizeof(VxGen), vptr + offsetof(VxGen, Pos));
	glEnableVertexAttribArray(locvUV);
	glVertexAttribPointer(locvUV, 2, GL_FLOAT, GL_FALSE, sizeof(VxGen), vptr + offsetof(VxGen, UV));
	glEnableVertexAttribArray(locvColor);
	glVertexAttribPointer(locvColor, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(VxGen), vptr + offsetof(VxGen, Color));
	if (locvExtra != -1) {
		glEnableVertexAttribArray(locvExtra);
		glVertexAttribPointer(locvExtra, 4, GL_FLOAT, GL_FALSE, sizeof(VxGen), vptr + offsetof(VxGen, Extra));
	}

	Mat4f mvp;
	mvp.MakeIdentity();
	mvp.Ortho(0, FBWidth, 0, FBHeight, -1, 1);
	auto mvpT = mvp;
	mvpT.Transpose();
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, &mvpT.row[0].x); // GLES doesn't support TRANSPOSE = TRUE
	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	glUniform1i(locTex, 0); // texture unit 0
	if (locTex2 != -1)
		glUniform1i(locTex2, 1); // texture unit 1

	glDrawElements(GL_TRIANGLES, nindices, GL_UNSIGNED_SHORT, indices);

	glDisableVertexAttribArray(locvPos);
	glDisableVertexAttribArray(locvUV);
	glDisableVertexAttribArray(locvColor);
	if (locvExtra != -1)
		glDisableVertexAttribArray(locvExtra);
	glDeleteTextures(2, tex);

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);
}

void MeshRenderer::DrawMeshWireframe(const Mesh& m, gfx::Color8 color, float width, gfx::Rect32 meshRenderRect) {
	MakeCurrent();
	if (meshRenderRect.IsInverted())
		meshRenderRect = Rect32(0, 0, m.Width, m.Height);
	auto mr = meshRenderRect;

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	vector<Vec2f> lv;
	for (int y = mr.y1; y < mr.y2; y++) {
		for (int x = mr.x1; x < mr.x2; x++) {
			const auto& v = m.At(x, y);
			IMQS_ASSERT(__finite(v.Pos.x) && __finite(v.Pos.y));
			if (x < mr.x2 - 1) {
				lv.emplace_back(v.Pos.x, v.Pos.y);
				const auto& vx = m.At(x + 1, y);
				lv.emplace_back(vx.Pos.x, vx.Pos.y);
			}
			if (y < mr.y2 - 1) {
				lv.emplace_back(v.Pos.x, v.Pos.y);
				const auto& vy = m.At(x, y + 1);
				lv.emplace_back(vy.Pos.x, vy.Pos.y);
			}
		}
	}

	DrawLines(lv.size() / 2, &lv[0], color, width);
}

void MeshRenderer::DrawLines(size_t nlines, const gfx::Vec2f* linevx, gfx::Color8 color, float width) {
	MakeCurrent();

	glUseProgram(LineShader);

	// We're roughly following https://blog.mapbox.com/drawing-antialiased-lines-with-opengl-8766f34192dc here, to draw lines

	if (width < 1.0f) {
		// this prevents thin lines from becoming "ropey". See http://www.humus.name/index.php?page=3D&ID=89
		color.a = (uint8_t) math::Clamp<float>((float) color.a * width, 0, 255);
		width   = 1.0f;
	}

	auto locMVP    = glGetUniformLocation(LineShader, "MVP");
	auto locvPos   = glGetAttribLocation(LineShader, "vPos");
	auto locvUV    = glGetAttribLocation(LineShader, "vUV");
	auto locvColor = glGetAttribLocation(LineShader, "vColor");
	auto locvExtra = glGetAttribLocation(LineShader, "vExtra");
	IMQS_ASSERT(locMVP != -1 && locvPos != -1 && locvUV != -1 && locvColor != -1 && locvExtra != -1);

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	VxGen*    vx       = new VxGen[nlines * 4];
	int       nindices = nlines * 6;
	uint32_t* indices  = new uint32_t[nindices];
	size_t    iout     = 0;
	size_t    iv       = 0;

	for (size_t i = 0; i < nlines; i++) {
		const auto& s1   = linevx[i * 2];
		const auto& s2   = linevx[i * 2 + 1];
		Vec2f       dir  = s2 - s1;
		float       size = dir.size();
		size             = max(size, FLT_EPSILON);
		dir /= size;
		Vec2f normal = Vec2f(dir.y, -dir.x);
		Vec4f extra;
		// yellow triangle (mapbox illustration)
		indices[iout++] = iv;
		indices[iout++] = iv + 2;
		indices[iout++] = iv + 1;
		// blue triangle
		indices[iout++] = iv + 1;
		indices[iout++] = iv + 2;
		indices[iout++] = iv + 3;
		vx[iv++]        = VxGen::Make(Vec3f(s1, 0), Vec2f(-normal.x, -normal.y), color, Vec4f(width, 0, 0, 0));
		vx[iv++]        = VxGen::Make(Vec3f(s1, 0), Vec2f(normal.x, normal.y), color, Vec4f(width, 0, 0, 0));
		vx[iv++]        = VxGen::Make(Vec3f(s2, 0), Vec2f(-normal.x, -normal.y), color, Vec4f(width, 0, 0, 0));
		vx[iv++]        = VxGen::Make(Vec3f(s2, 0), Vec2f(normal.x, normal.y), color, Vec4f(width, 0, 0, 0));
	}

	IMQS_ASSERT(iv == nlines * 4);
	IMQS_ASSERT(iout == nindices);

	const uint8_t* vptr = (const uint8_t*) vx;

	glEnableVertexAttribArray(locvPos);
	glVertexAttribPointer(locvPos, 3, GL_FLOAT, GL_FALSE, sizeof(VxGen), vptr + offsetof(VxGen, Pos));
	glEnableVertexAttribArray(locvUV);
	glVertexAttribPointer(locvUV, 2, GL_FLOAT, GL_FALSE, sizeof(VxGen), vptr + offsetof(VxGen, UV));
	glEnableVertexAttribArray(locvColor);
	glVertexAttribPointer(locvColor, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(VxGen), vptr + offsetof(VxGen, Color));
	glEnableVertexAttribArray(locvExtra);
	glVertexAttribPointer(locvExtra, 4, GL_FLOAT, GL_FALSE, sizeof(VxGen), vptr + offsetof(VxGen, Extra));

	Mat4f mvp;
	mvp.MakeIdentity();
	mvp.Ortho(0, FBWidth, 0, FBHeight, -1, 1);
	auto mvpT = mvp;
	mvpT.Transpose();
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, &mvpT.row[0].x); // GLES doesn't support TRANSPOSE = TRUE
	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	glDrawElements(GL_TRIANGLES, nindices, GL_UNSIGNED_INT, indices);

	glDisableVertexAttribArray(locvPos);
	glDisableVertexAttribArray(locvUV);
	glDisableVertexAttribArray(locvColor);
	glDisableVertexAttribArray(locvExtra);

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);
}

void MeshRenderer::SetTexture2D(const gfx::Image& img, bool sRGB) {
	GLint internalFormat;
	GLint format;
	if (img.NumChannels() == 1) {
		internalFormat = GL_R8;
		format         = GL_RED;
	} else if (img.NumChannels() == 4) {
		internalFormat = sRGB ? GL_SRGB8_ALPHA8 : GL_RGBA8;
		format         = GL_RGBA;
	} else {
		IMQS_ASSERT(false);
	}
	glPixelStorei(GL_UNPACK_ROW_LENGTH, img.Stride / img.BytesPerPixel());
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, img.Width, img.Height, 0, format, GL_UNSIGNED_BYTE, img.Data);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
}

void MeshRenderer::SetTextureLinearFilter() {
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void MeshRenderer::RemovePerspective(const gfx::Image& camera, const gfx::Image* adjuster, PerspectiveParams pp) {
	MakeCurrent();
	auto frustum = ComputeFrustum(camera.Width, camera.Height, pp);
	Mesh m;
	m.Initialize(2, 2);

	m.At(0, 0).Pos = Vec2f(0, 0);
	m.At(1, 0).Pos = Vec2f(frustum.Width, 0);

	m.At(0, 1).Pos = Vec2f(0, frustum.Height);
	m.At(1, 1).Pos = Vec2f(frustum.Width, frustum.Height);

	float scale = (float) camera.Width / (float) frustum.Width;

	auto ppNorm = pp;
	ppNorm.Z1 *= scale;
	ppNorm.ZX *= camera.Width;
	ppNorm.ZY *= camera.Height;

	// We don't need the 'flatOrigin' parameter that the CPU version does, because our pixel shader operates on normalized
	// position coordinates.

	for (int i = 0; i < m.Count; i++) {
		m.Vertices[i].UV    = scale * m.Vertices[i].Pos;
		m.Vertices[i].Extra = Vec4f(ppNorm.ZX, ppNorm.ZY, 0, ppNorm.Z1);
	}

	Image nullAdjuster;
	if (!adjuster) {
		nullAdjuster.Alloc(ImageFormat::RGBA, 2, 2);
		int one = LensCorrector::VignetteGrayMultiplier;
		nullAdjuster.Fill(Color8(one, one, one, one));
		adjuster = &nullAdjuster;
	}

	DrawMeshWithShader(RemovePerspectiveShader, m, camera, adjuster);
}

void MeshRenderer::RemovePerspectiveAndCopyOut(const gfx::Image& camera, const gfx::Image* adjuster, PerspectiveParams pp, gfx::Image& flat, gfx::Rect32 extractRect) {
	MakeCurrent();
	Clear(Color8(0, 0, 0, 0));
	RemovePerspective(camera, adjuster, pp);
	if (extractRect.IsInverted())
		extractRect = Rect32(0, 0, FBWidth, FBHeight);
	CopyDeviceToImage(extractRect, 0, 0, flat);
}

void MeshRenderer::SaveToFile(std::string filename) {
	MakeCurrent();
	Image img;
	CopyDeviceToImage(Rect32(0, 0, FBWidth, FBHeight), 0, 0, img);
	img.SaveFile(filename);
}

Error MeshRenderer::DrawHelloWorldTriangle() {
	/*
	MakeCurrent();
	GLuint vertex_buffer, vertex_shader, fragment_shader, program;
	GLint  mvp_location, vpos_location, vcol_location;
	float  ratio;
	int    width, height;
	mat4x4 mvp;
	char*  buffer;

	// NOTE: OpenGL error checks have been omitted for brevity

	glGenBuffers(1, &vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &vertexShaderSrc, NULL);
	glCompileShader(vertex_shader);
	if (glGetError() != GL_NO_ERROR)
		return Error("Failed to compile vertex shader");

	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &fragShaderSrc, NULL);
	glCompileShader(fragment_shader);
	if (glGetError() != GL_NO_ERROR)
		return Error("Failed to compile fragment shader");

	program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);
	if (glGetError() != GL_NO_ERROR)
		return Error("Failed to link vertex/fragment shader");

	mvp_location  = glGetUniformLocation(program, "MVP");
	vpos_location = glGetAttribLocation(program, "vPos");
	vcol_location = glGetAttribLocation(program, "vCol");

	glEnableVertexAttribArray(vpos_location);
	glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
	                      sizeof(vertices[0]), (void*) 0);
	glEnableVertexAttribArray(vcol_location);
	glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE,
	                      sizeof(vertices[0]), (void*) (sizeof(float) * 2));

	width  = FBWidth;
	height = FBHeight;
	//glfwGetFramebufferSize(Window, &width, &height);
	ratio = width / (float) height;

	glClear(GL_COLOR_BUFFER_BIT);

	mat4x4_ortho(mvp, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);

	glUseProgram(program);
	glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*) mvp);
	glDrawArrays(GL_TRIANGLES, 0, 3);

#if USE_NATIVE_OSMESA
	glfwGetOSMesaColorBuffer(window, &width, &height, NULL, (void**) &buffer);
#else
	buffer = (char*) calloc(4, width * height);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
#endif

	// Write image Y-flipped because OpenGL
	//stbi_write_png("offscreen.png",
	//               width, height, 4,
	//               buffer + (width * 4 * (height - 1)),
	//               -width * 4);
	gfx::ImageIO imgIO;
	imgIO.SavePngFile("offscreen.png", true, width, height, -width * 4, buffer + (width * 4 * (height - 1)), 1);

#if USE_NATIVE_OSMESA
	// Here is where there's nothing
#else
	free(buffer);
#endif
	*/
	return Error();
}

void MeshRenderer::DrawTestLines() {
	Vec2f lines[4] = {
	    {10, 10},
	    {50, 50},
	    {100, 100},
	    {200, 130},
	};
	DrawLines(2, lines, Color8(200, 0, 0, 255), 0.7f);
}

static Error CheckShaderCompilation(const std::string& shaderSrc, GLuint shader) {
	int       ilen;
	const int maxBuff = 8000;
	GLchar    ibuff[maxBuff];

	GLint compileStat;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStat);
	glGetShaderInfoLog(shader, maxBuff, &ilen, ibuff);
	if (compileStat == 0)
		return Error::Fmt("Shader failed to compile\nShader: %.50s\nError: %v", shaderSrc, ibuff);
	return Error();
}

void MeshRenderer::MakeCurrent() {
	//glfwMakeContextCurrent(Window);
	eglMakeCurrent(Display, EGL_NO_SURFACE, EGL_NO_SURFACE, Ctx);
}

Error MeshRenderer::CompileShader(std::string vertexSrc, std::string fragSrc, GLuint& shader) {
	const char* vertexSrcPtr[] = {GLSLPrefix, vertexSrc.c_str(), nullptr};
	const char* fragSrcPtr[]   = {GLSLPrefix, fragSrc.c_str(), nullptr};

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 2, vertexSrcPtr, nullptr);
	glCompileShader(vertexShader);
	if (glGetError() != GL_NO_ERROR)
		return Error("Failed to compile vertex shader");
	auto err = CheckShaderCompilation(vertexSrc, vertexShader);
	if (!err.OK())
		return err;

	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragShader, 2, fragSrcPtr, nullptr);
	glCompileShader(fragShader);
	if (glGetError() != GL_NO_ERROR)
		return Error("Failed to compile fragment shader");
	err = CheckShaderCompilation(fragSrc, fragShader);
	if (!err.OK())
		return err;

	shader = glCreateProgram();
	glAttachShader(shader, vertexShader);
	glAttachShader(shader, fragShader);
	glLinkProgram(shader);
	if (glGetError() != GL_NO_ERROR)
		return Error("Failed to link vertex/fragment shader");

	return Error();
}

} // namespace roadproc
} // namespace imqs