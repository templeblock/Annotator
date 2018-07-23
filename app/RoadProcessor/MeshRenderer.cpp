#include "pch.h"
#include "MeshRenderer.h"

#include <glfw/deps/linmath.h>

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
    gl_FragColor = texture2D(tex, uvr) * c;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MeshRenderer::~MeshRenderer() {
	Destroy();
}

Error MeshRenderer::Initialize(int fbWidth, int fbHeight) {
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

	// The GLFW docs recommend that you use a framebuffer object instead of the Window, when rendering offscreen,
	// so we follow that advice here.
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

	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_FRAMEBUFFER_SRGB);

	glViewport(0, 0, FBWidth, FBHeight);
	//Clear(Color8(0, 150, 0, 60));

	auto err = CompileShader(CopyShaderVertex, CopyShaderFrag, CopyShader);
	if (!err.OK())
		return err;
	err = CompileShader(RemovePerspectiveShaderVertex, RemovePerspectiveShaderFrag, RemovePerspectiveShader);
	if (!err.OK())
		return err;

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	//DrawHelloWorldTriangle();

	return Error();
}

void MeshRenderer::Destroy() {
	if (Window) {
		//MakeCurrent();
		glfwDestroyWindow(Window);
	}
	if (IsInitialized) {
		glfwTerminate();
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
	DrawMeshWithShader(CopyShader, m, img, meshRenderRect);
}

void MeshRenderer::DrawMeshWithShader(GLuint shader, const Mesh& m, const gfx::Image& img, gfx::Rect32 meshRenderRect) {
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
	IMQS_ASSERT(locMVP != -1 && locvPos != -1 && locvUV != -1 && locTex != -1 && locvColor != -1);
	//IMQS_ASSERT(locMVP != -1 && locvPos != -1 && locvColor != -1);

	// optional attribs
	auto locvExtra = glGetAttribLocation(shader, "vExtra");
	if (shader == RemovePerspectiveShader)
		IMQS_ASSERT(locvExtra != -1);

	GLuint tex = -1;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, img.Width, img.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.Data);
	IMQS_ASSERT(glGetError() == GL_NO_ERROR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	IMQS_ASSERT(mr.Width() * mr.Height() <= 65535); // If you blow this limit, then just switch to uint32 indices.

	VxGen*    vx       = new VxGen[mr.Width() * mr.Height()];
	int       nindices = (mr.Width() - 1) * (mr.Height() - 1) * 6;
	uint16_t* indices  = new uint16_t[nindices];

	VxGen* vxout = vx;
	for (int y = mr.y1; y < mr.y2; y++) {
		for (int x = mr.x1; x < mr.x2; x++) {
			const auto& mv = m.At(x, y);
			Vec2f       uv = mv.UV;
			uv.x           = uv.x / (float) img.Width;
			uv.y           = uv.y / (float) img.Height;
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

	//GLuint vxBuf = -1;
	//glGenBuffers(1, &vxBuf);
	//glBindBuffer(GL_ARRAY_BUFFER, vxBuf);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

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

	glDrawElements(GL_TRIANGLES, nindices, GL_UNSIGNED_SHORT, indices);

	glDisableVertexAttribArray(locvPos);
	glDisableVertexAttribArray(locvUV);
	glDisableVertexAttribArray(locvColor);
	if (locvExtra != -1)
		glDisableVertexAttribArray(locvExtra);
	//glDeleteBuffers(1, &vxBuf);
	glDeleteTextures(1, &tex);

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);
}

void MeshRenderer::RemovePerspective(const gfx::Image& camera, PerspectiveParams pp) {
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

	DrawMeshWithShader(RemovePerspectiveShader, m, camera);
}

void MeshRenderer::SaveToFile(std::string filename) {
	MakeCurrent();
	Image img;
	CopyDeviceToImage(Rect32(0, 0, FBWidth, FBHeight), 0, 0, img);
	img.SaveFile(filename);
}

Error MeshRenderer::DrawHelloWorldTriangle() {
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
	/*
	stbi_write_png("offscreen.png",
	               width, height, 4,
	               buffer + (width * 4 * (height - 1)),
	               -width * 4);*/
	gfx::ImageIO imgIO;
	imgIO.SavePngFile("offscreen.png", true, width, height, -width * 4, buffer + (width * 4 * (height - 1)), 1);

#if USE_NATIVE_OSMESA
	// Here is where there's nothing
#else
	free(buffer);
#endif

	return Error();
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
	glfwMakeContextCurrent(Window);
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