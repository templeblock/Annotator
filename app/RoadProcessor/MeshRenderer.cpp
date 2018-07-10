#include "pch.h"
#include "MeshRenderer.h"

#include <glfw/deps/linmath.h>

using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

static void myErrorCallback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

struct VxGen {
	gfx::Vec3f  Pos;
	gfx::Vec2f  UV;
	gfx::Color8 Color;

	static VxGen Make(const gfx::Vec3f& pos, const gfx::Vec2f& uv) {
		VxGen v;
		v.Pos = pos;
		v.UV  = uv;
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

static const char* CopyShaderVertex = R"(
#version 110
uniform mat4 MVP;
attribute vec3 vPos;
attribute vec2 vUV;
attribute vec4 vColor;
varying vec2 uv;
varying vec4 color;
void main()
{
    gl_Position = MVP * vec4(vPos, 1.0);
	color = vColor;
	uv = vUV;
}
)";

static const char* CopyShaderFrag = R"(
#version 110
uniform sampler2D tex;
varying vec2 uv;
varying vec4 color;
void main()
{
    gl_FragColor = texture2D(tex, uv) * color;
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
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// enable this when ready:
	glEnable(GL_FRAMEBUFFER_SRGB);
	// glClearColor(SRGB2Linear(clear.r), SRGB2Linear(clear.g), SRGB2Linear(clear.b), clear.a / 255.0f);

	glViewport(0, 0, FBWidth, FBHeight);
	Clear(Color8(0, 150, 0, 60));

	auto err = CompileShader(CopyShaderVertex, CopyShaderFrag, CopyShader);
	if (!err.OK())
		return err;

	//DrawHelloWorldTriangle();

	return Error();
}

void MeshRenderer::Destroy() {
	if (Window)
		glfwDestroyWindow(Window);
	if (IsInitialized) {
		glfwTerminate();
		IsInitialized = false;
	}
}

void MeshRenderer::Clear(gfx::Color8 color) {
	glClearColor(color.Rf(), color.Gf(), color.Bf(), color.Af());
	glClear(GL_COLOR_BUFFER_BIT);
}

void MeshRenderer::CopyImageToDevice(const Image& img) {
	glUseProgram(CopyShader);

	auto locMVP  = glGetUniformLocation(CopyShader, "MVP");
	auto locvPos = glGetAttribLocation(CopyShader, "vPos");
	auto locvUV  = glGetAttribLocation(CopyShader, "vUV");
	auto locTex  = glGetAttribLocation(CopyShader, "tex");
	IMQS_ASSERT(locMVP != -1 && locvPos != -1 && locvUV != -1 && locTex != -1);

	GLuint tex = -1;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, img.Width, img.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.Data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	VxGen vertices[4] = {
	    {{0, 0, 0}, {0, 0}},
	    {{1, 0, 0}, {1, 0}},
	    {{1, 1, 0}, {1, 1}},
	    {{0, 1, 0}, {0, 1}},
	};
	const int nindices          = 6;
	int       indices[nindices] = {0, 1, 2, 0, 2, 3};

	//GLuint vxBuf = -1;
	//glGenBuffers(1, &vxBuf);
	//glBindBuffer(GL_ARRAY_BUFFER, vxBuf);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	const uint8_t* vptr = (const uint8_t*) vertices;

	glEnableVertexAttribArray(locvPos);
	glVertexAttribPointer(locvPos, 3, GL_FLOAT, GL_FALSE, sizeof(VxGen), vptr + offsetof(VxGen, Pos));
	glEnableVertexAttribArray(locvUV);
	glVertexAttribPointer(locvUV, 2, GL_FLOAT, GL_FALSE, sizeof(VxGen), vptr + offsetof(VxGen, UV));

	Mat4f mvp;
	mvp.MakeIdentity();
	mvp.Ortho(0, FBWidth, 0, FBHeight, -1, 1);
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, &mvp.row[0].x);

	glUniform1i(locTex, 0); // texture unit 0

	glDrawElements(GL_TRIANGLES, nindices, GL_UNSIGNED_SHORT, indices);

	glDisableVertexAttribArray(locvPos);
	glDisableVertexAttribArray(locvUV);
	//glDeleteBuffers(1, &vxBuf);
	glDeleteTextures(1, &tex);
}

void MeshRenderer::CopyDeviceToImage(Image& img) {
	// glReadPixels returns an image that is bottom-up, so we need to flip it, in order to be top-down
	// We make a bunch of assumptions here (inside our assertions), which make the code simpler
	// HMMM. this is actually premultiplied alpha (ImageFormat::RGBAP)
	img.Alloc(ImageFormat::RGBA, FBWidth, FBHeight);
	int width     = FBWidth;
	int height    = FBHeight;
	int stripSize = 64;
	IMQS_ASSERT(height % stripSize == 0);
	uint8_t* buf = (uint8_t*) imqs_malloc_or_die(stripSize * width * 4);
	// read in strips, from the top down
	for (int y = 0; y < height; y += stripSize) {
		glReadPixels(0, height - y - stripSize, width, stripSize, GL_RGBA, GL_UNSIGNED_BYTE, buf);
		uint8_t* src = buf + (stripSize - 1) * width * 4;
		for (int i = 0; i < stripSize; i++) {
			memcpy(img.At(0, y + i), src, width * 4);
			src -= width * 4;
		}
	}
	free(buf);
}

void MeshRenderer::DrawMesh(const Mesh& m, const gfx::Image& img) {
	glUseProgram(CopyShader);

	auto locMVP    = glGetUniformLocation(CopyShader, "MVP");
	auto locvPos   = glGetAttribLocation(CopyShader, "vPos");
	auto locvUV    = glGetAttribLocation(CopyShader, "vUV");
	auto locvColor = glGetAttribLocation(CopyShader, "vColor");
	auto locTex    = glGetUniformLocation(CopyShader, "tex");
	IMQS_ASSERT(locMVP != -1 && locvPos != -1 && locvUV != -1 && locTex != -1 && locvColor != -1);
	//IMQS_ASSERT(locMVP != -1 && locvPos != -1 && locvColor != -1);

	GLuint tex = -1;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, img.Width, img.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.Data);
	IMQS_ASSERT(glGetError() == GL_NO_ERROR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	IMQS_ASSERT(m.Width * m.Height <= 65535); // If you blow this limit, then just switch to uint32 indices.

	VxGen*    vx       = new VxGen[m.Width * m.Height];
	int       nindices = (m.Width - 1) * (m.Height - 1) * 6;
	uint16_t* indices  = new uint16_t[nindices];

	VxGen* vxout = vx;
	for (int y = 0; y < m.Height; y++) {
		for (int x = 0; x < m.Width; x++) {
			*vxout++        = VxGen::Make(Vec3f(m.At(x, y), 0.f), m.UVAt(x, y));
			vxout[-1].Color = Color8(255, 255, 255, 255);
		}
	}

	uint16_t* indout = indices;
	for (int y = 0; y < m.Height - 1; y++) {
		size_t row1 = y * m.Width;
		size_t row2 = (y + 1) * m.Width;
		for (int x = 0; x < m.Width - 1; x++) {
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

	Mat4f mvp;
	mvp.MakeIdentity();
	mvp.Ortho(0, FBWidth, FBHeight, 0, -1, 1);
	auto mvpT = mvp;
	mvpT.Transpose();
	glUniformMatrix4fv(locMVP, 1, GL_FALSE, &mvpT.row[0].x); // GLES doesn't support TRANSPOSE = TRUE
	IMQS_ASSERT(glGetError() == GL_NO_ERROR);

	glUniform1i(locTex, 0); // texture unit 0

	glDrawElements(GL_TRIANGLES, nindices, GL_UNSIGNED_SHORT, indices);

	glDisableVertexAttribArray(locvPos);
	glDisableVertexAttribArray(locvUV);
	glDisableVertexAttribArray(locvColor);
	//glDeleteBuffers(1, &vxBuf);
	glDeleteTextures(1, &tex);

	IMQS_ASSERT(glGetError() == GL_NO_ERROR);
}

Error MeshRenderer::DrawHelloWorldTriangle() {
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

Error MeshRenderer::CompileShader(std::string vertexSrc, std::string fragSrc, GLuint& shader) {
	const char* vertexSrcPtr[] = {vertexSrc.c_str(), nullptr};
	const char* fragSrcPtr[]   = {fragSrc.c_str(), nullptr};

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, vertexSrcPtr, nullptr);
	glCompileShader(vertexShader);
	if (glGetError() != GL_NO_ERROR)
		return Error("Failed to compile vertex shader");

	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragShader, 1, fragSrcPtr, nullptr);
	glCompileShader(fragShader);
	if (glGetError() != GL_NO_ERROR)
		return Error("Failed to compile fragment shader");

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