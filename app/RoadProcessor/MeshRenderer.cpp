#include "pch.h"
#include "MeshRenderer.h"

#include <glfw/deps/linmath.h>

namespace imqs {
namespace roadproc {

static void myErrorCallback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

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

	// The GLFW docs recommend that you use a framebuffer object when rendering offscreen, so we do that here
	glGenFramebuffers(1, &FBO);
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glGenTextures(1, &FBTex);
	glBindTexture(GL_TEXTURE_2D, FBTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fbWidth, fbHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FBTex, 0);
	auto fbStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		return Error("Framebuffer not complete");
	FBWidth  = fbWidth;
	FBHeight = fbHeight;

	DrawHelloWorldTriangle();

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

	glViewport(0, 0, width, height);
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

} // namespace roadproc
} // namespace imqs