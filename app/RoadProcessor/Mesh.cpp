#include "pch.h"
#include "Mesh.h"

using namespace imqs::gfx;
using namespace std;

namespace imqs {
namespace roadproc {

Mesh::Mesh() {
}

Mesh::Mesh(int width, int height) {
	Initialize(width, height);
}

Mesh::~Mesh() {
	delete[] Vertices;
}

void Mesh::Initialize(int width, int height) {
	delete[] Vertices;
	Width    = width;
	Height   = height;
	Count    = Width * Height;
	Vertices = new VX[Count];
}

void Mesh::ResetUniformRectangular(gfx::Vec2f topLeft, gfx::Vec2f topRight, gfx::Vec2f bottomLeft, int imgWidth, int imgHeight) {
	auto right = topRight - topLeft;
	auto down  = bottomLeft - topLeft;
	for (int y = 0; y < Height; y++) {
		for (int x = 0; x < Width; x++) {
			float xf         = ((float) x / (float) (Width - 1));
			float yf         = ((float) y / (float) (Height - 1));
			At(x, y).Pos     = topLeft + xf * right + yf * down;
			At(x, y).UV      = Vec2f(xf * (float) imgWidth, yf * (float) imgHeight);
			At(x, y).IsValid = true;
		}
	}
}

// NOTE: The special tweak that we do at the bottom here turns out to be pointless, because we
// throw these pixels away anyway. We throw them away because of lens vignetting. Perhaps if we
// implement vignetting for lensfun, then this might become relevant again. It might give us
// slightly improved resolution. The reason it "might", is because the camera could be focused
// a little bit away from the nearest point (aka the bottom of the sensor).
void Mesh::ResetIdentityForWarpMesh(int imgWidth, int imgHeight, int matchRadius) {
	// This is for the typical case, where we're generating a warp mesh from a flattened camera frame.
	for (int y = 0; y < Height; y++) {
		for (int x = 0; x < Width; x++) {
			float xf         = ((float) x / (float) (Width - 1));
			float yf         = ((float) y / (float) (Height - 1));
			At(x, y).Pos     = Vec2f(xf * (float) imgWidth, yf * (float) imgHeight);
			At(x, y).UV      = Vec2f(xf * (float) imgWidth, yf * (float) imgHeight);
			At(x, y).IsValid = true;
		}
	}

	// It is vital in this case that the row of vertices which are second from the bottom, are positioned
	// such that they but up against the bottom, precisely. The bottom-most row of vertices must be the
	// edge of the image, so we don't mess with that.
	// What we're doing here is taking the second row from the bottom, and squishing it downwards so that
	// it is exactly matchRadius away from the bottom of the image. These vertices are the most important
	// vertices for optical flow, because they lie in the highest resolution area of the camera, where
	// we care about the image the most.
	int y = Height - 2;
	for (int x = 0; x < Width; x++) {
		auto& p = At(x, y);
		p.Pos.y = imgHeight - matchRadius;
		p.UV.y  = imgHeight - matchRadius;
	}
}

void Mesh::TransformTargets(gfx::Vec2f translate) {
	for (int y = 0; y < Height; y++) {
		for (int x = 0; x < Width; x++) {
			At(x, y).Pos += translate;
		}
	}
}

//void Mesh::SnapToPixelEdges(int imgWidth, int imgHeight) {
void Mesh::SnapToUVPixelEdges() {
	//float imgW = (float) imgWidth;
	//float imgH = (float) imgHeight;
	for (int y = 0; y < Height; y++) {
		for (int x = 0; x < Width; x++) {
			//float pixU    = At(x, y).UV.x * imgW;
			//float pixV    = At(x, y).UV.y * imgH;
			float pixU  = At(x, y).UV.x;
			float pixV  = At(x, y).UV.y;
			float snapU = floor(pixU + 0.5f);
			float snapV = floor(pixV + 0.5f);
			//At(x, y).UV.x = snapU / (float) imgW;
			//At(x, y).UV.y = snapV / (float) imgH;
			At(x, y).UV.x = snapU;
			At(x, y).UV.y = snapV;
			At(x, y).Pos.x += snapU - pixU;
			At(x, y).Pos.y += snapV - pixV;
		}
	}
}

void Mesh::PrintSample(int x, int y) const {
	auto& p = At(x, y);
	tsf::print("%v,%v -> %v,%v (%d,%d,%d,%d)\n", p.UV.x, p.UV.y, p.Pos.x, p.Pos.y, p.Color.r, p.Color.g, p.Color.b, p.Color.a);
}

//void Mesh::Print(gfx::Rect32 rect, int imgWidth, int imgHeight) const {
void Mesh::Print(gfx::Rect32 rect) const {
	for (int y = rect.y1; y < rect.y2; y++) {
		for (int x = rect.x1; x < rect.x2; x++) {
			auto& p = At(x, y);
			tsf::print("%4.1f,%4.1f ", p.Pos.x, p.Pos.y);
		}
		tsf::print(" | ");
		//if (imgWidth != 0) {
		for (int x = rect.x1; x < rect.x2; x++) {
			auto& p = At(x, y);
			//tsf::print("%4.1f,%4.1f ", p.UV.x * imgWidth, p.UV.y * imgHeight);
			tsf::print("%4.1f,%4.1f ", p.UV.x, p.UV.y);
		}
		//}
		tsf::print("\n");
	}
}

void Mesh::PrintValid() const {
	Rect32 rect(0, 0, Width, Height);
	for (int y = rect.y1; y < rect.y2; y++) {
		for (int x = rect.x1; x < rect.x2; x++) {
			tsf::print("%c", At(x, y).IsValid ? '*' : '.');
		}
		tsf::print("\n");
	}
}

void Mesh::PrintDeltaPos(gfx::Rect32 rect) const {
	gfx::Vec2f norm = At(rect.x1, rect.y1).Pos - At(rect.x1, rect.y1).UV;
	tsf::print("X Delta:\n");
	for (int y = rect.y1; y < rect.y2; y++) {
		for (int x = rect.x1; x < rect.x2; x++) {
			auto delta = At(x, y).Pos - At(x, y).UV;
			tsf::print("%2.0f ", At(x, y).IsValid ? delta.x - norm.x : 0);
		}
		tsf::print("\n");
	}
	tsf::print("Y Delta:\n");
	for (int y = rect.y1; y < rect.y2; y++) {
		for (int x = rect.x1; x < rect.x2; x++) {
			auto delta = At(x, y).Pos - At(x, y).UV;
			tsf::print("%3.0f ", At(x, y).IsValid ? delta.y - norm.y : 0);
		}
		tsf::print("\n");
	}
	tsf::print("\n");
}

void Mesh::DrawFlowImage(std::string filename) const {
	DrawFlowImage(Rect32(0, 0, Width, Height), filename);
}

void Mesh::DrawFlowImage(gfx::Rect32 rect, std::string filename) const {
	int        centerx = (rect.x1 + rect.x2) / 2;
	int        centery = (rect.y1 + rect.y2) / 2;
	gfx::Vec2f norm    = At(centerx, centery).Pos - At(centerx, centery).UV;
	int        space   = 15;
	Canvas     c(space * (Width + 2), space * (Height + 2), Color8(0, 0, 0, 255));
	for (int y = rect.y1; y < rect.y2; y++) {
		for (int x = rect.x1; x < rect.x2; x++) {
			Vec2f center((x + 1) * space, (y + 1) * space);
			Vec2f d = At(x, y).Pos - At(x, y).UV - norm;
			d *= 1.5;
			//c.Line(center.x, center.y, center.x + d.x * 0.3, center.y + d.y * 0.3, Color8(150, 0, 0, 150), 2.0);
			//c.Line(center.x, center.y, center.x + d.x * 0.6, center.y + d.y * 0.6, Color8(150, 0, 0, 150), 1.5);
			auto cc = At(x, y).Color;
			cc.r    = cc.a;
			cc.b    = cc.a;
			cc.a    = 255;
			//cc.a    = max<uint8_t>(cc.a, 50);
			c.FillCircle(center.x, center.y, 1.1, cc);
			c.Line(center.x, center.y, center.x + d.x, center.y + d.y, cc, 1.0);
		}
	}
	c.GetImage()->SavePng(filename, true, 1);
}

} // namespace roadproc
} // namespace imqs