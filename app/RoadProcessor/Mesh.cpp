#include "pch.h"
#include "Mesh.h"

using namespace imqs::gfx;
using namespace std;

namespace imqs {
namespace roadproc {

Mesh::Mesh() {
}

Mesh::Mesh(const Mesh& m) {
	*this = m;
}

Mesh::Mesh(int width, int height) {
	Initialize(width, height);
}

Mesh::~Mesh() {
	delete[] Vertices;
}

Mesh& Mesh::operator=(const Mesh& m) {
	if (Width != m.Width || Height != m.Height)
		Initialize(m.Width, m.Height);
	memcpy(Vertices, m.Vertices, Count * sizeof(Vertices[0]));
	return *this;
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
void Mesh::ResetIdentityForWarpMesh(int imgWidth, int imgHeight, int matchRadius, bool moveBottomRowToBottomOfImage) {
	// This is for the typical case, where we're generating a warp mesh from a flattened camera frame.
	for (int y = 0; y < Height; y++) {
		for (int x = 0; x < Width; x++) {
			float xf = ((float) x / (float) (Width - 1));
			float yf = ((float) y / (float) (Height - 1));
			VX    vx;
			vx.Pos   = Vec2f(xf * (float) imgWidth, yf * (float) imgHeight);
			vx.UV    = Vec2f(xf * (float) imgWidth, yf * (float) imgHeight);
			At(x, y) = vx;
		}
	}

	// It is vital in this case that the row of vertices which are second from the bottom, are positioned
	// such that they but up against the bottom, precisely. The bottom-most row of vertices must be the
	// edge of the image, so we don't mess with that.
	// What we're doing here is taking the second row from the bottom, and squishing it downwards so that
	// it is exactly matchRadius away from the bottom of the image. These vertices are the most important
	// vertices for optical flow, because they lie in the highest resolution area of the camera, where
	// we care about the image the most.
	// SEE COMMENT ABOVE FUNCTION
	if (moveBottomRowToBottomOfImage) {
		int y = Height - 2;
		for (int x = 0; x < Width; x++) {
			auto& p = At(x, y);
			p.Pos.y = imgHeight - matchRadius;
			p.UV.y  = imgHeight - matchRadius;
		}
	}
}

void Mesh::TransformTargets(gfx::Vec2f translate) {
	for (int y = 0; y < Height; y++) {
		for (int x = 0; x < Width; x++) {
			At(x, y).Pos += translate;
		}
	}
}

static float CompactMeshMaxDelta = 1000.0f;
static float CompactMeshScale    = 32766.0f / CompactMeshMaxDelta;
static float CompactMeshScaleInv = CompactMeshMaxDelta / 32766.0f;

struct CompactMeshHead {
	gfx::Vec2f Bias;
};
static_assert(sizeof(CompactMeshHead) == 8, "CompactMeshHead size");

// We compress our mesh here, so that a typical mesh takes 30 KB instead of 117 KB
// We could compress the color further, to perhaps just alpha, and we could quite
// likely also compress the deltas to int8, with quarter pixel precision.. -64..+64 range
// should be fine for any correct scene.
Error Mesh::SaveCompact(std::string filename) {
	Vec2f     bias  = At(0, 0).Pos - At(0, 0).UV;
	int16_t*  pos   = (int16_t*) malloc(Count * 2 * sizeof(int16_t));
	uint32_t* color = (uint32_t*) malloc(Count * sizeof(uint32_t));
	for (size_t i = 0; i < Count; i++) {
		const auto& v      = Vertices[i];
		Vec2f       scaled = CompactMeshScale * (v.Pos - v.UV - bias);
		if (fabs(scaled.x) > 32767 || fabs(scaled.y) > 32767)
			return Error("Mesh offset exceeds int16 limits. The mesh is probably ill-formed");
		pos[i * 2]     = (int16_t) scaled.x;
		pos[i * 2 + 1] = (int16_t) scaled.y;
		color[i]       = v.Color.u;
	}
	os::File f;
	auto     err = f.Create(filename);
	if (err.OK()) {
		CompactMeshHead head;
		head.Bias = bias;

		err = f.Write(&head, sizeof(head));
		err |= f.Write(pos, Count * 2 * sizeof(int16_t));
		err |= f.Write(color, Count * sizeof(uint32_t));
	}
	free(pos);
	free(color);
	f.Close();
	return err;
}

Error Mesh::LoadCompact(std::string filename) {
	os::File f;
	auto     err = f.Open(filename);
	if (!err.OK())
		return err;

	CompactMeshHead head;
	err = f.ReadExactly(&head, sizeof(head));
	if (!err.OK())
		return err;

	int16_t*  pos   = (int16_t*) malloc(Count * 2 * sizeof(int16_t));
	uint32_t* color = (uint32_t*) malloc(Count * sizeof(uint32_t));

	err = f.ReadExactly(pos, Count * 2 * sizeof(int16_t));
	if (err.OK())
		err = f.ReadExactly(color, Count * sizeof(uint32_t));

	if (err.OK()) {
		for (size_t i = 0; i < Count; i++) {
			auto& v   = Vertices[i];
			float x   = (float) pos[i * 2] * CompactMeshScaleInv;
			float y   = (float) pos[i * 2 + 1] * CompactMeshScaleInv;
			v.Pos     = v.UV + head.Bias + Vec2f(x, y);
			v.Color.u = color[i];
		}
	}
	free(pos);
	free(color);
	return Error();
}

gfx::Vec2f Mesh::AvgValidDisplacement() const {
	Vec2d  avg(0, 0);
	double n = 0;
	for (int i = 0; i < Count; i++) {
		if (Vertices[i].IsValid) {
			n++;
			avg += Vec2fTod(Vertices[i].Pos - Vertices[i].UV);
		}
	}
	avg = (1.0 / n) * avg;
	return Vec2dTof(avg);
}

Vec2f Mesh::PosAtFractionalUV(float u, float v) const {
	// assume UV is mostly uniform
	Vec2f interval = At(Width - 1, Height - 1).UV - At(0, 0).UV;
	interval /= Vec2f(Width - 1, Height - 1);
	int   uI = (int) floor(u / interval.x);
	int   vI = (int) floor(v / interval.y);
	float uF = fmod(u, interval.x) / interval.x;
	float vF = fmod(v, interval.y) / interval.y;
	if (uI < 0) {
		uI = 0;
		uF = 0;
	} else if (uI >= Width - 1) {
		uI = Width - 2;
		uF = 1;
	}
	if (vI < 0) {
		vI = 0;
		vF = 0;
	} else if (vI >= Height - 1) {
		vI = Height - 2;
		vF = 1;
	}
	auto a = At(uI, vI).Pos;
	auto b = At(uI + 1, vI + 1).Pos;
	return a + Vec2f(uF, vF) * (b - a);
}

bool Mesh::FirstValid(int& x, int& y) const {
	for (int my = 0; my < Height; my++) {
		for (int mx = 0; mx < Width; mx++) {
			if (At(mx, my).IsValid) {
				x = mx;
				y = my;
				return true;
			}
		}
	}
	return false;
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

void Mesh::PrintDeltaPos(gfx::Rect32 rect, gfx::Vec2f norm) const {
	if (norm == gfx::Vec2f(FLT_MAX, FLT_MAX))
		norm = At(rect.x1, rect.y1).Pos - At(rect.x1, rect.y1).UV;
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

/*
void Mesh::PrintDeltaStrength(gfx::Rect32 rect) const {
	tsf::print("Delta Strength:\n");
	for (int y = rect.y1; y < rect.y2; y++) {
		for (int x = rect.x1; x < rect.x2; x++) {
			tsf::print("%4.2f ", At(x, y).DeltaStrength);
		}
		tsf::print("\n");
	}
	tsf::print("\n");
}
*/

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