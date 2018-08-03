#include "pch.h"
#include "FeatureTracking.h"
#include "Globals.h"
#include "Perspective.h"
#include "OpticalFlow.h"

// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 5 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.0009
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 10 --start 5 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000879688

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

struct UV32 {
	int32_t U;
	int32_t V;
};

class Stitcher {
public:
	gfx::Vec2d StitchTopLeft;
	double     StitchAngle        = 0;
	int        StitchWindowWidth  = 0;
	int        StitchWindowHeight = 300;
	Image      Giant;

	//static const int MeshShift = 2; // 2 bits of sub-pixel precision should be enough
	//static const int MeshMul   = (1 << MeshShift);
	static const int MeshBias = 400 * 256;

	void BlendNewFrame(OpticalFlow& flow, Image& newFrame) {
		int ax = (int) StitchTopLeft.x;
		int ay = (int) StitchTopLeft.y;

		//newFrame.SaveJpeg("newFrame.jpeg");

		// compute a dense-ish warp mesh from the optical flow's output grid.
		// this dense-ish warp mesh is regular, so that we can read from it with reasonable speed,
		// and a bilinear filter
		const int warpSizeShift = 7;
		const int warpSize      = 1 << warpSizeShift;

		int meshW = 1 + (StitchWindowWidth + warpSize - 1) / warpSize;
		int meshH = 1 + (StitchWindowHeight * 2 + warpSize - 1) / warpSize; // we also add new footage "on top", which will become the stitch window for the next frame

		int meshOrgX = 0;
		int meshOrgY = -StitchWindowHeight;

		Vec2f flowGridSpacing;
		flowGridSpacing.x = (flow.GridCenterAt(flow.GridW - 1, 0).x - flow.GridCenterAt(0, 0).x) / (flow.GridW - 1);
		flowGridSpacing.y = (flow.GridCenterAt(0, flow.GridH - 1).y - flow.GridCenterAt(0, 0).y) / (flow.GridH - 1);
		Vec2f flowGridOrg;
		flowGridOrg.x = flow.GridCenterAt(0, 0).x;
		flowGridOrg.y = flow.GridCenterAt(0, 0).y;

		// hmm.. it would be good to get rid of this intermediate step. We could do so, by forcing the flow system
		// to produce a ready-to-use mesh. The real tedium here is caused by having to do the bilinear filtering
		// on the flow mesh.
		UV32*  mesh  = (UV32*) malloc(meshW * meshH * sizeof(UV32));
		Vec2f* mtest = new Vec2f[meshW * meshH];

		// populate the bottom half of the mesh with the flow mesh
		for (int x = 0; x < meshW; x++) {
			for (int y = 0; y < meshH; y++) {
				int m = y * meshW + x;
				// get these mesh coordinates into flow mesh coordinates
				int px = x * warpSize;
				int py = y * warpSize + meshOrgY;
				//if (py < -warpSize) {
				//	mesh[m].U  = 0;
				//	mesh[m].V  = 0;
				//	mtest[m].x = 0;
				//	mtest[m].y = 0;
				//	continue;
				//}
				float fgx = ((float) px - flowGridOrg.x) / flowGridSpacing.x;
				float fgy = ((float) py - flowGridOrg.y) / flowGridSpacing.y;
				// round down to integer
				int fgxi = (int) floor(fgx);
				int fgyi = (int) floor(fgy);
				// remainder
				float fgxr = fgx - fgxi;
				float fgyr = fgy - fgyi;
				if (fgxi < 0) {
					fgxi = 0;
					fgxr = 0;
				}
				if (fgyi < 0) {
					fgyi = 0;
					fgyr = 0;
				}
				if (fgxi > flow.GridW - 2) {
					fgxi = flow.GridW - 2;
					fgxr = 1;
				}
				if (fgyi > flow.GridH - 2) {
					fgyi = flow.GridH - 2;
					fgyr = 1;
				}
				Vec2f a = Vec2dTof(flow.LastGridEl(fgxi, fgyi));
				Vec2f b = Vec2dTof(flow.LastGridEl(fgxi + 1, fgyi));
				Vec2f c = Vec2dTof(flow.LastGridEl(fgxi, fgyi + 1));
				Vec2f d = Vec2dTof(flow.LastGridEl(fgxi + 1, fgyi + 1));
				a += fgxr * (b - a);
				c += fgxr * (d - c);
				a += fgyr * (c - a);
				mesh[m].U = (int32_t)(a.x * 256) + MeshBias;
				mesh[m].V = (int32_t)(a.y * 256) + MeshBias;
				mtest[m]  = a;
				//if (py < -warpSize) {
				//	mesh[m].U = 0;
				//	mesh[m].V  = 0;
				//	mtest[m].x = 0;
				//	mtest[m].y = 0;
				//	continue;
				//}
			}
		}

		// average the vertical displacements of the fresh bar (the upper half of the new imagery),
		// and zero out the horizontal displacement
		for (int y = 0; y < meshH; y++) {
			int py = y * warpSize;
			if (py >= StitchWindowHeight - warpSize)
				break;
			int sumV = 0;
			for (int x = 0; x < meshW; x++)
				sumV += mesh[y * meshW + x].V;
			sumV /= meshW;
			for (int x = 0; x < meshW; x++) {
				mesh[y * meshW + x].U = MeshBias;
				mesh[y * meshW + x].V = sumV;
			}
		}

		bool printWarpMesh = false;
		if (printWarpMesh) {
			tsf::print("\n");
			for (int y = 0; y < meshH; y++) {
				for (int x = 0; x < meshW; x++) {
					tsf::print("%4d ", (mesh[y * meshW + x].U - MeshBias) >> 8);
					//tsf::print("%3.0f ", mtest[y * meshW + x].x);
				}
				tsf::print(" | ");
				for (int x = 0; x < meshW; x++) {
					tsf::print("%4d ", (mesh[y * meshW + x].V - MeshBias) >> 8);
					//tsf::print("%4.0f ", mtest[y * meshW + x].y);
				}
				tsf::print("\n");
			}
		}

		int avgForwardMotion = (mesh[0].V - MeshBias) >> 8;

		int dstY = ay - StitchWindowHeight;
		for (int y = 0; y < StitchWindowHeight; y++) {
			int      w              = StitchWindowWidth;
			int      warpGridClampU = ((meshW - 1) * 256) - 1;
			int      warpGridClampV = ((meshH - 1) * 256) - 1;
			int      imgClampU      = ((newFrame.Width - 1) * 256) - 1;
			int      imgClampV      = ((newFrame.Height - 1) * 256) - 1;
			uint8_t* dstP           = Giant.At(ax, dstY);
			for (int x = 0; x < w; x++) {
				int      u    = x << 8;
				int      v    = y << 8;
				uint64_t warp = raster::ImageBilinear_RG_U24(mesh, meshW, warpGridClampU, warpGridClampV, u >> warpSizeShift, v >> warpSizeShift);
				v -= StitchWindowHeight * 256;
				int32_t warpU = (warp & 0xffffffff) - MeshBias;
				int32_t warpV = (warp >> 32) - MeshBias;
				u += warpU;
				v += warpV;
				uint32_t col        = raster::ImageBilinearRGBA(newFrame.Data, newFrame.Stride >> 2, imgClampU, imgClampV, u, v);
				*((uint32_t*) dstP) = col;
				dstP += 4;
			}
			dstY++;
		}

		StitchTopLeft.y = StitchTopLeft.y + StitchWindowHeight - avgForwardMotion;

		delete[] mtest;
		free(mesh);
	}

	Error DoStitch(string videoFile, float zx, float zy, double seconds, int count) {
		video::VideoFile video;
		auto             err = video.OpenFile(videoFile);
		if (!err.OK())
			return err;

		err = video.SeekToSecond(seconds, video::Seek::Any);
		if (!err.OK())
			return err;

		if (global::Lens != nullptr) {
			err = global::Lens->InitializeDistortionCorrect(video.Width(), video.Height());
			if (!err.OK())
				return err;
		}

		float z1 = FindZ1ForIdentityScaleAtBottom(video.Width(), video.Height(), zx, zy);
		auto  f  = ComputeFrustum(video.Width(), video.Height(), z1, zx, zy);
		Image flat, flatPrev;
		flat.Alloc(gfx::ImageFormat::RGBA, f.Width, f.Height);
		flatPrev.Alloc(gfx::ImageFormat::RGBA, f.Width, f.Height);
		flat.Fill(0);
		flatPrev.Fill(0);

		auto flatOrigin = CameraToFlat(video.Width(), video.Height(), Vec2f(0, 0), z1, zx, zy);

		Image img;
		img.Alloc(gfx::ImageFormat::RGBA, video.Width(), video.Height());

		Giant.Alloc(gfx::ImageFormat::RGBA, 1940, 6000);
		Giant.Fill(Color8(255, 255, 255, 255).u);

		OpticalFlow flow;

		for (int i = 0; i < count; i++) {
			err = video.DecodeFrameRGBA(img.Width, img.Height, img.Data, img.Stride);
			if (err == ErrEOF)
				break;
			if (!err.OK())
				return err;

			RemovePerspective(img, flat, z1, zx, zy, (int) flatOrigin.x, (int) flatOrigin.y);

			//flat.SaveJpeg(tsf::fmt("flat%d.jpeg", i));

			if (i == 0) {
				StitchWindowWidth = f.X2 - f.X1 - 2; // the -2 is a buffer
				int    windowLeft = (flat.Width - StitchWindowWidth) / 2;
				int    windowTop  = flat.Height - StitchWindowHeight * 2;
				Rect32 srcRect(windowLeft, windowTop, windowLeft + StitchWindowWidth, windowTop + StitchWindowHeight * 2);
				int    dstX = (Giant.Width - StitchWindowWidth) / 2;
				int    dstY = Giant.Height - StitchWindowHeight * 2;
				Giant.CopyFrom(flat, srcRect, dstX, dstY);
				//Giant.SaveJpeg("giant.jpeg");
				StitchTopLeft = Vec2d(dstX, dstY);
			}
			if (i != 0) {
				int  ax               = (int) StitchTopLeft.x;
				int  ay               = (int) StitchTopLeft.y;
				auto alignWindowSrc   = Giant.Window(ax, ay, StitchWindowWidth, StitchWindowHeight);
				int  dstLeft          = (flat.Width - StitchWindowWidth) / 2;
				auto alignWindowDst   = flat.Window(dstLeft, flat.Height - StitchWindowHeight * 3, StitchWindowWidth, StitchWindowHeight * 2.6);
				auto alignWindowBlend = flat.Window(dstLeft, flat.Height - StitchWindowHeight * 3, StitchWindowWidth + 10, StitchWindowHeight * 2.8);
				flow.FirstFrameBiasH  = 0;
				flow.FirstFrameBiasV  = StitchWindowHeight;
				//StitchFrames(flow, i, f, alignWindow, flat);
				//StitchFrames(flow, i, f, flatPrev, flat, giant);
				// alignWindow will have to be rotated by StitchAngle in future... once we support rotation
				flow.Frame(alignWindowSrc, alignWindowDst);
				//Giant.SaveJpeg("giant1.jpeg");
				BlendNewFrame(flow, alignWindowBlend);
				//Giant.SaveJpeg("giant2.jpeg");
			}

			if (false) {
				err = flat.SaveFile(tsf::fmt("flat-%04d.jpeg", i));
				if (!err.OK())
					return err;
			}
			//tsf::print("%v/%v\r", i + 1, count);
			fflush(stdout);
			std::swap(flatPrev, flat);
		}
		Giant.SaveJpeg("giant2.jpeg");
		//tsf::print("\n");
		return Error();
	}

	// old frame-by-frame code, might still use for perspective (zy) computation
	void StitchFrames(OpticalFlow& flow, int frameNumber, Frustum f, Image& img1, Image& img2) {
		int windowWidth  = f.X2 - f.X1 - 2; // the -2 is a buffer
		int windowHeight = 900;
		int windowTop    = img1.Height - windowHeight;
		int windowLeft   = (img1.Width - windowWidth) / 2;

		auto img1Crop = img1.Window(windowLeft, windowTop, windowWidth, windowHeight);
		auto img2Crop = img2.Window(windowLeft, windowTop, windowWidth, windowHeight);

		flow.Frame(img1Crop, img2Crop);
	}
};

int Stitch(argparse::Args& args) {
	auto     videoFile = args.Params[0];
	float    zx        = atof(args.Params[1].c_str());
	float    zy        = atof(args.Params[2].c_str());
	int      count     = args.GetInt("number");
	double   seek      = atof(args.Get("start").c_str());
	Stitcher s;
	auto     err = s.DoStitch(videoFile, zx, zy, seek, count);
	if (!err.OK()) {
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

/*
keypoint stitching (instead of optical flow)
	if (false) {
		cv::Mat m1 = ImageToMat(img1Crop);
		cv::Mat m2 = ImageToMat(img2Crop);
		cv::Mat mg1, mg2;
		cv::cvtColor(m1, mg1, cv::COLOR_RGB2GRAY);
		cv::cvtColor(m2, mg2, cv::COLOR_RGB2GRAY);
		int                maxPoints   = 10000;
		double             quality     = 0.01;
		int                minDistance = 5;
		KeyPointSet        kp1, kp2;
		vector<cv::DMatch> matches;
		ComputeKeyPointsAndMatch("FREAK", mg1, mg2, maxPoints, quality, minDistance, false, false, kp1, kp2, matches);
		cv::Mat matchImg;
		cv::drawMatches(m1, kp1.Points, m2, kp2.Points, matches, matchImg);
		auto diag = MatToImage(matchImg);
		diag.SavePng("match.png");
	}

*/

} // namespace roadproc
} // namespace imqs