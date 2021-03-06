#include "pch.h"
#include "Stitcher.h"
#include "FeatureTracking.h"
#include "Globals.h"
#include "Perspective.h"
#include "OpticalFlow.h"
#include "Mesh.h"

// Time to measure meters/pixel is about 0m20

// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 1 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 40 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 40 --start 0.7 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 30 --start 260 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999

// second video
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 30 --start 0 ~/mldata/DSCF3040.MOV ~/pos.json 0 -0.00095
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 200 --start 14 ~/mldata/DSCF3040.MOV ~/pos.json 0 -0.00095
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -m 0.003365 ~/mldata/DSCF3040.MOV ~/inf2 ~/dev/Annotator/pos.json 0 -0.00095

// mthata
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' measure-scale ~/mldata/mthata/DSCF0001-HG-3.MOV mthata-pos.json 0 -0.000411
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -s 20 ~/mldata/mthata/DSCF0001-HG-3.MOV ~/inf mthata-pos.json 0 -0.000411
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch ~/mldata/mthata/DSCF0001-HG-3.MOV ~/inf mthata-pos.json 0 -0.000411
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -m .0024 /home/ben/mldata/mthata/DSCF0001-HG-3.MOV,/home/ben/mldata/mthata/DSCF0001-HG-4.MOV,/home/ben/mldata/mthata/DSCF0001-HG-5.MOV ~/inf mthata-pos.json 0 -0.000411
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -m .0024 /home/ben/mldata/mthata/DSCF0001-HG-3.MOV ~/inf mthata-pos.json 0 -0.000411

// build/run-roadprocessor -r webtiles ~/inf

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

static StaticError ErrGeoVelocityZero("GPS position has no velocity");

Stitcher::Stitcher() {
	//ClearColor = Color8(0, 150, 0, 60);
	ClearColor = Color8(0, 0, 0, 0);
	for (int i = 0; i < NVignette; i++)
		Vignetting[i] = 1;
}

Error Stitcher::Initialize(std::string storageSpec, std::vector<std::string> videoFiles, FlattenParams fp, double seconds) {
	if (storageSpec != "")
		InfBmp.Initialize(storageSpec);

	VidStitcher.BlackenPercentage    = 0.15;
	VidStitcher.EnableFullFlatOutput = true;
	VidStitcher.StartVideoAt         = seconds;
	auto err                         = VidStitcher.Start(videoFiles, fp);
	if (!err.OK())
		return err;

	//err = Rend.Initialize(4096, 3072);
	//err = Rend.Initialize(5120, 4096);
	//err = Rend.Initialize(5120, 6144);
	//err = Rend.Initialize(6144, 6144);
	//err = Rend.Initialize(7168, 7168);
	//err = Rend.Initialize(7168, 4096);
	err = Rend.Initialize(8192, 8192);
	if (!err.OK())
		return err;
	IMQS_ASSERT(Rend.FBWidth % InfBmp.TileSize == 0);
	IMQS_ASSERT(Rend.FBHeight % InfBmp.TileSize == 0);
	InfBmpView = Rect64(0, 0, Rend.FBWidth, Rend.FBHeight);
	Rend.Clear(ClearColor);

	PrevDir         = Vec2f(0, -1);
	PrevGeoFramePos = Vec3d(0, 0, 0);

	return Error();
}

Error Stitcher::LoadTrack(std::string trackFile) {
	HavePositions = true;
	auto err      = Track.LoadFile(trackFile);
	if (!err.OK())
		return err;
	Track.ConvertToWebMercator();
	//Track.Simplify(0.001);
	Track.Smooth(0.5, 0.1);
	//Track.DumpRaw(0, 6);
	//Track.SaveCSV("/home/ben/tracks.csv");
	//exit(1);
	return Error();
}

Error Stitcher::DoMeasureScale(std::vector<std::string> videoFiles, std::string trackFile, FlattenParams fp) {
	auto err = LoadTrack(trackFile);
	if (!err.OK())
		return err;
	err = Initialize("", videoFiles, fp, 0);
	if (!err.OK())
		return err;

	//MetersPerPixel = 0.0033; // dev time

	if (MetersPerPixel == 0) {
		err = MeasurePixelScale();
		if (!err.OK())
			return err;
	}

	// This is our only output
	tsf::print("%.6f\n", MetersPerPixel);

	return Error();
}

Error Stitcher::DoStitch(std::string storageSpec, std::vector<std::string> videoFiles, std::string trackFile, FlattenParams fp, double seconds, int count) {
	//os::RemoveAll(bitmapDir);
	//os::MkDirAll(bitmapDir);

	auto err = LoadTrack(trackFile);

	err = Initialize(storageSpec, videoFiles, fp, seconds);
	if (!err.OK())
		return err;

	//MetersPerPixel = 0.0033; // dev time

	if (MetersPerPixel == 0) {
		tsf::print("Measuring meters/pixel\n");
		err = MeasurePixelScale();
		if (!err.OK())
			return err;
		tsf::print("meters/pixel: %.6f\n", MetersPerPixel);
	}

	SetupBaseMapScale();

	EnableSimpleRender = false;
	EnableGeoRender    = true;
	InfBmpView         = Rect64(0, 0, 0, 0);
	err                = Run(count);
	if (!err.OK())
		return err;

	//switch (phase) {
	//case Phases::InitialStitch: return DoStitchInitial(count);
	//case Phases::GeoReference: return DoGeoReference(count);
	//}
	// unreachable
	return Error();
}

// Stitch a bunch of frames, and with each stitch, compare our GPS velocity to our pixel velocity.
// Then, use that to generate a scale for our pixels, and thereafter use that scale for the entire video.
Error Stitcher::MeasurePixelScale() {
	EnableSimpleRender = false;
	EnableGeoRender    = false;

	bool debug = false;

	size_t        minSamples = 200;
	vector<float> metersPerPixelSamples;
	if (debug)
		tsf::print("\n");

	// In the first few seconds of a recording, the vehicle is not moving
	auto orgStart            = VidStitcher.StartVideoAt;
	VidStitcher.StartVideoAt = 20;
	auto err                 = VidStitcher.Rewind();
	if (!err.OK())
		return err;

	for (int i = 0; metersPerPixelSamples.size() < minSamples; i++) {
		err = VidStitcher.Next();
		if (!err.OK())
			return err;

		Vec3d pos;
		Vec2d vel2D;
		Track.GetPositionAndVelocity(VidStitcher.FrameTime, pos, vel2D);
		if (vel2D.size() < 5) {
			VidStitcher.StartVideoAt = VidStitcher.FrameTime + 5;
			VidStitcher.Rewind();
			if (debug)
				tsf::print("\nSeeking to %v\n", VidStitcher.StartVideoAt);
			continue;
		}
		//tsf::print("%v: vel2D: %v, disp: %v\n", VidStitcher.FrameTime, vel2D.size(), PrevDirUnfiltered.size());

		err = StitchFrame();
		if (!err.OK())
			return err;

		float pixelsPerFrame  = PrevDirUnfiltered.size();
		float pixelsPerSecond = pixelsPerFrame * VidStitcher.Video.GetVideoStreamInfo().FrameRateSeconds();
		metersPerPixelSamples.push_back((float) vel2D.size() / pixelsPerSecond);

		if (i % 10 == 0) {
			auto mv     = math::MeanAndVariance<float, double>(metersPerPixelSamples);
			auto median = math::Median(metersPerPixelSamples);
			if (debug) {
				tsf::print("MetersPerPixel mean: %v, median: %v, sd: %v (%v/%v samples)\r", mv.first, median, sqrt(mv.second), metersPerPixelSamples.size(), minSamples);
				fflush(stdout);
			}
		}
	}

	if (metersPerPixelSamples.size() < minSamples)
		return Error::Fmt("Unable to measure meters/pixel. Too few fast-moving samples (%v). Need at least %v", metersPerPixelSamples.size(), minSamples);

	VidStitcher.StartVideoAt = orgStart;
	VidStitcher.Rewind();

	MetersPerPixel = math::Median(metersPerPixelSamples);

	return Error();
}

namespace tiles {
static const double TileSize          = 256;
static const double InitialResolution = 2 * IMQS_PI * 6378137 / TileSize; // 156543.03392804062 for TileSize 256 pixels, in meters/pixel
static const double OriginShift       = 2 * IMQS_PI * 6378137 / 2.0;      // 20037508.342789244
} // namespace tiles

void Stitcher::SetupBaseMapScale() {
	BaseZoomLevel = (int) floor(log2(tiles::InitialResolution / (double) MetersPerPixel));
}

double Stitcher::BaseMapMetersPerPixel() {
	return tiles::InitialResolution / (double) (1 << BaseZoomLevel);
}

Error Stitcher::Run(int count) {
	auto err = VidStitcher.Rewind();
	if (!err.OK())
		return err;

	for (int i = 0; i < count || count == -1; i++) {
		err = VidStitcher.Next();
		if (err == ErrEOF)
			break;
		else if (!err.OK())
			return err;

		// For now we just use lensfun to do vignetting correction
		//MeasureVignetting();
		//Track.DumpRaw(0, 6);

		err = StitchFrame();
		if (!err.OK())
			return err;

		if ((EnableSimpleRender || EnableGeoRender) && (i % 100 == 0 || (i < 20 && i % 5 == 0)))
			Rend.SaveToFile("giant2.jpeg");
		//Rend.SaveToFile("giant2.jpeg");

		//if (VidStitcher.FrameNumber != 0) {
		//	err = VidStitcher.Mesh.SaveCompact(path::Join(TempDir, "mesh", tsf::fmt("%08d", VidStitcher.FrameNumber)));
		//	if (!err.OK())
		//		return err;
		//}

		//tsf::print("%v\n", VidStitcher.FrameNumber);
		//VidStitcher.PrintRemainingTime();

		if (EnableSimpleRender)
			AdjustInfiniteBitmapView(PrevFullMesh, PrevDir);
	}
	return Error();
}

static float LineLuminance(const Image& img, int x1, int x2, int y) {
	const Color8* c   = (const Color8*) img.At(x1, y);
	int           lum = 0;
	for (int x = x1; x < x2; x++) {
		lum += c->Lum();
		c++;
	}
	return (float) lum / (float) (x2 - x1);
}

void Stitcher::MeasureVignetting() {
	int xleft           = (int) VidStitcher.Frustum.X1FullFrame() + 20;
	int xright          = (int) VidStitcher.Frustum.X2FullFrame() - 20;
	int xloc[NVignette] = {xleft, (xleft + xright) / 2, xright};

	int h  = 50;
	int y1 = VidStitcher.FullFlat.Height - h - 1;
	int y2 = VidStitcher.FullFlat.Height;
	for (int loc = 0; loc < NVignette; loc++) {
		int   x1     = xloc[loc] - 20;
		int   x2     = xloc[loc] + 20;
		float l1     = LineLuminance(VidStitcher.FullFlat, x1, x2, y1);
		float l2     = LineLuminance(VidStitcher.FullFlat, x1, x2, y2);
		float adjust = l2 / l1;
		Vignetting[loc] += 0.5f * (adjust - Vignetting[loc]);
	}
	tsf::print("%5.2f %5.2f %5.2f\n", Vignetting[0], Vignetting[1], Vignetting[2]);
}

Error Stitcher::StitchFrame() {
	// The alignment mesh is produced on a crop of the full flattened frame.
	// Our job now is to turn that little alignment mesh into a full mesh that covers the entire
	// flattened frame.
	// In addition, we need to shift the coordinate frame from the previous flattened frame,
	// to the coordinate frame of our 8192x8192 composite "splat" image.
	auto cropRect = VidStitcher.CropRectFromFullFlat();

	Mesh  full;
	Vec2f uvAtTopLeftOfImage;
	Vec2f debugTopLeft;
	if (VidStitcher.FrameNumber == 0) {
		Vec2f topLeft((Rend.FBWidth - VidStitcher.FullFlat.Width) / 2, Rend.FBHeight - VidStitcher.FullFlat.Height);
		Vec2f topRight = topLeft + Vec2f(VidStitcher.FullFlat.Width, 0);
		Vec2f botLeft  = topLeft + Vec2f(0, VidStitcher.FullFlat.Height);
		full.Initialize(60, 30);
		full.ResetUniformRectangular(topLeft, topRight, botLeft, VidStitcher.FullFlat.Width, VidStitcher.FullFlat.Height);
		uvAtTopLeftOfImage = Vec2f(0, 0);
		debugTopLeft       = topLeft;
	} else {
		ExtrapolateMesh(VidStitcher.Mesh, full, uvAtTopLeftOfImage);
		TransformMeshIntoRendCoords(full);
	}

	if (EnableSimpleRender)
		Rend.DrawMesh(full, VidStitcher.FullFlat);

	if (EnableGeoRender) {
		FrameObject f;
		f.AvgDisp   = VidStitcher.FrameNumber == 0 ? Vec2f(0, 0) : VidStitcher.Mesh.AvgValidDisplacement();
		f.FrameTime = VidStitcher.FrameTime;
		f.Img       = VidStitcher.FullFlat;
		f.Mesh      = full;
		Frames.push_back(std::move(f));

		if (Frames.size() == 2) {
			auto err = DrawGeoReferencedFrame();
			if (!err.OK())
				return err;
			Frames.erase(Frames.begin());
		}
	}

	{
		// These parameters are not used for GeoReferenced rendering.
		// This is used for "EnableSimpleRender", and also for the meters/pixel scale measurement
		Vec2f newTopLeft = full.PosAtFractionalUV(uvAtTopLeftOfImage);
		Vec2f dir        = newTopLeft - PrevTopLeft;
		if (dir.size() > 5 && VidStitcher.FrameNumber >= 1) {
			// only update direction if we're actually moving
			PrevDir = dir;
		}
		PrevDirUnfiltered = dir;
		PrevTopLeft       = newTopLeft;
		PrevFullMesh      = full;
	}

	return Error();
}

Error Stitcher::DrawGeoReferencedFrame() {
	Vec3d geoPos;
	Vec2d vel2D;
	Track.GetPositionAndVelocity(Frames[0].FrameTime, geoPos, vel2D);
	//tsf::print("%.2f: Velocity: %.3f m/s. Distance: %.1f m (img AvgDisp: %.1f %.1f px)\n", VidStitcher.FrameTime, vel2D.size(), geoPos.distance2D(PrevGeoFramePos), Frames[1].AvgDisp.x, Frames[1].AvgDisp.y);

	// We just produce noise if we output frames when standing still, so when we come to a stop, then we cease outputting frames
	if (geoPos.distance2D(PrevGeoFramePos) > 1.0 || vel2D.size() > 4.0) {
		Vec3d geoOffset;
		auto  err = TransformFrameCoordsToGeo(geoOffset);
		if (err == ErrGeoVelocityZero) {
			// we have to skip this frame. We have zero velocity, so it's pointless continuing stitching,
			// and additionally, we don't know our direction of travel. I first saw this at the start of
			// a recording.
			return Error();
		} else if (!err.OK()) {
			return err;
		}
		err             = DrawGeoMesh(geoOffset);
		PrevGeoFramePos = geoPos;
		return err;
	}
	return Error();
}

Error Stitcher::TransformFrameCoordsToGeo(gfx::Vec3d& geoOffset) {
	// Firstly, imagine the (flattened) frame on your screen. X is pointing right, and Y is pointing
	// down, as is usual for images.
	// The bottom of the image is T0, the FrameTime at which the frame was captured. Now, imagine
	// a line that starts on the bottom of the image, and moves upwards, until it reaches the place
	// where the *next* frame blends on to this frame. That line (which is always close to horizontal),
	// is the line where time is T1. This is the FrameTime of the next frame. We work on the center
	// of the image first, and we compute a simple linear function that interpolates time from T0 to
	// T1. What happens beyond T1, we're not that much concerned with, because it's going to get
	// overwritten by the *next next* frame. However, we do still draw the entire frame, so we do need
	// *some* value there. What we do, is we simply extrapolate out beyond T1.
	//
	// NOTE: We might want to run this process with a history of more than just two frames, so that
	// we don't do as much extrapolation. This might improve the quality of the periphery of the images.
	//
	// So, to recap, we have defined a line that starts at the bottom, center, of this frame, and it
	// extends up to the place where the next frame joins onto this one. Along this line, we interpolate
	// time from T0 to T1, and beyond that joining position, we extrapolate.
	// Now that we have time, we look at our GPS track, and determine the position at that time.
	// Once we have the position along the center line, we're almost done. The only thing that remains,
	// is to compute the positions along the horizontal span of the road (ie everything except the
	// center line). In order to do this, we compute the normal of the GPS track (aka the velocity),
	// and we extend out perpendicularly, left and right, from the GPS track. If the GPS track is
	// going through a bend, then this computation naturally produces a bend in the transformed
	// coordinates, "pinching" the inner part of the bend.

	FrameObject& f0 = Frames[0];
	FrameObject& f1 = Frames[1];

	float  uCenter = (float) f0.Img.Width / 2;
	Vec2f  center0(uCenter, f0.Img.Height);
	Vec2f  center1       = center0 + f1.AvgDisp;
	Vec2f  centerLine[2] = {center0, center1};
	double t0            = f0.FrameTime;
	double t1            = f1.FrameTime;
	geoOffset            = Vec3d(0, 0, 0);

	for (int my = 0; my < f0.Mesh.Height; my++) {
		for (int mx = 0; mx < f0.Mesh.Width; mx++) {
			auto& vx = f0.Mesh.At(mx, my);
			//Vec2f snapped = vx.UV;
			float snapMu = 0;
			//float  pixelDistanceFromCenter = geom::SnapPointToLine(false, 2, centerLine, snapped, snapMu);
			Vec2f snapped = geom::ClosestPtOnLineT(vx.UV, center0, center1, false, &snapMu);
			// This clamp here is necessary for cases where the car is basically standing still. If we don't clamp this,
			// then we can end up producing a gigantic mesh, that is too large to render into our framebuffer.
			snapMu                         = math::Clamp<float>(snapMu, -5, 100);
			float  pixelDistanceFromCenter = snapped.distance(vx.UV);
			double sideOfCenter            = math::SignOrZero(geom2d::SideOfLine(center0.x, center0.y, center1.x, center1.y, vx.UV.x, vx.UV.y));
			sideOfCenter                   = -sideOfCenter;
			double ptime                   = t0 + snapMu * (t1 - t0);
			Vec3d  geoCenterPos;
			Vec2d  geoVel;
			Track.GetPositionAndVelocity(ptime, geoCenterPos, geoVel);
			if (geoVel.size() == 0)
				return ErrGeoVelocityZero;
			if (geoVel.size() > 100)
				return Error::Fmt("Speed is too fast (%v km/h). FrameTime: %.2f. Pos: (%.1f,%.1f)", geoVel.size() * (1000 / 3600.0), f0.FrameTime, geoCenterPos.x, geoCenterPos.y);
			Vec2d  right                   = Vec2d(geoVel.y, -geoVel.x).normalized();
			double meterDistanceFromCenter = pixelDistanceFromCenter * MetersPerPixel;
			Vec3d  geoPos                  = geoCenterPos + sideOfCenter * meterDistanceFromCenter * Vec3d(right.x, right.y, 0);
			if (mx == 0 && my == 0)
				geoOffset = geoPos;
			auto geoPosOrg = geoPos;
			geoPos -= geoOffset;
			vx.Pos = Vec2f(geoPos.x, geoPos.y);
			//if (my == f0.Mesh.Height - 5)
			//	tsf::print("%.1f %.1f (%f) (%f %f)\n", geoPos.x, geoPos.y, sideOfCenter, right.x, right.y);
		}
	}
	return Error();
}

Error Stitcher::DrawGeoMesh(gfx::Vec3d geoOffset) {
	// Adjust our infinite bitmap view, if necessary
	FrameObject& f       = Frames[0];
	auto         boundsF = f.Mesh.PosBounds();
	//f.Mesh.PrintPosX(Rect32(0, 0, f.Mesh.Width, f.Mesh.Height));
	RectD boundsM(boundsF.x1, boundsF.y1, boundsF.x2, boundsF.y2);
	boundsM.Offset(geoOffset.x, geoOffset.y);

	double baseMetersPerPixel = BaseMapMetersPerPixel();
	Rect64 boundsBasePix;
	boundsBasePix.x1 = int64_t(boundsM.x1 / baseMetersPerPixel);
	boundsBasePix.y1 = int64_t(boundsM.y1 / baseMetersPerPixel);
	boundsBasePix.x2 = int64_t(boundsM.x2 / baseMetersPerPixel);
	boundsBasePix.y2 = int64_t(boundsM.y2 / baseMetersPerPixel);

	auto err = AdjustInfiniteBitmapViewForGeo(boundsBasePix);
	if (!err.OK())
		return err;

	//Vec2d geoOffsetPix;
	//geoOffsetPix.x /= baseMetersPerPixel;
	//geoOffsetPix.y /= baseMetersPerPixel;
	Vec2d basePixelsPerMeter2d(1.0 / baseMetersPerPixel, 1.0 / baseMetersPerPixel);
	Vec2d rendGeoPixOrigin(InfBmpView.x1, InfBmpView.y1);

	// adjust mesh coordinates for final framebuffer position
	for (int i = 0; i < f.Mesh.Count; i++) {
		auto& v       = f.Mesh.Vertices[i];
		Vec2d geoM    = Vec2d(geoOffset.x + v.Pos.x, geoOffset.y + v.Pos.y);
		Vec2d geoPix  = geoM * basePixelsPerMeter2d;
		Vec2d rendPix = geoPix - rendGeoPixOrigin;
		v.Pos         = Vec2f(rendPix.x, rendPix.y);
	}
	Rend.DrawMesh(f.Mesh, f.Img);

	//Rend.DrawMeshWireframe(f.Mesh, Color8(200, 0, 0, 255), 0.6f);

	return Error();
}

void Stitcher::ExtrapolateMesh(const Mesh& smallMesh, Mesh& fullMesh, gfx::Vec2f& uvAtTopLeftOfImage) {
	Rect32 smallCropRect = VidStitcher.CropRectFromFullFlat();

	// This is the position of the upper-left corner of the small alignment image, within the full flattened image
	Vec2f smallPosInFullImg(smallCropRect.x1, smallCropRect.y1);

	//Vec2f uvInterval = smallMesh.At(smallMesh.Width - 1, smallMesh.Height - 1).UV - smallMesh.At(0, 0).UV;
	//uvInterval *= Vec2f(1.0f / (float) (smallMesh.Width - 1), 1.0f / (float) (smallMesh.Height - 1));
	Vec2f uvInterval(VidStitcher.PixelsPerMeshCell, VidStitcher.PixelsPerMeshCell);

	// Align to the first valid vertex of smallMesh
	int  alignMX, alignMY;
	bool ok = smallMesh.FirstValid(alignMX, alignMY);
	IMQS_ASSERT(ok);

	// This position of the (somewhat arbitrary) alignment vertex, inside the full flattened image
	Vec2f alignPos = smallMesh.At(alignMX, alignMY).UV + smallPosInFullImg;

	// If we were to compute a uniform full mesh right now, with the origin at (0,0), then
	// what we would be the nearest vertex to alignPos?
	Vec2f correction = Vec2f(fmod(alignPos.x, uvInterval.x), fmod(alignPos.y, uvInterval.y));

	// We shift the origin of our full mesh, by the correction factor, so that we end up with
	// vertices who's UV coordinates precisely match those of the small mesh
	Vec2f fullOrigin = -uvInterval + correction;

	// compute full mesh width/height, now that we know the correction factor
	int fullImgWidth  = VidStitcher.FullFlat.Width;
	int fullImgHeight = VidStitcher.FullFlat.Height;
	int mWidth        = (int) ((fullImgWidth - fullOrigin.x + uvInterval.x - 1) / uvInterval.x);
	int mHeight       = (int) ((fullImgHeight - fullOrigin.y + uvInterval.y - 1) / uvInterval.y);

	// delta in mesh coordinates, from small to full
	int smallToFullDX = 0;
	int smallToFullDY = 0;

	uvAtTopLeftOfImage = -fullOrigin;

	fullMesh.Initialize(mWidth, mHeight);
	for (int y = 0; y < mHeight; y++) {
		for (int x = 0; x < mWidth; x++) {
			Vec2f p = fullOrigin + Vec2f((float) x, (float) y) * uvInterval;
			if (p.distanceSQ(alignPos) < 1.0f) {
				smallToFullDX = x - alignMX;
				smallToFullDY = y - alignMY;
			}
			fullMesh.At(x, y).UV      = p;
			fullMesh.At(x, y).Pos     = p;
			fullMesh.At(x, y).IsValid = false;
		}
	}
	IMQS_ASSERT(smallToFullDX != 0 && smallToFullDY != 0);

	Rect32 validRectFull = Rect32::Inverted();

	// Copy the vertices from the small mesh
	for (int y = 0; y < smallMesh.Height; y++) {
		for (int x = 0; x < smallMesh.Width; x++) {
			if (smallMesh.At(x, y).IsValid) {
				int fx                      = x + smallToFullDX;
				int fy                      = y + smallToFullDY;
				fullMesh.At(fx, fy).IsValid = true;
				// The UV coordinates in the source will have been twiddled so that they fall precisely
				// in between a quad of pixels, so here we bring over that twiddling too. We expect
				// the UV coordinates here to change by at most 1 pixel in X and Y.
				// The Pos coordinates, on the other hand, are the alignment positions, so those we
				// expect to have changed a lot.
				fullMesh.At(fx, fy).UV  = smallMesh.At(x, y).UV + smallPosInFullImg;
				fullMesh.At(fx, fy).Pos = smallMesh.At(x, y).Pos + smallPosInFullImg;
				validRectFull.ExpandToFit(fx, fy);
			}
		}
	}

	// fill in all non-valid values with the average displacement
	Vec2f avgDisp = smallMesh.AvgValidDisplacement();
	for (int i = 0; i < fullMesh.Count; i++) {
		if (!fullMesh.Vertices[i].IsValid) {
			fullMesh.Vertices[i].Pos = fullMesh.Vertices[i].UV + avgDisp;
		}
	}
}

void Stitcher::TransformMeshIntoRendCoords(Mesh& mesh) {
	for (int i = 0; i < mesh.Count; i++) {
		mesh.Vertices[i].Pos += PrevTopLeft;
	}
	/*
	// thinking.. we want to rotate by current bearing, not by this number here
	// also.. need to do rotation before alignment..
	float angle = -PrevDir.angle(Vec2f(0, -1));
	angle       = 0;
	Vec2f r0(cos(angle), -sin(angle));
	Vec2f r1(sin(angle), cos(angle));
	for (int i = 0; i < mesh.Count; i++) {
		Vec2f p;
		// rotate about the origin
		p.x = mesh.Vertices[i].Pos.dot(r0);
		p.y = mesh.Vertices[i].Pos.dot(r1);
		// translate to the top-left of the previous frame
		p += PrevTopLeft;
		mesh.Vertices[i].Pos = p;
	}
	*/
}

Error Stitcher::AdjustInfiniteBitmapViewForGeo(gfx::Rect64 outRect) {
	int tilesX = Rend.FBWidth / InfBmp.TileSize;
	int tilesY = Rend.FBHeight / InfBmp.TileSize;

	//tsf::print("outRect: %v,%v,%v,%v\n", outRect.x1, outRect.y1, outRect.x2, outRect.y2);

	if (outRect.x1 >= InfBmpView.x1 &&
	    outRect.y1 >= InfBmpView.y1 &&
	    outRect.x2 <= InfBmpView.x2 &&
	    outRect.y2 <= InfBmpView.y2) {
		// viewport is OK

		// modify the dirty matrix, so that we know which tiles are about to be touched
		auto dirty = outRect;
		dirty.Offset(-InfBmpView.x1, -InfBmpView.y1);
		dirty.x1 = InfiniteBitmap::RoundDown64(dirty.x1, InfBmp.TileSize);
		dirty.y1 = InfiniteBitmap::RoundDown64(dirty.y1, InfBmp.TileSize);
		dirty.x2 = InfiniteBitmap::RoundUp64(dirty.x2, InfBmp.TileSize);
		dirty.y2 = InfiniteBitmap::RoundUp64(dirty.y2, InfBmp.TileSize);
		dirty.Divide(InfBmp.TileSize);

		//tsf::print("Dirty Tiles: %v,%v - %v,%v\n", dirty.x1, dirty.y1, dirty.x2, dirty.y2);
		IMQS_ASSERT(dirty.x1 >= 0 && dirty.y1 >= 0 && dirty.x2 <= tilesX && dirty.y2 <= tilesY);

		for (int64_t y = dirty.y1; y < dirty.y2; y++) {
			for (int64_t x = dirty.x1; x < dirty.x2; x++) {
				*InfBmpDirty.At(x, y) = 255;
			}
		}

		return Error();
	}

	// Persist current framebuffer
	Image oldImg;
	if (!DryRun && InfBmpView.Width() != 0) {
		bool* sparse = new bool[tilesX * tilesY];
		int   nwrite = 0;
		for (int ty = 0; ty < tilesY; ty++) {
			for (int tx = 0; tx < tilesX; tx++) {
				sparse[ty * tilesX + tx] = *InfBmpDirty.At(tx, ty) != 0;
				if (sparse[ty * tilesX + tx])
					nwrite++;
			}
		}
		if (PrintTileIOMessages)
			tsf::print("Writing %v/%v tiles\n", nwrite, tilesX * tilesY);
		Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, oldImg);
		auto err = InfBmp.Save(BaseZoomLevel, InfBmpView, oldImg, sparse);
		if (PrintTileIOMessages)
			tsf::print("Writing complete\n");
		delete[] sparse;
		if (!err.OK())
			return err;
	}

	Rect64 newView;
	if (outRect.CenterX() < InfBmpView.CenterX()) {
		newView.x2 = InfiniteBitmap::RoundUp64(outRect.x2, InfBmp.TileSize);
		newView.x1 = newView.x2 - Rend.FBWidth;
	} else {
		newView.x1 = InfiniteBitmap::RoundDown64(outRect.x1, InfBmp.TileSize);
		newView.x2 = newView.x1 + Rend.FBWidth;
	}

	if (outRect.CenterY() < InfBmpView.CenterY()) {
		newView.y2 = InfiniteBitmap::RoundUp64(outRect.y2, InfBmp.TileSize);
		newView.y1 = newView.y2 - Rend.FBHeight;
	} else {
		newView.y1 = InfiniteBitmap::RoundDown64(outRect.y1, InfBmp.TileSize);
		newView.y2 = newView.y1 + Rend.FBHeight;
	}

	if (outRect.x1 < newView.x1 ||
	    outRect.y1 < newView.y1 ||
	    outRect.x2 > newView.x2 ||
	    outRect.y2 > newView.y2) {
		return Error::Fmt("Resolution is too high to draw a single mesh into the framebuffer. Resolution is %v x %v",
		                  outRect.Width(), outRect.Height());
	}

	Image newImg;
	newImg.Alloc(gfx::ImageFormat::RGBAP, Rend.FBWidth, Rend.FBHeight);
	newImg.Fill(Color8(0, 0, 0, 0));
	if (!DryRun) {
		// Copy the existing tiles over from the previous image. It's wasteful to just rely on dumping & loading
		// to restore state, so it's worthwhile doing this, especially over a (relatively) high latency system like GCS.
		bool* sparse = new bool[tilesX * tilesY];
		memset(sparse, 1, tilesX * tilesY);
		int    nread = tilesX * tilesY;
		Rect64 oldInNew64(0, 0, oldImg.Width, oldImg.Height);
		oldInNew64.Offset(InfBmpView.x1 - newView.x1, InfBmpView.y1 - newView.y1);
		if (abs(oldInNew64.x1) < 1 << 20 && abs(oldInNew64.y1) < 1 << 20) {
			auto oldInNew = Rect64to32(oldInNew64);
			//tsf::print("%.2f: Keeping %v,%v - %v,%v\n", VidStitcher.FrameTime, oldInNew.x1, oldInNew.y1, oldInNew.x2, oldInNew.y2);
			newImg.CopyFrom(oldImg, Rect32(0, 0, oldImg.Width, oldImg.Height), oldInNew);

			auto validTiles = oldInNew;
			validTiles.CropTo(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight));
			validTiles.Divide(InfBmp.TileSize);

			for (int ty = 0; ty < tilesY; ty++) {
				for (int tx = 0; tx < tilesX; tx++) {
					if (validTiles.IsInsideMe(tx, ty)) {
						sparse[ty * tilesX + tx] = false;
						nread--;
					}
				}
			}
		}

		if (PrintTileIOMessages)
			tsf::print("Reading %v/%v tiles\n", nread, tilesX * tilesY);

		// load any tiles that were not in our previous view
		auto err = InfBmp.Load(BaseZoomLevel, newView, newImg, sparse);
		delete[] sparse;
		if (PrintTileIOMessages)
			tsf::print("Reading complete\n");
		if (!err.OK())
			return err;
	}
	Rend.Clear(ClearColor);
	Rend.CopyImageToDevice(newImg, 0, 0);
	InfBmpDirty.Alloc(ImageFormat::Gray, tilesX, tilesY);
	InfBmpDirty.Fill(Color8(0, 0, 0, 0));

	InfBmpView = newView;

	return Error();
}

Error Stitcher::AdjustInfiniteBitmapView(const Mesh& m, gfx::Vec2f travelDirection) {
	auto isInside = [&](Vec2f p) {
		return p.x >= 0 && p.y >= 0 && p.x < Rend.FBWidth && p.y < Rend.FBHeight;
	};
	// check the two extreme points at the front of the frustum, to detect if we're wandering outside of the framebuffer
	if (isInside(m.At(0, 0).Pos) && isInside(m.At(m.Width - 1, 0).Pos))
		return Error();

	bool persistToInfBmp = false;

	// Persist current framebuffer
	Image img;
	if (persistToInfBmp) {
		Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, img);
		auto err = InfBmp.Save(BaseZoomLevel, InfBmpView, img);
		if (!err.OK())
			return err;
	}

	if (false) {
		Image test;
		auto  err = InfBmp.Load(BaseZoomLevel, InfBmpView, test);
		test.SaveFile("test-inf-view-1.jpeg");
	}

	// Figure out the best new viewport, given the direction of travel

	// Compute the bounding box of the warp mesh, in InfBmp coordinates
	Rect64  meshBounds = Rect64::Inverted();
	Point32 samples[4] = {{0, 0}, {m.Width - 1, 0}, {m.Width - 1, m.Height - 1}, {0, m.Height - 1}};
	for (int i = 0; i < 4; i++) {
		auto p = m.At(samples[i].x, samples[i].y).Pos;
		meshBounds.ExpandToFit((int64_t) p.x + InfBmpView.x1, (int64_t) p.y + InfBmpView.y1);
	}

	Rect64 newView;
	if (travelDirection.x > 0) {
		newView.x1 = InfiniteBitmap::RoundDown64(meshBounds.x1, InfBmp.TileSize);
		newView.x2 = newView.x1 + Rend.FBWidth;
	} else {
		newView.x2 = InfiniteBitmap::RoundUp64(meshBounds.x2, InfBmp.TileSize);
		newView.x1 = newView.x2 - Rend.FBWidth;
	}

	if (travelDirection.y > 0) {
		newView.y1 = InfiniteBitmap::RoundDown64(meshBounds.y1, InfBmp.TileSize);
		newView.y2 = newView.y1 + Rend.FBHeight;
	} else {
		newView.y2 = InfiniteBitmap::RoundUp64(meshBounds.y2, InfBmp.TileSize);
		newView.y1 = newView.y2 - Rend.FBHeight;
	}

	if (persistToInfBmp) {
		img.Fill(Color8(0, 0, 0, 0));
		auto err = InfBmp.Load(BaseZoomLevel, newView, img);
		if (!err.OK())
			return err;
		Rend.Clear(ClearColor);
		Rend.CopyImageToDevice(img, 0, 0);
	} else {
		Rend.Clear(ClearColor);
	}

	if (false) {
		img.SaveFile("test-inf-view-load.jpeg");
		Image test;
		Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, test);
		test.SaveFile("test-inf-view-in-FB.jpeg");
	}

	Vec2f adjust = Vec2f(float(InfBmpView.x1 - newView.x1), float(InfBmpView.y1 - newView.y1));
	PrevTopLeft += adjust;
	//StitchTopLeft += adjust;
	//PrevBottomMidAlignPoint += adjust;
	InfBmpView = newView;

	return Error();
}

int WebTiles(argparse::Args& args) {
	string         storageSpec = args.Params[0];
	InfiniteBitmap bmp;
	auto           err = bmp.Initialize(storageSpec);
	if (err.OK())
		err = bmp.CreateWebTiles(25);
	if (!err.OK()) {
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

int MeasureScale(argparse::Args& args) {
	auto videoFiles = strings::Split(args.Params[0], ',');
	auto trackFile  = args.Params[1];
	auto flattenStr = args.Params[2];

	FlattenParams fp;
	auto          err = fp.ParseJson(flattenStr);
	if (err.OK()) {
		Stitcher s;
		err = s.DoMeasureScale(videoFiles, trackFile, fp);
	}
	if (!err.OK()) {
		tsf::print("Error measuring scale: %v\n", err.Message());
		return 1;
	}
	return 0;
}

int Stitch(argparse::Args& args) {
	auto   videoFiles     = strings::Split(args.Params[0], ',');
	auto   storageSpec    = args.Params[1];
	auto   trackFile      = args.Params[2];
	auto   flattenStr     = args.Params[3];
	int    count          = args.GetInt("number");
	double seek           = args.GetDouble("start");
	double metersPerPixel = args.GetDouble("mpp");

	FlattenParams fp;
	auto          err = fp.ParseJson(flattenStr);
	if (err.OK()) {
		Stitcher s;
		s.DryRun         = args.Has("dryrun");
		s.MetersPerPixel = metersPerPixel;
		err              = s.DoStitch(storageSpec, videoFiles, trackFile, fp, seek, count);
	}
	if (!err.OK()) {
		tsf::print("Error stitching: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs