#include "pch.h"
#include "LensCorrection.h"

using namespace std;

namespace imqs {
namespace roadproc {

LensCorrector::LensCorrector() {
}

LensCorrector::~LensCorrector() {
	free(InterpPos);
	if (Mod)
		lf_modifier_destroy(Mod);
	if (DB)
		lf_db_destroy(DB);
}

Error LensCorrector::LoadDatabase(std::string dbPath) {
	DB       = lf_db_create();
	auto err = DB->Load(dbPath.c_str());
	if (err != LF_NO_ERROR)
		return Error::Fmt("Failed to load lens correction database at '%v': %v", dbPath, err);
	return Error();
}

Error LensCorrector::LoadCameraAndLens(std::string spec) {
	auto parts = strings::Split(spec, ',');
	if (parts.size() != 2)
		return Error::Fmt("Expected 'Camera,Lens', eg 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS'");

	return LoadCameraAndLens(parts[0], parts[1]);
}

Error LensCorrector::LoadCameraAndLens(std::string camera, std::string lens) {
	auto camParts = strings::Split(camera, ' ');
	if (camParts.size() != 2)
		return Error::Fmt("Expected camera to be specified as 'Maker Model', eg 'Fujifilm X-T2'");

	const lfCamera** cameras = DB->FindCameras(camParts[0].c_str(), camParts[1].c_str());
	if (!cameras)
		return Error::Fmt("No matching cameras found");

	size_t ncam = 0;
	string camList;
	for (auto cam = cameras; *cam; cam++) {
		ncam++;
		camList += tsf::fmt("%v %v\n", (*cam)->Maker, (*cam)->Model);
	}
	if (ncam == 1)
		Camera = cameras[0];
	lf_free(cameras);
	if (ncam == 0)
		return Error("No matching cameras found");
	else if (ncam > 1)
		return Error::Fmt("More than one matching camera found:\n%v", camList);

	// Lens

	const lfLens** lenses = DB->FindLenses(Camera, nullptr, lens.c_str());
	if (!lenses)
		return Error::Fmt("No matching lenses found");

	size_t nlens = 0;
	string lensList;
	for (auto len = lenses; *len; len++) {
		nlens++;
		lensList += tsf::fmt("%v\n", (*len)->Model);
	}
	if (nlens == 1)
		Lens = lenses[0];
	lf_free(lenses);
	if (nlens == 0)
		return Error("No matching cameras found");
	else if (nlens > 1)
		return Error::Fmt("More than one matching lens found:\n%v", lensList);

	return Error();
}

Error LensCorrector::InitializeDistortionCorrect(int width, int height) {
	if (Mod) {
		lf_modifier_destroy(Mod);
		Mod = nullptr;
	}
	free(InterpPos);
	InterpPos = nullptr;

	float crop = Camera->CropFactor;
	Mod        = lf_modifier_create(crop, width, height, LF_PF_U8, false);
	if (!Mod)
		return Error("Unable to create lens modifier");

	// If this is a zoom lens, then one would need to enter this information
	float focal = Lens->MinFocal;

	int enabled = Mod->EnableDistortionCorrection(Lens, focal);
	if (!(enabled & LF_MODIFY_DISTORTION))
		return Error("Failed to initialize distortion correction");

	// *2 for X and Y coords
	// *3 for RGB channels
	InterpPos  = (float*) malloc(width * 2 * 3 * sizeof(float));
	ImageWidth = width;

	return Error();
}

struct DistortionEl {
	float Rx, Ry;
	float Gx, Gy;
	float Bx, By;
};

Error LensCorrector::ComputeCorrection(int width, int height, gfx::Image& vignetting, gfx::Image& distortion, gfx::Image* combined) {
	bool debugPrint    = false;
	int  maxDebugPrint = 20;
	// The Fuji X-T2 sensor is 23.6mm x 15.6 mm.
	// Assuming LensFun emits values for the entire sensor, but we're doing 16x9 shooting (FHD),
	// then we need to chop off some of the top/bottom results.
	// 23.6 / 15.6 = 1.5128
	// 16 / 9 = 1.7777
	// The sensor's aspect ratio is more square than 16/9.
	// Anyway - if we divide 1.77777 / 1.5128, we get 1.17, which is precisely the quoted "crop factor"
	// that the reviews of the X-T2 talk about, when they mention FHD/4K recording. So it looks like we've got the
	// right number here.
	// Unfortunately LensFun doesn't seem to have the sensor dimensions in it's database. But even if it did,
	// it would also need to specify how the cropping gets done. As far as I know, it's not always as obvious
	// as this case with the X-T2.

	// Hardcoded value for the Fuji X-T2. LensFun doesn't seem to have this data in it's DB, so if we
	// want to support different cameras, then we'll need to know this for that camera.
	double sensorAspectRatio = 23.6 / 15.6;

	// This is the aspect ratio of the footage that is coming out of the camera. This is often cropped for
	// video (as explained above). The part of the sensor that's thrown away comes from the top and bottom,
	// because camera sensors are usually more square, and videos are usually more wide.
	double usedAspectRatio = (double) width / (double) height;

	// When we get the distortion parameters out of LensFun, we want to give it a matrix that is the
	// same ratio as the camera's sensor. This seems like the most sane thing to do, because it
	// makes it easy to reason about; The data coming out of LensFun is 1:1 with the camera sensor.
	// Whatever data we want to throw away, we throw away after getting the 1:1 data from LensFun.
	int sensorWidth, sensorHeight;
	if (usedAspectRatio > sensorAspectRatio) {
		// expected case (sensor is square, video is wide)
		sensorWidth  = width;
		sensorHeight = height * (usedAspectRatio / sensorAspectRatio);
	} else {
		// unexpected case (sensor is wide, video is square)
		sensorWidth  = width / (usedAspectRatio / sensorAspectRatio);
		sensorHeight = height;
	}

	auto mod = lf_modifier_create(Camera->CropFactor, sensorWidth, sensorHeight, LF_PF_F32, false);
	if (!mod)
		return Error("Unable to create lens modifier");

	float focal    = Lens->MinFocal;
	float aperture = 5.6; // this is what we're recording on in August 2018, on the Fuji X-T2
	float distance = 2.0; // 2 meters
	int   enabled  = mod->EnableVignettingCorrection(Lens, focal, aperture, distance);
	if (!(enabled & LF_MODIFY_VIGNETTING))
		return Error("Failed to initialize vignetting correction");

	enabled = mod->EnableDistortionCorrection(Lens, focal);
	if (!(enabled & LF_MODIFY_DISTORTION))
		return Error("Failed to initialize distortion correction");

	float* pxColor = new float[sensorWidth * sensorHeight];
	for (int i = 0; i < sensorWidth * sensorHeight; i++)
		pxColor[i] = 1;

	bool ok = mod->ApplyColorModification(pxColor, 0.0, 0.0, sensorWidth, sensorHeight, LF_CR_1(INTENSITY), sensorWidth * sizeof(float));
	IMQS_ASSERT(ok);

#define SKIP_Y(Y, HEIGHT)                                   \
	if (HEIGHT > maxDebugPrint && Y == maxDebugPrint / 2) { \
		tsf::print(" .. \n");                               \
		Y = HEIGHT - maxDebugPrint / 2;                     \
		continue;                                           \
	}
#define SKIP_X(X, WIDTH)                                   \
	if (WIDTH > maxDebugPrint && X == maxDebugPrint / 2) { \
		tsf::print(".. ");                                 \
		X = WIDTH - maxDebugPrint / 2;                     \
		continue;                                          \
	}

	if (debugPrint) {
		tsf::print("---------------------------------Vignetting------------------------------------\n");
		for (int y = 0; y < sensorHeight; y++) {
			SKIP_Y(y, sensorHeight);
			for (int x = 0; x < sensorWidth; x++) {
				SKIP_X(x, sensorWidth);
				tsf::print("%4.2f ", pxColor[y * width + x]);
			}
			tsf::print("\n");
		}
	}

	// Assume here that the video crop region lies in the center of the sensor (a very reasonable assumption)
	vignetting.Alloc(gfx::ImageFormat::Gray, width, height);
	if (combined)
		combined->Alloc(gfx::ImageFormat::F32_RGBA, width, height);
	int x1 = (sensorWidth - width) / 2;
	int y1 = (sensorHeight - height) / 2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int   sx             = x + x1;
			int   sy             = y + y1;
			float v              = pxColor[sy * sensorWidth + sx];
			*vignetting.At(x, y) = (uint8_t) math::Clamp<float>(v * VignetteGrayMultiplier, 0, 255);
			if (combined) {
				float* com = combined->AtF32(x, y);
				com[0]     = v;
			}
		}
	}

	if (debugPrint) {
		for (int y = 0; y < height; y++) {
			SKIP_Y(y, height);
			for (int x = 0; x < width; x++) {
				SKIP_X(x, width);
				tsf::print("%3d ", (int) *vignetting.At(x, y));
			}
			tsf::print("\n");
		}
	}

	delete[] pxColor;

	// Distortion
	DistortionEl* pxDistort = new DistortionEl[sensorWidth * sensorHeight];
	ok                      = mod->ApplySubpixelGeometryDistortion(0.0, 0.0, sensorWidth, sensorHeight, (float*) pxDistort);
	IMQS_ASSERT(ok);

	maxDebugPrint = 10;

	if (debugPrint) {
		tsf::print("---------------------------------Distortion------------------------------------\n");
		for (int y = 0; y < sensorHeight; y++) {
			SKIP_Y(y, sensorHeight);
			for (int x = 0; x < sensorWidth; x++) {
				SKIP_X(x, sensorWidth);
				tsf::print("%4.2f,%4.2f ", pxDistort[y * width + x].Gx, pxDistort[y * width + x].Gy);
			}
			tsf::print("\n");
		}
	}

	// Same rules apply here, as for the color correction (ie crop is in center of sensor)
	distortion.Alloc(gfx::ImageFormat::F32_RG, width, height);
	auto bounds = gfx::RectF::Inverted();
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int         sx = x + x1;
			int         sy = y + y1;
			const auto& v  = pxDistort[sy * sensorWidth + sx];
			float*      xy = distortion.AtF32(x, y);
			// I find it strange that lensfun's values are scaled to width-1 and height-1, instead of width and height,
			// but the numbers look correct this way, in the sense that left/right and top/bottom are symmetrical.
			float nGx = v.Gx / (float) (width - 1);
			float nGy = v.Gy / (float) (height - 1);
			xy[0]     = nGx;
			xy[1]     = nGy;
			if (combined) {
				float* com = combined->AtF32(x, y);
				com[1]     = nGx;
				com[2]     = nGy;
			}
			bounds.ExpandToFit(v.Gx, v.Gy);
		}
	}

	if (debugPrint) {
		tsf::print("Bounds of raw distortion values: %v,%v - %v,%v\n", bounds.x1, bounds.y1, bounds.x2, bounds.y2);
		for (int y = 0; y < height; y++) {
			SKIP_Y(y, height);
			for (int x = 0; x < width; x++) {
				SKIP_X(x, width);
				tsf::print("%4.2f,%4.2f ", distortion.AtF32(x, y)[0], distortion.AtF32(x, y)[1]);
			}
			tsf::print("\n");
		}
	}

	delete[] pxDistort;

	return Error();
}

void LensCorrector::ComputeDistortionForLine(int y) {
	Mod->ApplySubpixelGeometryDistortion(0.0, (float) y, ImageWidth, 1, InterpPos);
}

Error LensCorrector::Correct(gfx::Image& raw, gfx::Image& fixed) {
	auto err = InitializeDistortionCorrect(raw.Width, raw.Height);
	if (!err.OK())
		return err;

	fixed.Alloc(gfx::ImageFormat::RGBA, raw.Width, raw.Height);
	size_t lwidth = raw.Width * 2 * 3;
	float* pos    = (float*) malloc(lwidth * sizeof(float));

	// This code is based on lenstool.cpp

	for (int step = 2; step < 3; step++) {
		for (int y = 0; y < raw.Height; y++) {
			uint8_t* imgIn = raw.Line(y);
			bool     ok    = false;
			switch (step) {
			case 1:
				// Colour correction: vignetting
				ok = Mod->ApplyColorModification(imgIn, 0.0, y, raw.Width, 1, LF_CR_4(RED, GREEN, BLUE, UNKNOWN), 0);
				break;
			case 2:
				// TCA and geometry correction
				ok = Mod->ApplySubpixelGeometryDistortion(0.0, y, raw.Width, 1, pos);
				if (ok) {
					float*    src    = pos;
					int       width  = raw.Width;
					uint8_t*  rawBuf = raw.Data;
					uint32_t* out    = (uint32_t*) fixed.Line(y);
					int32_t   uclamp = (raw.Width - 1) * 256 - 1;
					int32_t   vclamp = (raw.Height - 1) * 256 - 1;
					for (int x = 0; x < width; x++) {
						// just use green, because we're not interested in chromatic corrections
						int u = (int) src[2] * 256;
						int v = (int) src[3] * 256;
						*out  = gfx::raster::ImageBilinearRGBA(rawBuf, width, uclamp, vclamp, u, v);
						//gfx::raster::Bilinear_x64()
						//dst->red   = img->GetR(src[0], src[1]);
						//dst->green = img->GetG(src[2], src[3]);
						//dst->blue  = img->GetB(src[4], src[5]);
						src += 2 * 3;
						out++;
						//dst++;
					}
				}
				break;
			}
		}
	}

	free(pos);

	return Error();
}

} // namespace roadproc
} // namespace imqs