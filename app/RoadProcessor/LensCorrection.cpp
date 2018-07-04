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

	// If this is a zoom lense, then one would need to enter this information
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