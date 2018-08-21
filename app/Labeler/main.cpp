#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"
#include "SvgIcons.h"
#include "UI.h"

using namespace imqs;
using namespace std;

train::LabelTaxonomy CreateTaxonomy(bool isSegmentation) {
	train::LabelTaxonomy taxonomy;
	if (isSegmentation) {
		taxonomy.Classes.push_back({false, false, 'U', "", "unlabeled"});
		taxonomy.Classes.push_back({true, false, 'L', "mark", "lane"});
	} else {
		// !!!!!!!!!!!!!!! NOTE !!!!!!!!!!!!!!!
		// If you make changes that need once-off fixups, then the best place to
		// write the fix-up code is inside UI::LoadLabels()

		taxonomy.GridSize = 384;

		// first class must be unlabeled - as embodied by UI::UnlabeledClass
		taxonomy.Classes.push_back({false, false, 'U', "", "unlabeled"});

		// type of surface
		taxonomy.Classes.push_back({false, false, 'A', "type", "tar"});
		taxonomy.Classes.push_back({false, false, 'D', "type", "dirt"});
		taxonomy.Classes.push_back({false, false, 'Z', "type", "curb"});
		taxonomy.Classes.push_back({false, true, 'E', "type", "edge"});
		taxonomy.Classes.push_back({false, false, 'W', "type", "grass"});
		taxonomy.Classes.push_back({false, false, 'V', "type", "vehicle"});
		taxonomy.Classes.push_back({false, false, 'M', "type", "manhole"});
		taxonomy.Classes.push_back({false, false, 'R', "type", "rubbish"});
		taxonomy.Classes.push_back({false, false, 'Q', "type", "speed bump"});
		taxonomy.Classes.push_back({false, false, 'X', "type", "pedestrian"});

		// surface artifacts
		taxonomy.Classes.push_back({false, false, 'N', "defect", "nothing"});
		taxonomy.Classes.push_back({false, true, 'F', "defect", "surf. failure"}); // degree only really starts at 3. 1 is probably too small to see.
		taxonomy.Classes.push_back({false, true, 'S', "defect", "surf. crack"});   // "random" other cracks. 1 is probably too small to see.
		taxonomy.Classes.push_back({false, true, 'L', "defect", "long. crack"});   // crack direction is same as road direction
		taxonomy.Classes.push_back({false, true, 'T', "defect", "trans. crack"});  // crack direction is across road
		taxonomy.Classes.push_back({false, true, 'B', "defect", "block crack"});   // blocks (bigger than crocodile, rectangular)
		taxonomy.Classes.push_back({false, true, 'C', "defect", "croc. crack"});   // crocodile
		taxonomy.Classes.push_back({false, true, 'G', "defect", "patching"});      // patching
		taxonomy.Classes.push_back({false, true, 'H', "defect", "pothole"});       // pothole
		taxonomy.Classes.push_back({false, false, 'Y', "defect", "cut"});          // man-made cutting

		// aggregate loss
		taxonomy.Classes.push_back({false, true, 'I', "agg. loss", "agg. loss"});

		// binder
		taxonomy.Classes.push_back({false, true, 'J', "binder", "binder"});

		// bleeding
		taxonomy.Classes.push_back({false, true, 'K', "bleeding", "bleeding"});

		// pumping
		taxonomy.Classes.push_back({false, true, 'P', "pumping", "pumping"});

		imqs::train::ExportClassTaxonomy("taxonomy.json", taxonomy.Classes);
	}
	return taxonomy;
}

void xoMain(xo::SysWnd* wnd) {
	using namespace imqs::anno;
	imqs::video::VideoFile::Initialize();

	/*
	// Bulk conversion, when changing on-disk format
	{
		using namespace imqs::train;
		VideoLabels labels;
		auto        err = LoadVideoLabels("c:\\mldata\\GOPR0080.MP4", labels);
		err             = SaveVideoLabels("c:\\mldata\\GOPR0080.MP4", labels);
		return;
	}
	*/

	wnd->SetTitle("IMQS Video Labeler");
	wnd->Doc()->ClassParse("font-medium", "font-size: 14ep");
	wnd->Doc()->ClassParse("shortcut", "font-size: 15ep; color: #000; width: 1em");
	wnd->Doc()->ClassParse("severity", "font-size: 20ep; color: #000; width: 1em; font-weight: bold");
	wnd->Doc()->ClassParse("label-group", "padding: 3ep; margin: 2ep 0ep 2ep 0ep; border: 1px #ddd; border-radius: 4ep; background: #f0f0f0");
	wnd->Doc()->ClassParse("active-label", "font-weight: bold");

	svg::LoadAll(wnd->Doc());
	//wnd->SetPosition(xo::Box(0, 0, 1600, 1020), xo::SysWnd::SetPosition_Size);

	auto ui       = new UI(&wnd->Doc()->Root);
	ui->LabelMode = UI::LabelModes::FixedBoxes;
	ui->Taxonomy  = CreateTaxonomy(ui->LabelMode == UI::LabelModes::Segmentation);

	//ui->VideoFilename = "c:\\mldata\\GOPR0080.MP4";
	//ui->VideoFilename = "C:\\mldata\\DSCF3022.MOV";
	//ui->VideoFilename = "T:\\IMQS8_Data\\ML\\DSCF3022.MOV";
	ui->VideoFilename = "/home/ben/mldata/mthata/Day3-11.MOV";
	//ui->VideoFilename = "LOAD FILE";
	if (!ui->OpenVideo())
		ui->Render();
}

// On Windows, you must uncomment the line:
//   PROGOPTS = { "/SUBSYSTEM:CONSOLE"; Config = winFilter },
// inside units.lua, for "Labeler" project
int main(int argc, char** argv) {
	using namespace imqs::train;

	imqs::video::VideoFile::Initialize();

	// Bulk export of jpeg patches
	{
		auto taxonomy = CreateTaxonomy(false);
		auto err      = ExportLabeledImagePatches_Video_Bulk(ExportTypes::Jpeg, "T:\\Temp\\ML", taxonomy);
		if (!err.OK()) {
			tsf::print("Error: %v\n", err.Message());
			return 1;
		}
		return 0;
	}

	return 1;
}
