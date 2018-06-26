#include "pch.h"
#include "third_party/xo/templates/xoWinMain.cpp"
#include "SvgIcons.h"
#include "UI.h"

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
	ui->LabelMode = UI::LabelModes::Segmentation;

#ifdef IMQS_AI_API
	auto err         = ui->Model.Load("c:\\mldata\\cp\\model.cntk");
	ui->ModelLoadErr = err.Message();
#endif

	if (ui->LabelMode == UI::LabelModes::Segmentation) {
		ui->Taxonomy.Classes.push_back({false, false, 'U', "", "unlabeled"});
		ui->Taxonomy.Classes.push_back({true, false, 'L', "mark", "lane"});
	} else if (ui->LabelMode == UI::LabelModes::FixedBoxes) {
		// !!!!!!!!!!!!!!! NOTE !!!!!!!!!!!!!!!
		// If you make changes that need once-off fixups, then the best place to
		// write the fix-up code is inside UI::LoadLabels()

		// first class must be unlabeled - as embodied by UI::UnlabeledClass
		ui->Taxonomy.Classes.push_back({false, false, 'U', "", "unlabeled"});

		// type of surface
		ui->Taxonomy.Classes.push_back({false, false, 'A', "type", "tar"});
		ui->Taxonomy.Classes.push_back({false, false, 'D', "type", "dirt"});
		ui->Taxonomy.Classes.push_back({false, false, 'Z', "type", "curb"});
		ui->Taxonomy.Classes.push_back({false, true, 'E', "type", "edge"});
		ui->Taxonomy.Classes.push_back({false, false, 'W', "type", "grass"});
		ui->Taxonomy.Classes.push_back({false, false, 'V', "type", "vehicle"});
		ui->Taxonomy.Classes.push_back({false, false, 'M', "type", "manhole"});

		// surface artifacts
		ui->Taxonomy.Classes.push_back({false, false, 'N', "defect", "nothing"});
		ui->Taxonomy.Classes.push_back({false, true, 'F', "defect", "surf. failure"}); // degree only really starts at 3. 1 is probably too small to see.
		ui->Taxonomy.Classes.push_back({false, true, 'S', "defect", "surf. crack"});   // "random" other cracks. 1 is probably too small to see.
		ui->Taxonomy.Classes.push_back({false, true, 'L', "defect", "long. crack"});   // crack direction is same as road direction
		ui->Taxonomy.Classes.push_back({false, true, 'T', "defect", "trans. crack"});  // crack direction is across road
		ui->Taxonomy.Classes.push_back({false, true, 'B', "defect", "block crack"});   // blocks (bigger than crocodile, rectangular)
		ui->Taxonomy.Classes.push_back({false, true, 'C', "defect", "croc. crack"});   // crocodile
		ui->Taxonomy.Classes.push_back({false, true, 'G', "defect", "patching"});      // patching
		ui->Taxonomy.Classes.push_back({false, true, 'H', "defect", "pothole"});       // pothole
		ui->Taxonomy.Classes.push_back({false, false, 'Y', "defect", "cut"});          // man-made cutting

		// aggregate loss
		ui->Taxonomy.Classes.push_back({false, true, 'I', "agg. loss", "agg. loss"});

		// binder
		ui->Taxonomy.Classes.push_back({false, true, 'J', "binder", "binder"});

		// bleeding
		ui->Taxonomy.Classes.push_back({false, true, 'K', "bleeding", "bleeding"});

		// pumping
		ui->Taxonomy.Classes.push_back({false, true, 'P', "pumping", "pumping"});

		imqs::train::ExportClassTaxonomy("c:\\mldata\\taxonomy.json", ui->Taxonomy.Classes);
	} else {
		IMQS_DIE();
	}

	//ui->VideoFilename = "c:\\mldata\\GOPR0080.MP4";
	//ui->VideoFilename = "C:\\mldata\\DSCF3022.MOV";
	ui->VideoFilename = "T:\\IMQS8_Data\\ML\\DSCF3022.MOV";
	//ui->VideoFilename = "LOAD FILE";
	if (!ui->OpenVideo())
		ui->Render();
}
