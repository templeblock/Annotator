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

	auto ui = new UI(&wnd->Doc()->Root);

#ifdef IMQS_AI_API
	auto err         = ui->Model.Load("c:\\mldata\\cp\\model.cntk");
	ui->ModelLoadErr = err.Message();
#endif

	// !!!!!!!!!!!!!!! NOTE !!!!!!!!!!!!!!!
	// If you make changes that need once-off fixups, then the best place to
	// write the fix-up code is inside UI::LoadLabels()

	// first class must be unlabeled - as embodied by UI::UnlabeledClass
	ui->Classes.push_back({false, 'U', "", "unlabeled"});

	// type of surface
	ui->Classes.push_back({false, 'A', "type", "tar"});
	ui->Classes.push_back({false, 'D', "type", "dirt"});
	ui->Classes.push_back({false, 'Z', "type", "curb"});
	ui->Classes.push_back({true, 'E', "type", "edge"});
	ui->Classes.push_back({false, 'W', "type", "grass"});
	ui->Classes.push_back({false, 'V', "type", "vehicle"});
	ui->Classes.push_back({false, 'M', "type", "manhole"});

	// surface artifacts
	ui->Classes.push_back({false, 'N', "defect", "nothing"});
	ui->Classes.push_back({true, 'F', "defect", "surf. failure"}); // degree only really starts at 3. 1 is probably too small to see.
	ui->Classes.push_back({true, 'S', "defect", "surf. crack"});   // "random" other cracks. 1 is probably too small to see.
	ui->Classes.push_back({true, 'L', "defect", "long. crack"});   // crack direction is same as road direction
	ui->Classes.push_back({true, 'T', "defect", "trans. crack"});  // crack direction is across road
	ui->Classes.push_back({true, 'B', "defect", "block crack"});   // blocks (bigger than crocodile, rectangular)
	ui->Classes.push_back({true, 'C', "defect", "croc. crack"});   // crocodile
	ui->Classes.push_back({true, 'G', "defect", "patching"});      // patching
	ui->Classes.push_back({true, 'H', "defect", "pothole"});       // pothole
	ui->Classes.push_back({false, 'Y', "defect", "cut"});          // man-made cutting

	// aggregate loss
	ui->Classes.push_back({true, 'I', "agg. loss", "agg. loss"});

	// binder
	ui->Classes.push_back({true, 'J', "binder", "binder"});

	// bleeding
	ui->Classes.push_back({true, 'K', "bleeding", "bleeding"});

	// pumping
	ui->Classes.push_back({true, 'P', "pumping", "pumping"});

	imqs::train::ExportClassTaxonomy("c:\\mldata\\taxonomy.json", ui->Classes);

	//ui->VideoFilename = "c:\\mldata\\GOPR0080.MP4";
	//ui->VideoFilename = "C:\\mldata\\DSCF3022.MOV";
	ui->VideoFilename = "LOAD FILE";
	if (!ui->OpenVideo())
		ui->Render();
}
