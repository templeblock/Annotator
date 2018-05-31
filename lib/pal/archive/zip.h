#pragma once

namespace imqs {
namespace archive {

// Zip file reader
class IMQS_PAL_API ZipFile {
public:
	ZipFile();
	~ZipFile();

	void Close();

	// Open a zip file in memory
	Error OpenMem(const void* data, size_t len);

	// Get a list of all filenames inside the archive
	std::vector<std::string> Files() const { return Filenames; }

	// Read an entire file out of the archive
	Error ReadWholeFile(const std::string& name, std::string& content);

private:
	void*  MemBuf     = nullptr; // Copy of data from OpenMem()
	size_t MemBufSize = 0;

	std::vector<std::string> Filenames;

	void* MzStream = nullptr;
	void* MzZip    = nullptr;
};

} // namespace archive
} // namespace imqs