#include "pch.h"
#include "LocalFileStorage.h"

namespace imqs {
namespace roadproc {

Error LocalFileStorage::Initialize(std::string rootDir) {
	RootDir = rootDir;
	return os::MkDirAll(RootDir);
}

Error LocalFileStorage::Create(std::string filename, FileStorageClass klass, const void* buf, size_t len) {
	os::File f;
	auto     err = f.Create(path::Join(RootDir, filename));
	if (!err.OK())
		return err;
	return f.Write(buf, len);
}

Error LocalFileStorage::Open(std::string filename, io::Reader*& reader) {
	auto f   = new os::File();
	auto err = f->Open(path::Join(RootDir, filename));
	if (!err.OK()) {
		delete f;
		return err;
	}
	reader = f;
	return Error();
}

} // namespace roadproc
} // namespace imqs