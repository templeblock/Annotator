#pragma once

#include "FileStorage.h"

namespace imqs {
namespace roadproc {

class LocalFileStorage : public IFileStorage {
public:
	std::string RootDir;

	Error Initialize(std::string rootDir);

	Error Create(std::string filename, FileStorageClass klass, const void* buf, size_t len) override;
	Error Open(std::string filename, io::Reader*& reader) override;
};

} // namespace roadproc
} // namespace imqs