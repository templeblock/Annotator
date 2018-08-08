#pragma once

namespace imqs {
namespace roadproc {

enum class FileStorageClass {
	MultiRegional,
	Regional,
	Nearline,
	Coldline,
};

// Interface to a file storage system
// We abstract the notion of a file, so that we can use GCS in the cloud, or a plain old
// file system when developing locally.
class IFileStorage {
public:
	virtual ~IFileStorage() {}
	virtual Error Create(std::string filename, FileStorageClass klass, const void* buf, size_t len) = 0;
	virtual Error Open(std::string filename, io::Reader*& reader)                                   = 0;
};

} // namespace roadproc
} // namespace imqs