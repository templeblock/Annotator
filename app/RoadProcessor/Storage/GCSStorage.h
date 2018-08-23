#pragma once

#include "FileStorage.h"

namespace imqs {
namespace roadproc {

class GCSStorage : public IFileStorage {
public:
	std::string BucketName;
	std::string APIKey;
	bool        DebugMessages = true;
	size_t      MaxQueueSize  = 200; // Once writer queue reaches this size, we stall on Create()

	~GCSStorage() override;

	Error Initialize(std::string bucketName, std::string apiKey);

	Error Create(std::string filename, FileStorageClass klass, const void* buf, size_t len) override;
	Error Open(std::string filename, io::Reader*& reader) override;

private:
	struct CreateItem {
		std::string      Filename;
		FileStorageClass Class;
		std::string      Data;
	};
	http::Connection        ReadClient;
	std::mutex              LastErrorLock; // Guards access to LastError
	Error                   LastError;
	std::atomic<int32_t>    IsDying;
	std::thread             WriteThread;
	std::mutex              QueueLock; // Guards access to Queue
	std::vector<CreateItem> Queue;

	static std::string MakeFullname(std::string filename);
	static void        WriteThreadFuncWrapper(GCSStorage* self);
	void               WriteThreadFunc();
	Error              WriteThreadFunc_WriteItem(http::Connection& httpClient, CreateItem& item);
};

} // namespace roadproc
} // namespace imqs