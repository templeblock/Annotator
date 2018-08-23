#include "pch.h"
#include "GCSStorage.h"

/*

API keys don't seem to work for this API. I tried, but it told me that my IP was invalid, which seems like
a bogus error.

So.. you need an OAuth2 token. How do you acquire one in C++?
Like this: https://stackoverflow.com/questions/41648094/what-authorization-token-is-required-for-google-cloud-storage-json-api
Basically, I've just hacked it out from the gcloud util.
Not sure how we're going to do this in production. Maybe shell out to a little Go program or something.

	gcloud auth activate-service-account roadprocessor-gcs@roads-188714.iam.gserviceaccount.com --key-file=roads-188714-b0f74e93342d.json
	gcloud auth print-access-token

*/

using namespace std;
using namespace nlohmann;

namespace imqs {
namespace roadproc {

GCSStorage::~GCSStorage() {
	IsDying = 1;
	if (WriteThread.joinable())
		WriteThread.join();
}

Error GCSStorage::Initialize(std::string bucketName, std::string apiKey) {
	IsDying     = 0;
	BucketName  = bucketName;
	APIKey      = apiKey;
	WriteThread = std::thread(WriteThreadFuncWrapper, this);
	return Error();
}

Error GCSStorage::Create(std::string filename, FileStorageClass klass, const void* buf, size_t len) {
	{
		lock_guard<mutex> lock(LastErrorLock);
		if (!LastError.OK())
			return LastError;
	}

	for (int delay = 1; true; delay = min(delay + 1, 5)) {
		QueueLock.lock();
		if (Queue.size() < MaxQueueSize) {
			QueueLock.unlock();
			break;
		}
		tsf::print("Writer queue is full (%v). Waiting...\n", Queue.size());
		QueueLock.unlock();
		os::Sleep(delay * time::Second);
	}

	CreateItem ci;
	ci.Filename = filename;
	ci.Class    = klass;
	ci.Data.assign((const char*) buf, len);
	QueueLock.lock();
	Queue.push_back(std::move(ci));
	QueueLock.unlock();
	return LastError;
}

Error GCSStorage::Open(std::string filename, io::Reader*& reader) {
	return os::ErrENOENT;
	//auto queryParams = url::Encode({{"key", APIKey}});
	auto queryParams = "";
	auto encodedName = url::Encode(MakeFullname(filename));
	auto rUrl        = tsf::fmt("https://www.googleapis.com/storage/v1/b/%v/o/%v?alt=media&", BucketName, encodedName) + queryParams;
	auto req         = http::Request::GET(rUrl);
	req.SetHeader("Authorization", "Bearer " + APIKey);
	auto resp = ReadClient.Perform(req);
	if (!resp.Is200()) {
		if (DebugMessages)
			tsf::print("Open(%v) failed: %v %v\n", filename, resp.StatusCodeStr(), resp.Body);
		if (resp.StatusCodeInt() == 404)
			return os::ErrENOENT;
		//tsf::print("%v\n", APIKey);
		//tsf::print("%v\n", rUrl);
		//tsf::print("%v %v\n", resp.StatusCodeStr(), resp.Body);
		return resp.ToError();
	}
	reader = new io::StringReader(std::move(resp.Body));
	return Error();
}

std::string GCSStorage::MakeFullname(std::string filename) {
	// The GCS docs recommend using a well distributed prefix for the object names, so that
	// objects end up hitting a good distribution of index servers, so that is why we
	// prepend a hash to the object name.
	uint32_t hash = XXH32(filename.c_str(), filename.size(), 0);
	return tsf::fmt("%04x-%v", hash & 0xffff, filename);
}

void GCSStorage::WriteThreadFuncWrapper(GCSStorage* self) {
	self->WriteThreadFunc();
}

void GCSStorage::WriteThreadFunc() {
	http::Connection httpClient;
	int              idle = 0;
	while (true) {
		if (IsDying != 0)
			break;
		CreateItem ci;
		{
			lock_guard<mutex> lock(QueueLock);
			if (Queue.size() != 0) {
				ci = Queue.front();
				Queue.erase(Queue.begin());
			}
		}
		// empty filename = empty queue
		if (ci.Filename == "") {
			os::Sleep(500 * time::Millisecond);
			continue;
		}

		Error err;
		for (int attempt = 0; attempt < 5; attempt++) {
			err = WriteThreadFunc_WriteItem(httpClient, ci);
			if (err.OK())
				break;
			os::Sleep((1 + (1 << attempt)) * time::Second);
		}
		if (!err.OK()) {
			lock_guard<mutex> lock(LastErrorLock);
			LastError = err;
		}
	}
}

Error GCSStorage::WriteThreadFunc_WriteItem(http::Connection& httpClient, CreateItem& item) {
	auto fullname = MakeFullname(item.Filename);

	string mime = "";
	if (item.Filename.find(".jpeg") != -1)
		mime = "image/jpeg";
	else if (item.Filename.find(".png") != -1)
		mime = "image/png";
	else if (item.Filename.find(".lz4") != -1)
		mime = "image/imqs-roads-lz4";
	else
		return Error::Fmt("Unrecognized content type: %v", item.Filename);

	//auto queryParams = url::Encode({{"name", fullname}, {"key", APIKey}});
	auto queryParams = url::Encode({{"name", fullname}});
	auto rUrl        = tsf::fmt("https://www.googleapis.com/upload/storage/v1/b/%v/o?uploadType=media&", BucketName) + queryParams;
	auto req         = http::Request::POST(rUrl);
	req.Body         = item.Data;
	req.SetHeader("Authorization", "Bearer " + APIKey);
	req.SetHeader("Content-Type", mime);
	req.SetHeader("Content-Length", ItoA(item.Data.size()));
	if (DebugMessages)
		tsf::print("Starting upload...\n");
	auto resp = httpClient.Perform(req);
	if (DebugMessages)
		tsf::print("Create(%v, %v KB): %v %v\nURL: %v", item.Filename, item.Data.size() / 1024, resp.StatusCodeStr(), resp.Body, rUrl);
	return resp.ToError();
}

} // namespace roadproc
} // namespace imqs