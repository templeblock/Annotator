#include "pch.h"
#include "Session.h"

using namespace std;

namespace imqs {
namespace frameserver {

SessionStore::SessionStore(std::string rootPath, uberlog::Logger* log) : RootPath(rootPath), Log(log) {
}

SessionStore::~SessionStore() {
	Sessions.clear();
}

Error SessionStore::CreateSession(std::string path, std::string& id) {
	auto ses      = make_shared<Session>();
	auto fullpath = path::SafeJoin(RootPath, path);
	auto err      = ses->Video.OpenFile(fullpath);
	if (!err.OK())
		return err;

	err = train::LoadVideoLabels(fullpath, ses->Labels);
	if (!err.OK() && !os::IsNotExist(err))
		return err;

	char buf[20];
	crypto::RandomBytes(buf, sizeof(buf));
	id = strings::ToHex(buf, sizeof(buf));

	Item it;
	it.LastUsed = time::Now();
	it.Session  = ses;

	Lock.lock();
	PurgeExpiredSessions_Internal();
	Sessions.insert(id, it);
	Lock.unlock();

	Log->Info("Created session %v: %v (%v labeled frames)", id, path, ses->Labels.Frames.size());

	return Error();
}

SessionPtr SessionStore::GetSession(std::string id) {
	lock_guard<mutex> lock(Lock);
	auto              it = Sessions.getp(id);
	if (!it)
		return nullptr;

	it->LastUsed = time::Now();
	return it->Session;
}

void SessionStore::PurgeExpiredSessions() {
	lock_guard<mutex> lock(Lock);
	PurgeExpiredSessions_Internal();
}

// assume lock is held
void SessionStore::PurgeExpiredSessions_Internal() {
	// Build a new list that contains only the non-expired sessions.
	// Let shared_ptr take care of the actual deleting of the session object.
	// By using shared_ptr, somebody who is using an expired video will still be fine.
	decltype(Sessions) cleaned;
	auto               now = time::Now();
	for (auto it : Sessions) {
		if (now - it.second.LastUsed < ExpiryTimeout)
			cleaned.insert(it.first, it.second);
	}
	Sessions = cleaned;
}

} // namespace frameserver
} // namespace imqs