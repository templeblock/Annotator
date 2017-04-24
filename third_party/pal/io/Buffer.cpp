#include "pch.h"
#include "alloc.h"
#include "Buffer.h"

namespace imqs {
namespace io {

Buffer::Buffer() {
}

Buffer::~Buffer() {
	free(Buf);
}

Error Buffer::Write(const void* buf, size_t len) {
	Add(buf, len);
	return Error();
}

void Buffer::Add(const void* buf, size_t bytes) {
	Ensure(bytes);
	memmove(Buf + Len, buf, bytes);
	Len += bytes;
}

void Buffer::Ensure(size_t additionalBytes) {
	if (Len + additionalBytes <= Cap)
		return;

	size_t need   = Len + additionalBytes;
	size_t newCap = Cap * 2;
	if (newCap < 64)
		newCap = 64;
	while (newCap < need)
		newCap *= 2;

	void* newBuf = imqs_realloc_or_die(Buf, newCap);

	Buf = (uint8_t*) newBuf;
	Cap = newCap;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Error ByteReader::Read(void* buf, size_t& len) {
	if (Pos >= Len)
		return ErrEOF;
	len = std::min(len, Len - Pos);
	memcpy(buf, Buf + Pos, len);
	Pos += len;
	return Error();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Decoder::Decoder(io::Reader* reader) : Reader(reader) {
}

Decoder::Decoder(const void* buf, size_t len) {
	OwnReader = true;
	Reader    = new ByteReader(buf, len);
}

Decoder::~Decoder() {
	if (OwnReader)
		delete Reader;
}

Error Decoder::Read(void* buf, size_t& len) {
	Error err = Reader->Read(buf, len);
	Pos += len;
	return err;
}
}
}