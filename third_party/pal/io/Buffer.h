#pragma once

#include "../io/io.h"

namespace imqs {
namespace io {

// Memory buffer
class IMQS_PAL_API Buffer : public Writer {
public:
	uint8_t* Buf = nullptr;
	size_t   Cap = 0;
	size_t   Len = 0;

	Buffer();
	~Buffer();

	Error Write(const void* buf, size_t len) override;

	void     Add(const void* buf, size_t bytes);
	void     Ensure(size_t additionalBytes);        // If necessary, grow to accommodate an additional number of bytes
	size_t   Remain() const { return Cap - Len; }   // Number of bytes available in buffer
	uint8_t* WritePos() const { return Buf + Len; } // Write position
};

// Wraps a memory buffer in a Reader
class IMQS_PAL_API ByteReader : public Reader {
public:
	const uint8_t* Buf = nullptr;
	size_t         Len = 0;
	size_t         Pos = 0;

	ByteReader() {}
	ByteReader(const void* buf, size_t len) : Buf((const uint8_t*) buf), Len(len) {}

	Error Read(void* buf, size_t& len) override;
};

// Helper that is made for decoding binary streams
class IMQS_PAL_API Decoder {
public:
	Decoder(io::Reader* reader);
	Decoder(const void* buf, size_t len);
	~Decoder();

	int8_t   ReadInt8() { return ReadT<int8_t>(); }
	uint8_t  ReadUint8() { return ReadT<uint8_t>(); }
	int16_t  ReadInt16() { return ReadT<int16_t>(); }
	uint16_t ReadUint16() { return ReadT<uint16_t>(); }
	int32_t  ReadInt32() { return ReadT<int32_t>(); }
	uint32_t ReadUint32() { return ReadT<uint32_t>(); }
	int64_t  ReadInt64() { return ReadT<int64_t>(); }
	uint64_t ReadUint64() { return ReadT<uint64_t>(); }
	char     ReadChar() { return ReadT<char>(); }
	float    ReadFloat() { return ReadT<float>(); }
	double   ReadDouble() { return ReadT<double>(); }

	Error Read(void* buf, size_t& len);

	template <typename V>
	V ReadT() {
		V      v   = 0;
		size_t len = sizeof(v);
		Reader->Read(&v, len); // discard error
		Pos += len;
		return v;
	}

	// Returns the number of bytes decoded
	int64_t Position() const { return Pos; }

private:
	bool    OwnReader = false;
	Reader* Reader    = nullptr;
	int64_t Pos       = 0;
};
}
}