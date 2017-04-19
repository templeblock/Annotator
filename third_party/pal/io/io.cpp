#include "pch.h"
#include "../io/io.h"

namespace imqs {

IMQS_PAL_API StaticError ErrEOF("EOF");

namespace io {
Writer::~Writer() {}
Reader::~Reader() {}
}
}