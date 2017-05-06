#pragma once

#include "../io/io.h"

namespace imqs {
namespace csv {

// Encoder encodes a single CSV cell.
class IMQS_PAL_API Encoder {
public:
	Error Write(io::Writer* w, const char* str, size_t len = -1);          // Write only the cell - do not add the comma
	Error WriteWithComma(io::Writer* w, const char* str, size_t len = -1); // Write the cell and add a comma afterwards

private:
	std::vector<char> CellBuf; // This avoids many tiny writes for escaped cells.
};

} // namespace csv
} // namespace imqs
