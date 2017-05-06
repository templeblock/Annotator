#include "pch.h"
#include "csv.h"

namespace imqs {
namespace csv {

Error Encoder::Write(io::Writer* w, const char* str, size_t len) {
	bool needEscape = false;
	if (len == -1)
		len = strlen(str);
	for (size_t i = 0; i < len; i++) {
		if (str[i] == ',' || str[i] == '"' || str[i] == 10 || str[i] == 13) {
			needEscape = true;
			break;
		}
	}
	if (!needEscape)
		return w->Write(str, len);

	CellBuf.resize(0);
	CellBuf.push_back('"');
	for (size_t i = 0; i < len; i++) {
		if (str[i] == '"') {
			CellBuf.push_back('"');
			CellBuf.push_back('"');
		} else {
			CellBuf.push_back(str[i]);
		}
	}
	CellBuf.push_back('"');
	return w->Write(&CellBuf[0], CellBuf.size());
}

Error Encoder::WriteWithComma(io::Writer* w, const char* str, size_t len) {
	auto err = Write(w, str, len);
	if (err.OK())
		err = w->Write(",", 1);
	return err;
}

} // namespace csv
} // namespace imqs
