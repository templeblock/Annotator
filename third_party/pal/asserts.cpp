#include "pch.h"
#include "asserts.h"

namespace imqs {

void IMQS_NORETURN Die(const char* file, int line, const char* msg) {
	fprintf(stdout, "Program is self-terminating at %s:%d. (%s)\n", file, line, msg);
	fprintf(stderr, "Program is self-terminating at %s:%d. (%s)\n", file, line, msg);
	fflush(stdout);
	fflush(stderr);
	IMQS_DEBUG_BREAK();
	*((int*) 0) = 1;
}
}