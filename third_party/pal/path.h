#pragma once

#include "Error.h"

namespace imqs {

// Functions for manipulating paths.
// On Windows, separators are forward slashes or backslashes.
// On Unix, separators are forward slashes only.
namespace path {

IMQS_PAL_API bool IsSeparator(int ch);          // On Windows, returns true for forward or backslash. On Unix, returns true for forward slash.
IMQS_PAL_API std::string Dir(const char* path); // Return the parent directory of the given file or directory
}
}