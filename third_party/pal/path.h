#pragma once

#include "Error.h"

namespace imqs {

// Functions for manipulating paths.
// On Windows, separators are forward slashes or backslashes.
// On Unix, separators are forward slashes only.
namespace path {

IMQS_PAL_API bool IsSeparator(int ch);                      // On Windows, returns true for forward or backslash. On Unix, returns true for forward slash.
IMQS_PAL_API std::string Dir(const std::string& path);      // Return the parent directory of the given file or directory
IMQS_PAL_API std::string Filename(const std::string& path); // Return the filename portion of the path, which is everything after the last slash, or the entire path, if no slashes exist.
IMQS_PAL_API std::string Join(const std::string& a, const std::string& b,
                              const std::string& c = "", const std::string& d = ""); // Join two or more paths together with slashes
} // namespace path
} // namespace imqs