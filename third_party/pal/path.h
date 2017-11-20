#pragma once

#include "Error.h"
#include "std_utils.h"

namespace imqs {

// Functions for manipulating paths.
// On Windows, separators are forward slashes or backslashes.
// On Unix, separators are forward slashes only.
namespace path {

IMQS_PAL_API bool IsAnySeparator(int ch);                   // Returns true for forward or backslash (same behaviour on all platforms).
IMQS_PAL_API bool IsSeparator(int ch);                      // On Windows, returns true for forward or backslash. On Unix, returns true for forward slash.
IMQS_PAL_API std::string Dir(const std::string& path);      // Return the parent directory of the given file or directory
IMQS_PAL_API std::string Filename(const std::string& path); // Return the filename portion of the path, which is everything after the last slash, or the entire path, if no slashes exist.

// Split a path into directory and filename. If the path contains no slashes, then dir is empty, and filename equals path.
// There is an exception to that rule - which is for Windows paths that are just a directory name. If path is "c:", then dir is set to "c:", and
// filename is empty.
// 'dir' and 'filename' can always be recombined to produce 'path'.
IMQS_PAL_API void SplitDir(const std::string& path, std::string& dir, std::string& filename);

// Splits into two parts, such that name contains everything before the last dot, and ext contains the last dot, plus everything after it.
// If there is no dot in the filename, then ext is empty.
// The original string is always name + ext.
IMQS_PAL_API void SplitExt(const std::string& path, std::string& name, std::string& ext);

// Return the extension only, according to the rules of SplitExt
IMQS_PAL_API std::string Extension(const std::string& path);

// Use SplitExt to change the extension. To remove an extension, make newExt = ""
IMQS_PAL_API std::string ChangeExtension(const std::string& path, const std::string& newExt);

// Join two or more paths together with slashes
IMQS_PAL_API std::string Join(size_t n, const std::string** parts);

// Join two or more paths together with slashes
IMQS_PAL_API std::string Join(const std::string& a, const std::string& b = "", const std::string& c = "", const std::string& d = "", const std::string& e = "");

// Join two or more paths together with slashes. However, if any components contains a "..", then return 'a'
IMQS_PAL_API std::string SafeJoin(size_t n, const std::string** parts);

// Join two or more paths together with slashes. However, if any components contains a "..", then return 'a'
IMQS_PAL_API std::string SafeJoin(size_t n, const std::string* parts);

// Join two or more paths together with slashes. However, if any components contains a "..", then return 'a'
IMQS_PAL_API std::string SafeJoin(const std::string& a, const std::string& b = "", const std::string& c = "", const std::string& d = "", const std::string& e = "");

} // namespace path
} // namespace imqs