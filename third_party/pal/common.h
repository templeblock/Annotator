// #includes common to the pal library itself, and external users of pal

#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#define _CRT_SECURE_NO_WARNINGS 1

///////////////////////////////////////////////////////////////////////////////////
// Windows 10 SDK 10240 has some /analyze warnings inside it's headers (BEGIN)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 28020)
#endif
///////////////////////////////////////////////////////////////////////////////////

#include <string.h>
#include <assert.h>
#include <time.h>

#include <memory>
#include <string>
#include <atomic>

#ifdef _WIN32
#include <BaseTsd.h>
#include <WinSock2.h>
#include <windows.h>
#endif

///////////////////////////////////////////////////////////////////////////////////
// Windows 10 SDK 10240 has some /analyze warnings inside it's headers (END)
#ifdef _MSC_VER
#pragma warning(pop)
#endif
///////////////////////////////////////////////////////////////////////////////////

#include <tsf/tsf.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6313 6282)
#endif

#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/allocators.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/error/en.h>
#include <rapidjson/schema.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif


#include "defs.h"
#include "asserts.h"
#include "Error.h"

#include "hash/siphash.h"
#include "hash/xxhash.h"

#include "containers/ohashmap.h"
#include "containers/ohashset.h"
#include "std_utils.h"
