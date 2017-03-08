#ifdef _WIN32
#define IMQS_PAL_API __declspec(dllexport)
#else
#define IMQS_PAL_API
#endif

#include "common.h"
#include <utfz/utfz.h>
#include <curl/curl.h>
