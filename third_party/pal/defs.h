#pragma once

// Found this in the Chrome sources, via a PVS studio blog post
template <typename T, size_t N>
char (&ArraySizeHelper(T (&array)[N]))[N];
#define arraysize(array) (sizeof(ArraySizeHelper(array)))

#if defined(_MSC_VER)
typedef SSIZE_T ssize_t;
#endif

#define IMQS_ENDIAN_LITTLE 1

#ifdef _MSC_VER
#define IMQS_NORETURN __declspec(noreturn)
#else
#define IMQS_NORETURN __attribute__((noreturn)) __attribute__((analyzer_noreturn))
#endif

#ifdef _MSC_VER
#define IMQS_NOINLINE __declspec(noinline)
#else
#define IMQS_NOINLINE __attribute__((noinline))
#endif

#ifdef _MSC_VER
#define IMQS_DEBUG_BREAK() __debugbreak()
#else
#define IMQS_DEBUG_BREAK() __builtin_trap()
#endif

#ifdef _MSC_VER
#define IMQS_ANALYSIS_ASSUME(expr) __analysis_assume(expr)
#else
#define IMQS_ANALYSIS_ASSUME(expr)
#endif
