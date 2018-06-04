#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#define _CRT_SECURE_NO_WARNINGS 1

#include <lib/pal/pal.h>
#include "common.h"

#ifdef _MSC_VER
#define IMQS_AI_API __declspec(dllexport)
#else
#define IMQS_AI_API
#endif

#ifdef IMQS_TENSORFLOW
// We REALLY want to keep the Tensorflow dependency chain out of our downstream projects
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244 4267 4554 4800)
#endif

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif // IMQS_TENSORFLOW
