#pragma once

#include "common.h"

#ifdef _MSC_VER
#define IMQS_AI_API __declspec(dllimport)
#else
#define IMQS_AI_API
#endif

#include "Tensorflow.h"