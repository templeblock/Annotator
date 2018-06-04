#pragma once

#include "common.h"

#ifdef _WIN32
#define IMQS_TRAIN_API __declspec(dllimport)
#else
#define IMQS_TRAIN_API
#endif

#include "Exporter.h"
#include "LabelIO.h"