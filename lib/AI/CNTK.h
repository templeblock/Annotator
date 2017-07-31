#pragma once

namespace imqs {
namespace AI {

class IMQS_AI_API Library {
public:
	CNTK::DeviceDescriptor* Device = nullptr;

	void Initialize();
};

class IMQS_AI_API Model {
public:
	CNTK::FunctionPtr Func;
	CNTK::Variable    Input;
	CNTK::Variable    Output;

	Error                Load(std::string filename);
	size_t               OutputSize(); // Size of first (and only) dimension of output. Asserts if output has more than 1 dimension.
	CNTK::NDArrayViewPtr MakeRGBBuffer(const void* buf, int width, int height, int stride);
	void                 EvalRGBA(const void* buf, int width, int height, int stride, float* output);
	int                  EvalRGBAClassify(const void* buf, int width, int height, int stride);
	void                 EvalRaw(CNTK::NDArrayViewPtr input, float* output);
};

} // namespace AI
} // namespace imqs
