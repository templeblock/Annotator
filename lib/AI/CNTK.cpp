#include "pch.h"
#include "CNTK.h"

using namespace std;
using namespace CNTK;

namespace imqs {
namespace AI {

static Library StaticLib;

void Library::Initialize() {
	try {
		auto dev = DeviceDescriptor::GPUDevice(0);
		Device   = new CNTK::DeviceDescriptor(dev);
	} catch (std::exception e) {
		auto dev = DeviceDescriptor::CPUDevice();
		Device   = new CNTK::DeviceDescriptor(dev);
	}
}

static void DumpVariable(const CNTK::Variable& v) {
	tsf::print("Name=%-15v Kind=%-3v Shape=%v\n", v.Name(), (int) v.Kind(), v.Shape().AsString());
}

static void DumpParameter(const CNTK::Parameter& v) {
	DumpVariable(v);
}

static void DumpModel(const CNTK::FunctionPtr& f) {
	tsf::print("Inputs:\n");
	for (const auto& v : f->Inputs()) {
		DumpVariable(v);
	}

	tsf::print("Outputs:\n");
	for (const auto& v : f->Outputs()) {
		DumpVariable(v);
	}
}
Error Model::Load(std::string filename) {
	if (!StaticLib.Device)
		StaticLib.Initialize();

	Func = CNTK::Function::Load(towide(filename), *StaticLib.Device);
	DumpModel(Func);

	// Model.Inputs contains not just the input we're interested in, but also all parameters, so we need to find the one real input.
	for (const auto& v : Func->Inputs()) {
		if (v.IsInput()) {
			Input = v;
			break;
		}
	}

	// We expect only one output
	Output        = Func->Output();
	auto outshape = Output.Shape().Dimensions();

	return Error();
}

size_t Model::OutputSize() {
	IMQS_ASSERT(Output.Shape().Dimensions().size() == 1);
	return Output.Shape().Dimensions()[0];
}

void Model::EvalRaw(CNTK::NDArrayViewPtr input, float* output) {
	std::unordered_map<Variable, ValuePtr> mInputs;
	std::unordered_map<Variable, ValuePtr> mOutputs;

	NDShape outputShape = {Output.Shape().Dimensions()[0], 1};

	ValuePtr inval  = MakeSharedObject<Value>(input);
	ValuePtr outval = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(DataType::Float, outputShape, DeviceDescriptor::CPUDevice()));

	mInputs.insert({Input, inval});
	mOutputs.insert({Output, outval});

	Func->Evaluate(mInputs, mOutputs, *StaticLib.Device);

	size_t nOutVals = outputShape.Dimensions()[0];
	memcpy(output, outval->Data()->DataBuffer<float>(), nOutVals * sizeof(output[0]));
}

NDArrayViewPtr Model::MakeRGBBuffer(const void* buf, int width, int height, int stride) {
	// I don't understand why cntk wants these extra 1-long dimensions. I can understand it, I guess, for the input,
	// because we'd have '3' there if it was an RGB image. But for the output, I would expect just a single dimension.
	NDShape shape = {3, (size_t) width, (size_t) height};

	auto v = MakeSharedObject<NDArrayView>(DataType::Float, shape, DeviceDescriptor::CPUDevice());

	const size_t outChan = 3;
	size_t bufBytes = width * height * outChan * sizeof(float);
	//float*  tmp      = (float*) imqs_malloc_or_die(bufBytes);
	float* tmp   = v->WritableDataBuffer<float>();
	float  scale = 1.0f / 255.0f;
	for (int y = 0; y < height; y++) {
		auto src = ((const uint8_t*) buf) + stride * y;
		auto dst = tmp + width * outChan * y;
		for (int x = 0; x < width; x++) {
			dst[0] = (float) src[0] * scale;
			dst[1] = (float) src[1] * scale;
			dst[2] = (float) src[2] * scale;
			dst += outChan;
			src += 4;
		}
	}
	//auto v = MakeSharedObject<NDArrayView>(DataType::Float, shape, tmp, bufBytes, DeviceDescriptor::CPUDevice());

	//free(tmp);
	return v;
}

void Model::EvalRGBA(const void* buf, int width, int height, int stride, float* output) {
	auto shape = Input.Shape().Dimensions();
	IMQS_ASSERT(width == shape[1]);
	IMQS_ASSERT(height == shape[2]);
	auto inPtr = MakeRGBBuffer(buf, width, height, stride);
	EvalRaw(inPtr, output);
}

int Model::EvalRGBAClassify(const void* buf, int width, int height, int stride) {
	auto               shape = Output.Shape().Dimensions();
	std::vector<float> output;
	output.resize(shape[0]);
	EvalRGBA(buf, width, height, stride, &output[0]);
	size_t imax = 0;
	float  vmax = -FLT_MAX;
	for (size_t i = 0; i < output.size(); i++) {
		if (output[i] > vmax) {
			imax = i;
			vmax = output[i];
		}
	}
	return (int) imax;
}

} // namespace AI
} // namespace imqs
