#include "pch.h"
#include "Tensorflow.h"

#ifdef IMQS_TENSORFLOW
using namespace tensorflow;

namespace imqs {
namespace AI {

//void Model::Initialize(int& argc, char*** argv) {
void Model::Initialize() {
	//tensorflow::port::InitMain((*argv)[0], &argc, argv);

	// Creates a session.
	SessionOptions           options;
	std::unique_ptr<Session> session(NewSession(options));

	bool useGPU = true;

	Scope    root = Scope::NewRootScope();
	GraphDef def;
	root.ToGraphDef(&def);
	graph::SetDefaultDevice(useGPU ? "/gpu:0" : "/cpu:0", &def);

	auto s = session->Create(def);
	TF_CHECK_OK(s);

	Tensor              x(DT_FLOAT, TensorShape({2, 1}));
	std::vector<Tensor> outputs;
	TF_CHECK_OK(session->Run({{"x", x}}, {"y:0", "y_normalized:0"}, {}, &outputs));
}

} // namespace AI
} // namespace imqs

#endif // IMQS_TENSORFLOW
