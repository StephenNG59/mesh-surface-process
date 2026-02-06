/*********************************************
* 读取包含['V'], ['F'], ['K1']...的.npz文件;   *
* 重新进行UV-mapping;                         *
* 保存到['UV']中, 覆盖原先的UV坐标;             *
* 保存到新的.npz文件中.                        *
*********************************************/
// == std
#include <format>
// == geometry-central
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>
#include <geometrycentral/surface/scalar_fields.h>
#include <geometrycentral/surface/meshio.h>
//
#include <cnpy.h>
#include "log.h"
#include "cxxopts.hpp"

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

unique_ptr<ManifoldSurfaceMesh> gc_mesh;
unique_ptr<VertexPositionGeometry> gc_geom;

Eigen::MatrixXd V, N, UV;
Eigen::MatrixXi F;
Eigen::VectorXd K1, K2;
Eigen::VectorXi labels, instances;

// == Params
float regularize_lambda = 0.0;


vector<double> toStdVector(const Eigen::MatrixXd& mat) {
	vector<double> data(mat.size());
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data.data(), mat.rows(), mat.cols()) = mat;
	return data;
}

vector<int> toStdVector(const Eigen::MatrixXi& mat) {
	vector<int> data(mat.size());
	Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data.data(), mat.rows(), mat.cols()) = mat;
	return data;
}

vector<double> toStdVector(const Eigen::VectorXd& vec) {
	return vector<double>(vec.data(), vec.data() + vec.size());
}

vector<int> toStdVector(const Eigen::VectorXi& vec) {
	return vector<int>(vec.data(), vec.data() + vec.size());
}

void readFromNPZ(const string& filename) {
	auto load_matrixXd = [](const cnpy::NpyArray& arr) -> Eigen::MatrixXd {
		if (arr.word_size != sizeof(double)) {
			throw runtime_error("Expected double type for MatrixXd.");
		}
		size_t rows = arr.shape[0];
		size_t cols = arr.shape.size() > 1 ? arr.shape[1] : 1;
		Eigen::MatrixXd mat(rows, cols);
		const double* data = arr.data<double>();
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				mat(i, j) = data[i * cols + j];  // Numpy row-major layout
		return mat;
		};

	auto load_matrixXi = [](const cnpy::NpyArray& arr) -> Eigen::MatrixXi {
		if (arr.word_size != sizeof(int)) {
			throw runtime_error("Expected int type for MatrixXi.");
		}
		size_t rows = arr.shape[0];
		size_t cols = arr.shape.size() > 1 ? arr.shape[1] : 1;
		Eigen::MatrixXi mat(rows, cols);
		const int* data = arr.data<int>();
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
				mat(i, j) = data[i * cols + j];  // Numpy row-major layout
		return mat;
		};

	auto load_vectorXd = [](const cnpy::NpyArray& arr) -> Eigen::VectorXd {
		if (arr.word_size != sizeof(double)) {
			throw runtime_error("Expected double type for VectorXd.");
		}
		size_t len = arr.shape[0];
		Eigen::VectorXd vec(len);
		const double* data = arr.data<double>();
		for (size_t i = 0; i < len; ++i)
			vec(i) = data[i];
		return vec;
		};

	auto load_vectorXi = [](const cnpy::NpyArray& arr) -> Eigen::VectorXi {
		if (arr.word_size != sizeof(int)) {
			throw runtime_error("Expected int type for VectorXi.");
		}
		size_t len = arr.shape[0];
		Eigen::VectorXi vec(len);
		const int* data = arr.data<int>();
		for (size_t i = 0; i < len; ++i)
			vec(i) = data[i];
		return vec;
		};

	cnpy::npz_t npz = cnpy::npz_load(filename);

	V = load_matrixXd(npz["V"]);
	N = load_matrixXd(npz["N"]);
	F = load_matrixXi(npz["F"]);
	K1 = load_vectorXd(npz["K1"]);
	K2 = load_vectorXd(npz["K2"]);
	labels = load_vectorXi(npz["labels"]);
	instances = load_vectorXi(npz["instances"]);

	UV.resize(V.rows(), 2);
}

void writeToNPZ(const string& filename) {
	// ...
	vector<double> v_vec = toStdVector(V);
	vector<int>    f_vec = toStdVector(F);
	vector<double> n_vec = toStdVector(N);
	vector<double> uv_vec = toStdVector(UV);
	vector<double> k1_vec = toStdVector(K1);
	vector<double> k2_vec = toStdVector(K2);
	vector<int> labels_vec = toStdVector(labels);
	vector<int> instances_vec = toStdVector(instances);

	cnpy::npz_save(filename, "V", v_vec.data(), { static_cast<size_t>(V.rows()), static_cast<size_t>(V.cols()) }, "w");
	cnpy::npz_save(filename, "F", f_vec.data(), { static_cast<size_t>(F.rows()), static_cast<size_t>(F.cols()) }, "a");
	cnpy::npz_save(filename, "N", n_vec.data(), { static_cast<size_t>(N.rows()), static_cast<size_t>(N.cols()) }, "a");
	cnpy::npz_save(filename, "UV", uv_vec.data(), { static_cast<size_t>(UV.rows()), static_cast<size_t>(UV.cols()) }, "a");
	cnpy::npz_save(filename, "K1", k1_vec.data(), { static_cast<size_t>(k1_vec.size()) }, "a");
	cnpy::npz_save(filename, "K2", k2_vec.data(), { static_cast<size_t>(k2_vec.size()) }, "a");
	cnpy::npz_save(filename, "labels", labels_vec.data(), { static_cast<size_t>(labels.size()) }, "a");
	cnpy::npz_save(filename, "instances", instances_vec.data(), { static_cast<size_t>(instances.size()) }, "a");
}

void remapUV() {
	auto t0 = clock();

	// Init geometry-central
	try {
		gc_mesh = unique_ptr<ManifoldSurfaceMesh>(new ManifoldSurfaceMesh(F));
	}
	catch (const std::exception& e) {
		cout << e.what();
	}
	gc_geom = unique_ptr<VertexPositionGeometry>(new VertexPositionGeometry(*gc_mesh, V));
	LOG(LogLevel::trace, format("gc_geom init finished."));

	gc_geom->requireVertexNormals();
	gc_geom->requireFaceNormals();
	gc_geom->requireFaceGradientOfEigenVector3D();
	gc_geom->requireVertexGradientOfEigenVector3D();
	gc_geom->requireCotanLaplacian();
	LOG(LogLevel::trace, format("gc_geom requirements satisfied."));


	// Get origin gradients, and rotate 90 degree ccw
	VertexData<Vector3> v_grads(*gc_mesh);
	VertexData<Vector3> v_rotated_grads(*gc_mesh);
	for (Vertex v : gc_mesh->vertices()) {
		v_grads[v] = -gc_geom->vertexGradientOfEigenVector3D[v]; // 为了UV图方向和原来的较一致，加个负号旋转180度
		v_rotated_grads[v] = cross(gc_geom->vertexNormals[v], v_grads[v]);
	}
	auto t1 = clock();
	LOG(LogLevel::debug, format("Eigen vectors & gradients calculated. {}ms.", t1 - t0));

	// Compute scalar fields based on normalized gradients
	VertexData<double> x = computeSmoothestVertexScalarField(*gc_geom, v_grads, 0, regularize_lambda, true);
	LOG(LogLevel::trace, format("UV-x calculated."));
	VertexData<double> y = computeSmoothestVertexScalarField(*gc_geom, v_rotated_grads, 0, regularize_lambda, true);
	auto t2 = clock();
	LOG(LogLevel::debug, format("UV calculated. {}ms.", t2 - t1));
	gc_geom->unrequireCotanLaplacian();

	// UV.resize(...)
	for (Vertex v : gc_mesh->vertices()) {
		UV.row(v.getIndex()) << x[v], y[v], 0;
	}
}

int main(int argc, char* argv[]) {
	// options...
	string help_string("1.读取包含['V'], ['F'], ['K1']...的.npz文件;");
	help_string += "\n2.重新进行UV - mapping;";
	help_string += "\n3.保存到['UV']中, 覆盖原先的UV坐标;";
	help_string += "\n4.保存到新的.npz文件中.";
	cxxopts::Options options("RemapUV", help_string);
	options.add_options()
		("i, input", "Input .npz file", cxxopts::value<string>())
		("o, output", "Output .npz file", cxxopts::value<string>())
		("c, coeff", "Regularize coefficient", cxxopts::value<float>())
		("l, log-level", "Log level (error|warning|info|debug|trace)", cxxopts::value<string>()->default_value("debug"))
		("h, help", "Print help information");

	// == Parse args
	auto result = options.parse(argc, argv);
	// help
	if (argc < 2 || result.count("help")
		|| !result.count("input")
		|| !result.count("output")) {
		std::cout << options.help() << std::endl;
		return EXIT_SUCCESS;
	}
	// retrieve args
	if (result.count("coeff")) {
	    regularize_lambda = result["coeff"].as<float>();
	}
	string input_filename = result["input"].as<string>();
	string output_filename = result["output"].as<string>();
	// log level
	const string lvl = result["log-level"].as<string>();
	if (lvl == "error")   current_log_level() = LogLevel::error;
	else if (lvl == "warning") current_log_level() = LogLevel::warning;
	else if (lvl == "info")    current_log_level() = LogLevel::info;
	else if (lvl == "debug")   current_log_level() = LogLevel::debug;
	else if (lvl == "trace")   current_log_level() = LogLevel::trace;


	readFromNPZ(input_filename);
	LOG(LogLevel::trace, format("Read finished."));
	remapUV();
	LOG(LogLevel::trace, format("Remap finished."));
	writeToNPZ(output_filename);
	LOG(LogLevel::trace, format("Write finished."));

	return EXIT_SUCCESS;
}