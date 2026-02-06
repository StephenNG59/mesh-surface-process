// == geometry-central
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>
#include <geometrycentral/surface/scalar_fields.h>
#include <geometrycentral/surface/meshio.h>
// == vcglib
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/curvature_fitting.h>
// == std
#include <format>
#include <iostream>
#include <filesystem>
#include "cxxopts.hpp"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"
#include "json.hpp"


using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Vcg MyMesh structure
class MyVertex;
class MyFace;
class MyEdge;
struct MyUsedTypes : public vcg::UsedTypes<
	vcg::Use<MyVertex>     ::AsVertexType,
	vcg::Use<MyFace>       ::AsFaceType,
	vcg::Use<MyEdge>       ::AsEdgeType> {};
class  MyVertex : public vcg::Vertex<
	MyUsedTypes,
	vcg::vertex::Coord3f,
	vcg::vertex::Normal3f,
	vcg::vertex::VFAdj,
	vcg::vertex::VEAdj,
	vcg::vertex::CurvatureDirf,
	vcg::vertex::Qualityf,
	vcg::vertex::Mark,
	vcg::vertex::BitFlags> {};
class  MyFace : public vcg::Face<
	MyUsedTypes,
	vcg::face::Normal3f,
	vcg::face::VFAdj,
	vcg::face::FFAdj,
	vcg::face::VertexRef,
	vcg::face::Mark,
	vcg::face::BitFlags> {};
class  MyEdge : public vcg::Edge<
	MyUsedTypes,
	vcg::edge::VEAdj,
	vcg::edge::VertexRef,
	vcg::edge::Mark,
	vcg::edge::BitFlags> {};
class  MyMesh : public vcg::tri::TriMesh<
	std::vector<MyVertex>,
	std::vector<MyFace>,
	std::vector<MyEdge> > {};

// == Global
string output_name;
unique_ptr<MyMesh> vcg_mesh;
unique_ptr<ManifoldSurfaceMesh> gc_mesh;
unique_ptr<VertexPositionGeometry> gc_geom;
unique_ptr<VertexPositionGeometry> gc_geom2d;
Eigen::MatrixXf V;
Eigen::MatrixXi F;
Eigen::VectorXi face_labels;
Eigen::MatrixXf XY;
Eigen::VectorXf vK1, vK2;

// == Params
float regularize_lambda = 0.0;
float p_lower_bound = 0.005, p_upper_bound = 0.995;
int width = 1024, height = 256;
bool debug = false;

bool loadMyMesh(const std::string& mesh_filepath) {
	using vcg::tri::io::Importer;

	// Import mesh
	vcg_mesh = make_unique<MyMesh>();
	int result = Importer<MyMesh>::Open(*vcg_mesh, mesh_filepath.c_str());
	if (result != Importer<MyMesh>::E_NOERROR) {
		cerr << "Error loading mesh: " << Importer<MyMesh>::ErrorMsg(result) << endl;
		return false;
	}

	//name = mesh_filepath;
	if (debug)
		std::cout << std::format("{} sucessfully loaded with {} faces & {} verts.\n",
		mesh_filepath, vcg_mesh->FN(), vcg_mesh->VN());

	return true;
}

bool loadLabel(const std::string& label_filepath) {
	// Default with all zeros
	Eigen::VectorXi label_vec = Eigen::VectorXi::Zero(vcg_mesh->FN());

	// Load labels if file exists
	std::ifstream f(label_filepath);
	if (f.is_open()) {
		int lbl, i = 0;
		while (f >> lbl) {
			label_vec[i] = lbl;
			i++;
		}

		if (i != vcg_mesh->FN()) {
			std::cerr << std::format("[Error] Label size {} not matching face number {}.\n", i, vcg_mesh->FN());
			return false;
		}
	}

	// Add vcg attribute
	MyMesh::PerFaceAttributeHandle<int> label_handle =
		vcg::tri::Allocator<MyMesh>::AddPerFaceAttribute<int>(*vcg_mesh, "face_labels");
	for (int i = 0; i < vcg_mesh->FN(); i++) {
		label_handle[i] = label_vec[i];
	}

	return true;
}

bool repairMesh() {
	using namespace vcg::tri;

	// Merge duplicated verts
	int removed_verts = Clean<MyMesh>::MergeCloseVertex(*vcg_mesh, 1e-6);
	if (removed_verts > 0) {
		if (debug) std::cout << std::format("Merged {} close vertices.\n", removed_verts);
	}

	// Re-index and compact vertex list
	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	//Clean<MyMesh>::RemoveUnreferencedVertex(*vcg_mesh);

	// === Ensure Manifold & Update Topology ===
	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);

	// "Repair non Manifold Vertices by splitting"
	int num = Clean<MyMesh>::SplitNonManifoldVertex(*vcg_mesh, 0);
	if (debug) cout << format("Splited {} non-manifold vertices.\n", num);
	// "Remove Isolated pieces (wrt Diameter)"
	pair<int, int> comp = Clean<MyMesh>::RemoveSmallConnectedComponentsDiameter(*vcg_mesh, 0.05 * vcg_mesh->bbox.Diag());
	if (debug) cout << format("Removed {} out of all {} components.\n", comp.second, comp.first);
	// "Repair non Manifold Edges"
	num = Clean<MyMesh>::RemoveNonManifoldFace(*vcg_mesh);
	if (debug) cout << format("Removed {} non-manifold faces.\n", num);
	// "Remove Unreferenced Vertices"
	num = Clean<MyMesh>::RemoveUnreferencedVertex(*vcg_mesh);
	if (debug) cout << format("Removed {} unreferenced vertices.\n", num);
	if (debug) cout << std::format("{} faces & {} vertices after cleaning.\n", vcg_mesh->FN(), vcg_mesh->VN());

	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexEdge(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);
	// Normals - used by curvature
	UpdateNormal<MyMesh>::PerVertexAngleWeighted(*vcg_mesh);
	UpdateNormal<MyMesh>::PerFaceNormalized(*vcg_mesh);

	return true;
}

bool initGeometrycentral() {
	using namespace vcg::tri;

	// Get V, F & label
	V = Eigen::MatrixXf(vcg_mesh->VN(), 3);
	F = Eigen::MatrixXi(vcg_mesh->FN(), 3);
	face_labels = Eigen::VectorXi(vcg_mesh->FN());
	// Normals
	UpdateNormal<MyMesh>::PerVertexAngleWeighted(*vcg_mesh);
	UpdateNormal<MyMesh>::PerFaceNormalized(*vcg_mesh);

	// Face handle for labels
	MyMesh::PerFaceAttributeHandle<int> label_handle =
		Allocator<MyMesh>::FindPerFaceAttribute<int>(*vcg_mesh, "face_labels");

	int vi = 0, fi = 0;
	for (const auto& v : vcg_mesh->vert) {
		if (!v.IsD()) {
			V.row(vi) = Eigen::Vector3f(v.P()[0], v.P()[1], v.P()[2]);
			vi++;
		}
	}
	for (const auto& f : vcg_mesh->face) {
		if (!f.IsD()) {
			F.row(fi) = Eigen::Vector3i(
				Index(*vcg_mesh, f.V(0)),
				Index(*vcg_mesh, f.V(1)),
				Index(*vcg_mesh, f.V(2)));
			face_labels[fi] = label_handle[fi];
			fi++;
		}
	}

	// Init geometrycentral
	gc_mesh = unique_ptr<ManifoldSurfaceMesh>(new ManifoldSurfaceMesh(F));
	gc_geom = unique_ptr<VertexPositionGeometry>(new VertexPositionGeometry(*gc_mesh, V/*, face_labels*/));

	return true;
}

void saveNewMesh() {
	namespace fs = filesystem;

	fs::path obj_fn = output_name + ".obj";

	vcg::tri::io::Exporter<MyMesh>::Save(*vcg_mesh, (obj_fn).string().c_str());
}

void updateAndSaveFaceLabels() {
	namespace fs = filesystem;

	face_labels = Eigen::VectorXi(vcg_mesh->FN());
	vector<int> label_vec;

	MyMesh::PerFaceAttributeHandle<int> label_handle = vcg::tri::Allocator<MyMesh>::FindPerFaceAttribute<int>(*vcg_mesh, "face_labels");
	for (int i = 0; i < vcg_mesh->FN(); i++) {
		face_labels(i) = label_handle[i];
		label_vec.push_back(label_handle[i]);
	}

	nlohmann::json json_data;
	json_data["labels"] = label_vec;

	// save to json file
	fs::path json_fn = output_name + ".json";
	ofstream f(json_fn);
	f << json_data.dump();
	f.close();

	return;
}

void map2xy() {
	gc_geom->requireVertexNormals();
	gc_geom->requireFaceNormals();
	gc_geom->requireVertexGradientOfEigenVector3D();

	// Get origin gradients, and rotate 90 degree ccw
	VertexData<Vector3> v_grads(*gc_mesh);
	VertexData<Vector3> v_rotated_grads(*gc_mesh);
	for (Vertex v : gc_mesh->vertices()) {
		v_grads[v] = gc_geom->vertexGradientOfEigenVector3D[v];
		v_rotated_grads[v] = cross(gc_geom->vertexNormals[v], v_grads[v]);
	}

	// Compute scalar fields based on normalized gradients
	VertexData<double> x = computeSmoothestVertexScalarField(
		*gc_geom, v_grads, 0, regularize_lambda, true);
	VertexData<double> y = computeSmoothestVertexScalarField(
		*gc_geom, v_rotated_grads, 0, regularize_lambda, true);
	XY = Eigen::MatrixXf(gc_mesh->nVertices(), 2);
	for (Vertex v : gc_mesh->vertices()) {
		XY.row(v.getIndex()) << float(x[v]), float(y[v]);
	}
}

void clampToPercentile(Eigen::VectorXf& vec, float low_perc, float up_perc) {
	// Copy and sort
	vector<float> sorted_vec(vec.data(), vec.data() + vec.size());
	sort(sorted_vec.begin(), sorted_vec.end());

	// Get index
	size_t low_idx = static_cast<size_t>(low_perc * sorted_vec.size());
	size_t up_idx = static_cast<size_t>(up_perc * sorted_vec.size()) - 1;

	// Get threshold
	float low_bound = sorted_vec[low_idx];
	float up_bound = sorted_vec[up_idx];

	// Clamp vector
	vec = vec.unaryExpr([low_bound, up_bound](float val) {
		return std::clamp(val, low_bound, up_bound);
		});
}

void computeMeanCurvaturesAndClamp() {
	using namespace vcg::tri;

	// Calculate
	UpdateNormal<MyMesh>::NormalizePerVertex(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);
	double start = clock();
	UpdateCurvatureFitting<MyMesh>::updateCurvatureLocal(*vcg_mesh, 0.01f * vcg_mesh->bbox.Diag());
	//cout << format("{}ms cost by `updateCurvatureLocal()`\n", clock() - start);

	// Retrieve results
	int vn = vcg_mesh->VN(), i = 0;
	//vK1.resize(vn);
	vK2.resize(vn);
	ForEachVertex(*vcg_mesh, [&i](const MyMesh::VertexType& v) {
		//vK1(i) = v.cK1();  // max curv
		vK2(i) = v.cK2();  // min curv
		i++;
		});

	// Clamp by percentile
	//clampToPercentile(vK1, p_lower_bound, p_upper_bound);
	clampToPercentile(vK2, p_lower_bound, p_upper_bound);
}

void generateImageAndAnnotations() {
	// Translate and scale XY to image's width x height
	float trans_x = -(XY.col(0).minCoeff());
	float trans_y = -(XY.col(1).minCoeff());
	float scale_x = width / (XY.col(0).maxCoeff() - XY.col(0).minCoeff() + 1e-6);
	float scale_y = height / (XY.col(1).maxCoeff() - XY.col(1).minCoeff() + 1e-6);
	XY.rowwise() += Eigen::RowVector2f(trans_x, trans_y);
	XY.col(0) *= scale_x;
	XY.col(1) *= scale_y;

	// Translate and scale vK2 to [0, 1]
	float trans_2 = -(vK2.minCoeff());
	float scale_2 = 1 / (vK2.maxCoeff() - vK2.minCoeff() + 1e-6);
	vK2.array() += trans_2;
	vK2.array() *= scale_2;


	// 1) F, XY + vK2 ==> width x height: [0, 1]
	// 2) F, XY + label ==> width x height: {labels}
	// ===
	Eigen::MatrixXf image_value = Eigen::MatrixXf::Zero(height, width);
	Eigen::MatrixXi image_mask = Eigen::MatrixXi::Constant(height, width, 255);

	// traverse each tri and linear interpolate by vK2
	for (int i = 0; i < F.rows(); ++i) {
		int i0 = F(i, 0);
		int i1 = F(i, 1);
		int i2 = F(i, 2);

		Eigen::Vector2f p0 = XY.row(i0);
		Eigen::Vector2f p1 = XY.row(i1);
		Eigen::Vector2f p2 = XY.row(i2);

		float v0 = vK2(i0);
		float v1 = vK2(i1);
		float v2 = vK2(i2);
		int face_label = face_labels(i);

		// calc tri bounding box
		int min_x = max(0, (int)floor(min({ p0.x(), p1.x(), p2.x() })));
		int max_x = min(width - 1, (int)ceil(max({ p0.x(), p1.x(), p2.x() })));
		int min_y = max(0, (int)floor(min({ p0.y(), p1.y(), p2.y() })));
		int max_y = min(height - 1, (int)ceil(max({ p0.y(), p1.y(), p2.y() })));

		// calc tri area
		float area = (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p2.x() - p0.x()) * (p1.y() - p0.y());
		if (std::abs(area) < 1e-6) continue;

		// traverse pixels in tri
		for (int y = min_y; y <= max_y; ++y) {
			for (int x = min_x; x <= max_x; ++x) {
				Eigen::Vector2f p(x + 0.5f, y + 0.5f); // 像素中心点

				// calc barycentric coords
				float w0 = ((p1.x() - p.x()) * (p2.y() - p.y()) - (p2.x() - p.x()) * (p1.y() - p.y())) / area;
				float w1 = ((p2.x() - p.x()) * (p0.y() - p.y()) - (p0.x() - p.x()) * (p2.y() - p.y())) / area;
				float w2 = 1.0f - w0 - w1;

				// if pixel inside tri, interpolate
				if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
					float interpolated_value = w0 * v0 + w1 * v1 + w2 * v2;

					// if has higher value, update
					if (interpolated_value > image_value(y, x)) {
						image_value(y, x) = interpolated_value;
					}
					image_mask(y, x) = face_label;
				}
			}
		}
	}

	// 由于 Eigen 默认按列存储，而大多数图像库按行存储，需要转换存储顺序
	vector<unsigned char> image_value_data(width * height), image_mask_data(width * height);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			image_value_data[y * width + x] = (unsigned char)(image_value(y, x) * 255);
			image_mask_data[y * width + x] = (unsigned char)(image_mask(y, x));
		}
	}
	// save to png
	stbi_write_png((output_name + "_value.png").c_str(), width, height, 1, image_value_data.data(), width);
	stbi_write_png((output_name + "_mask.png").c_str(), width, height, 1, image_mask_data.data(), width);

}

void saveXYBinary() {
	namespace fs = filesystem;
	//fs::path dir = output_dir;
	fs::path xy_fn = output_name + "_xy.bin";
	fs::path xy_fp = xy_fn;

	// csv
	ofstream file(xy_fp, ios::binary);
	if (file.is_open()) {
		int rows = XY.rows();
		int cols = XY.cols();
		// write matrix rows and cols
		file.write(reinterpret_cast<char*>(&rows), sizeof(int));
		file.write(reinterpret_cast<char*>(&cols), sizeof(int));

		// write matrix data
		file.write(reinterpret_cast<const char*>(XY.data()), rows * cols * sizeof(float));
		file.close();
	}
}

int main(int argc, char** argv) {
	cxxopts::Options options("Data-Prepare", "Used to preprocess mesh data to be used for detection training");
	options.add_options()
		("m,mesh", "Input mesh file name", cxxopts::value<string>())
		("l,label", "Input label file name", cxxopts::value<string>())
		("w,width", "Width of image_value to be mapped to", cxxopts::value<int>()->default_value("1024"))
		("h,height", "Height of image_value to be mapped to", cxxopts::value<int>()->default_value("256"))
		("o,lower_bound", "Lower bound percentile of min curv", cxxopts::value<float>()->default_value("0.005"))
	    ("u,upper_bound", "Upper bound percentile of min curv", cxxopts::value<float>()->default_value("0.995"))
		("d,debug", "Print debug info", cxxopts::value<bool>()->default_value("false"))
		("help", "Print help information")
		;

	// parse args
	auto result = options.parse(argc, argv);

	// help
	if (argc < 2 || result.count("help")) {
		cout << options.help() << endl;
		return EXIT_SUCCESS;
	}

	// retrieve args
	string mesh_fp = result["mesh"].as<string>();
	output_name = filesystem::path(mesh_fp).replace_extension("").string();
	string label_fp = result["label"].as<string>();
	if (result.count("width")) 
	    width = result["width"].as<int>();
	if (result.count("height"))
	    height = result["height"].as<int>();
	if (result.count("lower_bound"))
		p_lower_bound = result["lower_bound"].as<float>();
	if (result.count("upper_bound"))
		p_upper_bound = result["upper_bound"].as<float>();
	debug = result["debug"].as<bool>();
	
	// load stuffs and preprocess
	if (!loadMyMesh(mesh_fp)) return EXIT_FAILURE;
	if (!loadLabel(label_fp)) return EXIT_FAILURE;
	if (!repairMesh()) return EXIT_FAILURE;
	if (!initGeometrycentral()) return EXIT_FAILURE;

	// 
	updateAndSaveFaceLabels();
	//saveNewMesh();

	//map2xy();
	//computeMeanCurvaturesAndClamp();

	//generateImageAndAnnotations();

	//saveXYBinary();

	cout << "Done!" << endl;

	return EXIT_SUCCESS;
}