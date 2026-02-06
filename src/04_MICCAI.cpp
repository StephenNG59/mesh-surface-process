// ==
#include <Eigen/core>
// == polyscope
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
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
#include <vcg/complex/algorithms/nring.h>
#include <vcg/complex/algorithms/mesh_to_matrix.h>
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
using namespace polyscope;

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
string label_fp = "";
unique_ptr<MyMesh> vcg_mesh;
unique_ptr<vcg::tri::MyNring<MyMesh>> vcg_rw;
unique_ptr<geometrycentral::surface::SurfaceMesh> gc_mesh;
unique_ptr<VertexPositionGeometry> gc_geom;
unique_ptr<VertexPositionGeometry> gc_geom2d;
unique_ptr<polyscope::SurfaceMesh> ps_mesh;
Eigen::MatrixXf V;
Eigen::MatrixXi F;
Eigen::VectorXi vertex_labels;
Eigen::MatrixXf XY;
Eigen::VectorXf vK1, vK2;

// == Params
float regularize_lambda = 0.0;
float p_lower_bound = 0.005, p_upper_bound = 0.995;
int width = 1024, height = 256;
bool debug = true;

// === Nrings
int nring_source_v = 0;
int nring_expand_k = 50;

bool loadMyMesh(const std::string& mesh_filepath) {
	using vcg::tri::io::Importer;

	// Import mesh
	vcg_mesh = make_unique<MyMesh>();
	int result = Importer<MyMesh>::Open(*vcg_mesh, mesh_filepath.c_str());
	if (result != Importer<MyMesh>::E_NOERROR) {
		cerr << "Error loading mesh: " << Importer<MyMesh>::ErrorMsg(result) << endl;
		//return false;
	}

	//name = mesh_filepath;
	if (debug)
		std::cout << std::format("{} sucessfully loaded with {} faces & {} verts.\n",
			mesh_filepath, vcg_mesh->FN(), vcg_mesh->VN());

	return true;
}

bool loadLabel(const std::string& label_filepath) {
	using json = nlohmann::json;

	// Default with all zeros
	Eigen::VectorXi label_vec = Eigen::VectorXi::Zero(vcg_mesh->VN());

	// Load labels if file exists
	std::ifstream f(label_filepath);
	if (f.is_open()) {
		json j;
		f >> j;

		vector<int> labels = j["labels"].get<vector<int>>();
		if (labels.size() != vcg_mesh->VN()) {
			std::cerr << std::format("[Error] Label size {} not matching face number {}.\n", labels.size(), vcg_mesh->VN());
			return false;
		}

		for (int i = 0; i < static_cast<int>(labels.size()); i++) {
			label_vec(i) = labels[i];
		}
	}

	// Add vcg attribute
	MyMesh::PerVertexAttributeHandle<int> label_handle =
		vcg::tri::Allocator<MyMesh>::AddPerVertexAttribute<int>(*vcg_mesh, "vertex_labels");
	for (int i = 0; i < vcg_mesh->VN(); i++) {
		label_handle[i] = label_vec[i];
	}

	return true;
}

void updateTopology() {
	using namespace vcg::tri;

	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexEdge(*vcg_mesh);
}

bool initGeometrycentral() {
	using namespace vcg::tri;

	// Get V, F & label
	V = Eigen::MatrixXf(vcg_mesh->VN(), 3);
	F = Eigen::MatrixXi(vcg_mesh->FN(), 3);
	vertex_labels = Eigen::VectorXi(vcg_mesh->VN());
	// Normals
	UpdateNormal<MyMesh>::PerVertexAngleWeighted(*vcg_mesh);
	UpdateNormal<MyMesh>::PerFaceNormalized(*vcg_mesh);

	// Face handle for labels
	MyMesh::PerVertexAttributeHandle<int> label_handle =
		Allocator<MyMesh>::FindPerVertexAttribute<int>(*vcg_mesh, "vertex_labels");

	int vi = 0, fi = 0;
	for (const auto& v : vcg_mesh->vert) {
		if (!v.IsD()) {
			V.row(vi) = Eigen::Vector3f(v.P()[0], v.P()[1], v.P()[2]);
			vertex_labels[vi] = label_handle[vi];
			vi++;
		}
	}
	for (const auto& f : vcg_mesh->face) {
		if (!f.IsD()) {
			F.row(fi) = Eigen::Vector3i(
				Index(*vcg_mesh, f.V(0)),
				Index(*vcg_mesh, f.V(1)),
				Index(*vcg_mesh, f.V(2)));
			fi++;
		}
	}

	// Init geometrycentral
	gc_mesh = unique_ptr<geometrycentral::surface::SurfaceMesh>(new geometrycentral::surface::SurfaceMesh(F));
	gc_geom = unique_ptr<VertexPositionGeometry>(new VertexPositionGeometry(*gc_mesh, V/*, face_labels*/));

	return true;
}

bool initPolyscope() {
	polyscope::view::setNavigateStyle(polyscope::NavigateStyle::Free);
	polyscope::init();
	ps_mesh = std::unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(output_name, V, F));
	auto ps_permutations = polyscopePermutations(*gc_mesh);
	ps_mesh->setAllPermutations<std::vector<size_t>>(ps_permutations);
	ps_mesh->addVertexScalarQuantity("vertex_labels", vertex_labels)->setEnabled(true);

	return true;
}

void calcAreaRatioTimesLabel() {
	// Calculate each vertex area, and save per vertex quality as 
	//  quality = area / average_area * (label==0)
	Eigen::VectorXf vertex_stat = Eigen::VectorXf::Zero(vcg_mesh->VN());
	vcg::tri::MeshToMatrix<MyMesh>::PerVertexArea(*vcg_mesh, vertex_stat);
	vertex_stat /= vertex_stat.mean();

	ps_mesh->addVertexScalarQuantity("area ratio", vertex_stat);

	for (auto& v : vcg_mesh->vert) {
		size_t vi = vcg::tri::Index(*vcg_mesh, v);
		float q = vertex_stat(vi) * (vertex_labels(vi) == 0);
		vertex_stat(vi) = q;
		v.Q() = q;
	}

	// Set the source vertex to whom has the biggest quality
	size_t max_coeff = -1;
	vertex_stat.maxCoeff(&max_coeff);
	nring_source_v = max_coeff;

	ps_mesh->addVertexScalarQuantity("delete potential", vertex_stat)->setEnabled(true);
}

void expandNrings(int n) {
	// Expand N-rings based on quality. (basically, expand if q > 1)

	using namespace std;
	using namespace vcg;

	if (vcg_rw.get() == nullptr) {
		tri::UpdateFlags<MyMesh>::FaceClearV(*vcg_mesh);
		tri::UpdateFlags<MyMesh>::VertexClearV(*vcg_mesh);
		vcg_rw = unique_ptr<tri::MyNring<MyMesh>>(new tri::MyNring<MyMesh>(&(vcg_mesh->vert[nring_source_v]), &(*vcg_mesh)));
	}

	vcg_rw->expand(n);

	Eigen::VectorXi flags = Eigen::VectorXi::Zero(vcg_mesh->FN());
	for (const auto &f : vcg_rw->allF) {
		flags(tri::Index<MyMesh>(*vcg_mesh, f)) = 1;
	}
	ps_mesh->addFaceScalarQuantity(std::to_string(n) + "-rings f", flags);

	flags = Eigen::VectorXi::Zero(vcg_mesh->VN());
	for (const auto& v : vcg_rw->allV) {
		flags(tri::Index<MyMesh>(*vcg_mesh, v)) = 1;
	}
	ps_mesh->addVertexScalarQuantity(std::to_string(n) + "-rings v", flags)->setEnabled(true);

	cout << format("Nring has {} faces, {} verts.\n", vcg_rw->allF.size(), vcg_rw->allV.size());
}

void deleteNrings() {
	using namespace vcg::tri;

	//int fn = Clean<MyMesh>::RemoveDegenerateFace(*vcg_mesh);
	//if (debug) cout << format("Removed {} degenerate faces.\n", fn);

	int di = 0;
	for (auto& v : vcg_rw->allV) {
		Allocator<MyMesh>::DeleteVertex(*vcg_mesh, *v);
		di++;
	}
	if (debug) cout << format("Deleted {} verts.\n", di);

	di = 0;
	for (auto& f : vcg_rw->allF) {
		Allocator<MyMesh>::DeleteFace(*vcg_mesh, *f);
		di++;
	}
	if (debug) cout << format("Deleted {} faces.\n", di);

	// all the faces incident on deleted vertices will be deleted (although return == 0)
	int vn = Clean<MyMesh>::RemoveDegenerateVertex(*vcg_mesh);  
	if (debug) cout << format("Removed {} degenerate verts.\n", vn);

	// this maybe not necessary
	//fn = Clean<MyMesh>::RemoveDegenerateFace(*vcg_mesh);
	//if (debug) cout << format("Removed {} degenerate faces.\n", fn);

	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);

	if (debug) cout << format("Now mesh has {} faces and {} verts.\n", vcg_mesh->FN(), vcg_mesh->VN());
}

void makeMeshManifoldAndSave() {
	using namespace vcg::tri;
	using json = nlohmann::json;

	// 1. remain 1 component
	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);

	// "Repair non Manifold Vertices by splitting"
	int num = Clean<MyMesh>::SplitNonManifoldVertex(*vcg_mesh, 0);
	if (debug) cout << format("Splited {} non-manifold vertices.\n", num);

	float threshold = 0;
	pair<int, int> comp(0, 0);
	do
	{
		threshold += 0.1;
		comp = Clean<MyMesh>::RemoveSmallConnectedComponentsDiameter(
			*vcg_mesh, threshold * vcg_mesh->bbox.Diag());

		if (debug) cout << format("Removed {} out of all {} components.\n", comp.second, comp.first);

	} while (comp.first - comp.second > 1);

	// 2. non manifold edges/vertices handle
	// "Repair non Manifold Vertices by splitting"
	num = Clean<MyMesh>::SplitNonManifoldVertex(*vcg_mesh, 0);
	if (debug) cout << format("Splited {} non-manifold vertices.\n", num);
	// "Remove Isolated pieces (wrt Diameter)"
	comp = Clean<MyMesh>::RemoveSmallConnectedComponentsDiameter(*vcg_mesh, 0.05 * vcg_mesh->bbox.Diag());
	if (debug) cout << format("Removed {} out of all {} components.\n", comp.second, comp.first);
	// "Repair non Manifold Edges"
	num = Clean<MyMesh>::RemoveNonManifoldFace(*vcg_mesh);
	if (debug) cout << format("Removed {} non-manifold faces.\n", num);
	// "Remove Unreferenced Vertices"
	num = Clean<MyMesh>::RemoveUnreferencedVertex(*vcg_mesh);
	if (debug) cout << format("Removed {} unreferenced vertices.\n", num);
	if (debug) cout << std::format("{} faces & {} vertices after cleaning.\n", vcg_mesh->FN(), vcg_mesh->VN());

	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);

	// 3. export mesh file
	io::Exporter<MyMesh>::Save(*vcg_mesh, (output_name + "-new.obj").c_str());

	// 4. export label file
	MyMesh::PerVertexAttributeHandle<int> label_handle = Allocator<MyMesh>::FindPerVertexAttribute<int>(*vcg_mesh, "vertex_labels");
	if (Allocator<MyMesh>::IsValidHandle(*vcg_mesh, label_handle)) {
		json j;
		j["labels"] = json::array();
		for (const auto& v : vcg_mesh->vert) {
			j["labels"].push_back(label_handle[v]);
		}

		ofstream f(output_name + "-new.json");
		f << j.dump(4);
		f.close();
	}

}

void callback() {
	using namespace ImGui;

	// Area
	if (Button("Step1. Calculate Area")) {
		calcAreaRatioTimesLabel();
	} Separator();

	// Nrings
	PushItemWidth(120); 
	InputInt("rings once from source vertex:", &nring_expand_k, 1, 5); SameLine();
	InputInt("vertex", &nring_source_v, 1, 100);
	if (Button("Step2. Expand Ring")) {
		expandNrings(nring_expand_k);
	} SameLine();
	if (Button("Reset Nrings")) {
		vcg_rw.reset();
	}
	PopItemWidth();

	// Delete
	if (Button("Step3. Delete Ring's Verts")) {
		deleteNrings();
	}

	if (Button("Step4. Repair, Clean and Save new mesh")) {
		makeMeshManifoldAndSave();
	}

}

int main(int argc, char** argv) {
	cxxopts::Options options("MICCAI data visualization", "Used to visualize MICCAI dataset and the processing of it.");
	options.add_options()
		("m,mesh", "Input mesh file name", cxxopts::value<string>())
		("l,label", "Input label file name", cxxopts::value<string>())
		("w,width", "Width of image_value to be mapped to", cxxopts::value<int>()->default_value("1024"))
		("h,height", "Height of image_value to be mapped to", cxxopts::value<int>()->default_value("256"))
		("o,lower_bound", "Lower bound percentile of min curv", cxxopts::value<float>()->default_value("0.005"))
		("u,upper_bound", "Upper bound percentile of min curv", cxxopts::value<float>()->default_value("0.995"))
		("d,debug", "Print debug info", cxxopts::value<bool>()->default_value("true"))
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
	if (result.count("label"))
		label_fp = result["label"].as<string>();
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
	updateTopology();
	if (!initGeometrycentral()) return EXIT_FAILURE;
	if (!initPolyscope()) return EXIT_FAILURE;

	polyscope::state::userCallback = callback;
	polyscope::show();

	return EXIT_SUCCESS;
}