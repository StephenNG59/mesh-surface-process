// == Eigen
#include <Eigen/Core>
// == vcglib
#include <wrap/io_trimesh/import_obj.h>
#include <wrap/io_trimesh/export_obj.h>
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/nring.h>
#include <vcg/complex/algorithms/mesh_to_matrix.h>
// == std
#include <format>

#include "log.h"
#include "cxxopts.hpp"
#include "json.hpp"

using namespace std;

// == Vcg MyMesh structure
class MyVertex;
class MyFace;
class MyEdge;
struct MyUsedTypes : public vcg::UsedTypes<
	vcg::Use<MyVertex>     ::AsVertexType,
	vcg::Use<MyFace>       ::AsFaceType,
	vcg::Use<MyEdge>       ::AsEdgeType> {
};
class  MyVertex : public vcg::Vertex<
	MyUsedTypes,
	vcg::vertex::Coord3f,
	vcg::vertex::Normal3f,
	vcg::vertex::VFAdj,
	vcg::vertex::VEAdj,
	vcg::vertex::CurvatureDirf,
	vcg::vertex::Qualityf,
	vcg::vertex::Mark,
	vcg::vertex::BitFlags> {
};
class  MyFace : public vcg::Face<
	MyUsedTypes,
	vcg::face::Normal3f,
	vcg::face::VFAdj,
	vcg::face::FFAdj,
	vcg::face::VertexRef,
	vcg::face::Mark,
	vcg::face::BitFlags> {
};
class  MyEdge : public vcg::Edge<
	MyUsedTypes,
	vcg::edge::VEAdj,
	vcg::edge::VertexRef,
	vcg::edge::Mark,
	vcg::edge::BitFlags> {
};
class  MyMesh : public vcg::tri::TriMesh<
	std::vector<MyVertex>,
	std::vector<MyFace>,
	std::vector<MyEdge> > {
};

// == Global
string mesh_name;
nlohmann::json json_data;
unique_ptr<MyMesh> vcg_mesh;
unique_ptr<vcg::tri::MyNring<MyMesh>> vcg_rw;


// == Params
string mesh_fp, anno_fp;
string output_dir;
float area_ratio_threshold = 20;

bool loadMyMesh(const string& mesh_filepath) {
	using vcg::tri::io::ImporterOBJ;

	vcg_mesh = make_unique<MyMesh>();
	mesh_name = filesystem::path(mesh_filepath).stem().string();

	int dummymask = 0;
	int result = ImporterOBJ<MyMesh>::Open(*vcg_mesh, mesh_filepath.c_str(), dummymask, 0);
	if (result != ImporterOBJ<MyMesh>::E_NOERROR) {
		LOG(LogLevel::error, format("Error loading mesh [{}]: {}", mesh_filepath, ImporterOBJ<MyMesh>::ErrorMsg(result)));

		if (ImporterOBJ<MyMesh>::ErrorCritical(result)) return false;
	}
	LOG(LogLevel::debug, format("[{}] sucessfully loaded with [{}] faces & [{}] verts.", mesh_name, vcg_mesh->FN(), vcg_mesh->VN()));

	return true;
}

bool loadAnnotation(const string& anno_filepath) {
	using json = nlohmann::json;

	// Default with all zeros
	Eigen::VectorXi vertex_labels, vertex_instances;
	vertex_labels = Eigen::VectorXi::Zero(vcg_mesh->VN());
	vertex_instances = Eigen::VectorXi::Zero(vcg_mesh->VN());

	// Load labels if file exists
	ifstream f(anno_filepath);
	if (f.is_open()) {
		f >> json_data;

		vector<int> labels = json_data["labels"].get<vector<int>>();
		vector<int> instances = json_data["instances"].get<vector<int>>();
		if (labels.size() != vcg_mesh->VN()) {
			LOG(LogLevel::error, format("[Error] Label size {} not matching vertex number {}.\n", labels.size(), vcg_mesh->VN()));
			return false;
		}
		if (instances.size() != vcg_mesh->VN()) {
			LOG(LogLevel::error, format("[Error] Instance size {} not matching vertex number {}.\n", instances.size(), vcg_mesh->VN()));
			return false;
		}

		for (int i = 0; i < static_cast<int>(labels.size()); i++) {
			vertex_labels(i) = labels[i];
			vertex_instances(i) = instances[i];
		}
	}

	// Add vcg attribute
	MyMesh::PerVertexAttributeHandle<int> label_handle = vcg::tri::Allocator<MyMesh>::AddPerVertexAttribute<int>(*vcg_mesh, "vertex_labels");
	MyMesh::PerVertexAttributeHandle<int> instance_handle = vcg::tri::Allocator<MyMesh>::AddPerVertexAttribute<int>(*vcg_mesh, "vertex_instances");
	for (int i = 0; i < vcg_mesh->VN(); i++) {
		label_handle[i] = vertex_labels(i);
		instance_handle[i] = vertex_instances(i);
	}

	return true;
}

void updateTopology() {
	using namespace vcg::tri;
	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexEdge(*vcg_mesh);
}

void removeComponentsUntilOneRemained() {
	using namespace vcg::tri;

	updateTopology();
	UpdateBounding<MyMesh>::Box(*vcg_mesh);

	float threshold = 0;
	pair<int, int> comp(0, 0);
	do
	{
		threshold += 0.1;
		comp = Clean<MyMesh>::RemoveSmallConnectedComponentsDiameter(
			*vcg_mesh, threshold * vcg_mesh->bbox.Diag());

		LOG(LogLevel::debug, format("Removed {} out of all {} components.", comp.second, comp.first));

	} while (comp.first - comp.second > 1);

	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);

	return;
}

void removeOuterRings(const float area_ratio_threshold) {
	using namespace vcg::tri;

	// === 1. Calculate each vertex area, and save per vertex quality as 
	//  quality = area / average_area * (label==0)
	Eigen::VectorXf vertex_stat = Eigen::VectorXf::Zero(vcg_mesh->VN());
	MeshToMatrix<MyMesh>::PerVertexArea(*vcg_mesh, vertex_stat);
	vertex_stat /= vertex_stat.mean();

	// if not has basin, i.e. max(area_ratio_max) < 20, return
	float max_ratio = vertex_stat.maxCoeff();
	if (max_ratio < area_ratio_threshold) {
		LOG(LogLevel::debug, format("max area ratio = {}, skip N-rings step.", max_ratio));
		return;
	}
	LOG(LogLevel::debug, format("max area ratio = {}, will do N-rings step.", max_ratio));

	for (auto& v : vcg_mesh->vert) {
		size_t vi = vcg::tri::Index(*vcg_mesh, v);
		float q = vertex_stat(vi)/* * (vertex_labels(vi) == 0)*/;
		vertex_stat(vi) = q;
		v.Q() = q;
	}

	// Set the source vertex to whom has the biggest quality
	int source_vid = -1;
	vertex_stat.maxCoeff(&source_vid);

	//if (vertex_labels(source_vid) != 0) {
	//    LOG(LogLevel::error, format("Nring source vertex's label = {}", vertex_labels(source_vid)));
	//}


	// === 2. Expand rings from source, based on quality,
	//  (basically, expand if q > 1.5) until non-expandable
	if (vcg_rw.get() == nullptr) {
		UpdateFlags<MyMesh>::FaceClearV(*vcg_mesh);
		UpdateFlags<MyMesh>::VertexClearV(*vcg_mesh);
		vcg_rw = unique_ptr<MyNring<MyMesh>>(new MyNring<MyMesh>(&(vcg_mesh->vert[source_vid]), &(*vcg_mesh)));
	}

	int last_vn, this_vn = 0, ring_k = 0;
	do {
		ring_k++;

		last_vn = this_vn;
		vcg_rw->expand(1);
		this_vn = vcg_rw->allV.size();
	} while (this_vn > last_vn);
	LOG(LogLevel::debug, format("[{}-Rings] has {} faces & {} verts.",
		ring_k, vcg_rw->allF.size(), vcg_rw->allV.size()));


	// === 3. Remove verts & faces in rings
	for (auto& v : vcg_rw->allV) {
		Allocator<MyMesh>::DeleteVertex(*vcg_mesh, *v);
	}
	for (auto& f : vcg_rw->allF) {
		Allocator<MyMesh>::DeleteFace(*vcg_mesh, *f);
	}

	// remove faces incident on deleted vertices
	Clean<MyMesh>::RemoveDegenerateVertex(*vcg_mesh);

	// compact vector
	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);

	return;
}

void makeMeshManifold() {
	using namespace vcg::tri;

	int nmv = 0, nmf = 0, urv = 0;
	do
	{
		updateTopology();
		UpdateBounding<MyMesh>::Box(*vcg_mesh);

		// "Repair non Manifold Vertices by splitting"
		nmv = Clean<MyMesh>::SplitNonManifoldVertex(*vcg_mesh, 0);
		//nmv = Clean<MyMesh>::RemoveNonManifoldVertex(*vcg_mesh);
		LOG(LogLevel::debug, format("Splited {} non-manifold vertices.", nmv));

		// "Remove Isolated pieces (wrt Diameter)"
		removeComponentsUntilOneRemained();
		//pair<int, int> comp = Clean<MyMesh>::RemoveSmallConnectedComponentsDiameter(*vcg_mesh, 0.05 * vcg_mesh->bbox.Diag());
		//if (debug) cout << format("Removed {} out of all {} components.\n", comp.second, comp.first);

		// "Repair non Manifold Edges"
		nmf = Clean<MyMesh>::RemoveNonManifoldFace(*vcg_mesh);
		LOG(LogLevel::debug, format("Removed {} non-manifold faces.", nmf));

		// "Remove Unreferenced Vertices"
		urv = Clean<MyMesh>::RemoveUnreferencedVertex(*vcg_mesh);
		LOG(LogLevel::debug, format("Removed {} unreferenced vertices.", urv));
		LOG(LogLevel::debug, format("After cleaning: [{}] faces & [{}] vertices.", vcg_mesh->FN(), vcg_mesh->VN()));

	} while (nmv > 0 || nmf > 0);


	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);
	updateTopology();

	return;
}

void writeToMFile() {
	// Find vcg attribute
	MyMesh::PerVertexAttributeHandle<int> label_handle =
		vcg::tri::Allocator<MyMesh>::FindPerVertexAttribute<int>(*vcg_mesh, "vertex_labels");
	MyMesh::PerVertexAttributeHandle<int> instance_handle =
		vcg::tri::Allocator<MyMesh>::FindPerVertexAttribute<int>(*vcg_mesh, "vertex_instances");


	filesystem::create_directories(output_dir);
	string filename = output_dir + "/" + mesh_name + ".A1.m";

	ofstream ofs(filename);
	if (!ofs.is_open()) {
		LOG(LogLevel::error, "Failed to open file for writing: " + filename);
		return;
	}

	LOG(LogLevel::debug, "Start writing " + filename);

	// Write vertices
	int vi = 0, fi = 0;
	for (const auto& v : vcg_mesh->vert) {
		// Vertex 1 0.1 0.2 0.3 {label=0 instance=0} 
		ofs << "Vertex " << (vi + 1) << " "
			<< v.P()[0] << " " << v.P()[1] << " " << v.P()[2]
			<< " {label=" << label_handle[v]
			<< " instance=" << instance_handle[v] << "}\n";
		vi++;
	}

	// Write faces
	using namespace vcg::tri;
	for (const auto& f : vcg_mesh->face) {
		// Face 1 1 2 3
		ofs << "Face " << (fi + 1) << " "
			<< Index(*vcg_mesh, f.V(0)) + 1 << " "
			<< Index(*vcg_mesh, f.V(1)) + 1 << " " 
			<< Index(*vcg_mesh, f.V(2)) + 1 << "\n";
		fi++;
	}

	ofs.close();
}

void writeToObj() {
	string filename = output_dir + "/" + mesh_name + ".A1.obj";
	vcg::tri::io::ExporterOBJ<MyMesh>::Save(*vcg_mesh, filename.c_str(), 0, false, 0);
	LOG(LogLevel::debug, "Obj file written: " + filename);
}

int main(int argc, char** argv) {
	cxxopts::Options options("A1-MeshCleaner", 
		"Phrase 1. Used to remove components and clean mesh.\n\t[Input]  mesh(.obj) & annotation(.json) files\n\t[Output] mesh(.A1.m) file");
	options.add_options()
		("m, mesh", "Input mesh file", cxxopts::value<string>())
		("a, annotation", "Input annotation file", cxxopts::value<string>())
		("o, output-dir", "Output file directory", cxxopts::value<string>())
		("t, threshold", "If max area ratio is lower than this, ignore basin-removal step.", cxxopts::value<float>())
		("l, log-level", "Log level (error|warning|info|debug|trace)", cxxopts::value<string>()->default_value("debug"))
		("h, help", "Print help information")
		;

	// parse args
	auto result = options.parse(argc, argv);

	// help
	if (argc < 3 || result.count("help")
		|| !result.count("mesh")
		|| !result.count("annotation")) {
		cout << options.help() << endl;
		return EXIT_SUCCESS;
	}

	// retrieve args
	mesh_fp = result["mesh"].as<string>();
	anno_fp = result["annotation"].as<string>();
	if (result.count("output-dir")) {
		output_dir = result["output-dir"].as<string>();
	}
	else {
		output_dir = filesystem::path(mesh_fp).parent_path().string();
	}
	if (result.count("threshold")) {
		area_ratio_threshold = result["threshold"].as<float>();
	}
	const auto lvl = result["log-level"].as<string>();
	if      (lvl == "error")   current_log_level() = LogLevel::error;
	else if (lvl == "warning") current_log_level() = LogLevel::warning;
	else if (lvl == "info")    current_log_level() = LogLevel::info;
	else if (lvl == "debug")   current_log_level() = LogLevel::debug;
	else if (lvl == "trace")   current_log_level() = LogLevel::trace;

	// Start
	auto t0 = clock();
	if (!loadMyMesh(mesh_fp)) return EXIT_FAILURE;
	if (!loadAnnotation(anno_fp)) return EXIT_FAILURE;

	removeComponentsUntilOneRemained();

	//// added to adapt some issue cases
	vcg::tri::Clean<MyMesh>::SplitNonManifoldVertex(*vcg_mesh, 0);
	vcg::tri::Clean<MyMesh>::RemoveNonManifoldFace(*vcg_mesh);
	//vcg::tri::Clean<MyMesh>::RemoveUnreferencedVertex(*vcg_mesh);
	vcg::tri::Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	vcg::tri::Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);
	updateTopology();

	removeOuterRings(area_ratio_threshold);
	makeMeshManifold();

	writeToMFile();
	writeToObj();

	return EXIT_SUCCESS;
}
