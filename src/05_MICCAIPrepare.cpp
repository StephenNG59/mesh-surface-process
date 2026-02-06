// == Eigen
#include <Eigen/core>
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
#include <cctype>
#include "cxxopts.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "json.hpp"
//// == polyscope
//#include <polyscope/polyscope.h>
//#include <polyscope/surface_mesh.h>


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
string mesh_name;
nlohmann::json json_data;
unique_ptr<MyMesh> vcg_mesh, vcg_mesh2d;
unique_ptr<vcg::tri::MyNring<MyMesh>> vcg_rw;
unique_ptr<geometrycentral::surface::SurfaceMesh> gc_mesh;
unique_ptr<VertexPositionGeometry> gc_geom;
unique_ptr<VertexPositionGeometry> gc_geom2d;
//unique_ptr<polyscope::SurfaceMesh> ps_mesh;
Eigen::MatrixXf V;
Eigen::MatrixXi F;
Eigen::VectorXi vertex_labels, vertex_instances;
Eigen::MatrixXf XY;
Eigen::VectorXf vK1, vK2, vK3;


// == Params
string mesh_fp, label_fp, output_dir;
int width = 1024, height = 256;
float p_lower_bound = 0.005, p_upper_bound = 0.995;
bool ignore_nrings = false;
bool debug = false;
float regularize_lambda = 0.0;


// == Miscs
void updateTopology() {
	using namespace vcg::tri;
	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexEdge(*vcg_mesh);
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

bool needToRemoveOuterRings(const string& full_name) {
	if (ignore_nrings) return false;

	string name = full_name;

	// ABCDE_lower -> ABCDE
	size_t pos = full_name.find('_');
	if (pos != string::npos) {
		name = full_name.substr(0, pos);

	}
	//cout << "name=" << name << endl;

	// 1. Upper case + number
	bool isUpper = true;
	for (char ch : name) {
		if (!(isupper(ch)) && !(isdigit(ch))) {
			isUpper = false;
			break;
		}
	}

	// 2. 'patient' + number
	bool isPatient = false;
	if (name.rfind("patient", 0) == 0) {
		for (size_t i = 7; i < name.size(); i++) {
			if (!isdigit(name[i])) {
				isPatient = false;
				break;
			}
		}
	}

	if (!isUpper && !isPatient) return false;

	// 3. does not have basin (put this in removeOuterRings())
	return true;
}

void saveNewMesh() {
	namespace fs = filesystem;

	fs::path dir = output_dir;
	fs::path obj_fn = mesh_name + ".obj";
	
	vcg::tri::io::Exporter<MyMesh>::Save(*vcg_mesh, (dir / obj_fn).string().c_str());
}

void saveXYBinary() {
	namespace fs = filesystem;
	fs::path dir = output_dir;
	fs::path xy_fn = mesh_name + "_xy.bin";
	fs::path xy_fp = dir / xy_fn;

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

float calcCorrelation() {
	using namespace Eigen;
	VectorXf Vx = V.col(0);
	VectorXf XYx = XY.col(0);

	float mean_Vx = Vx.mean();
	float mean_XYx = XYx.mean();
	float numerator = (Vx.array() - mean_Vx).matrix().dot((XYx.array() - mean_XYx).matrix());
	float denominator = sqrt(
		(Vx.array() - mean_Vx).square().matrix().sum() * 
		(XYx.array() - mean_XYx).square().matrix().sum());
	float correlation = numerator / denominator;

	return correlation;
}

// == Functions
bool loadMyMesh(const string& mesh_filepath) {
	using vcg::tri::io::Importer;

	// Import mesh
	vcg_mesh = make_unique<MyMesh>();
	mesh_name = filesystem::path(mesh_filepath).stem().string();

	int result = Importer<MyMesh>::Open(*vcg_mesh, mesh_filepath.c_str());
	if (result != Importer<MyMesh>::E_NOERROR) {
		cerr << "Oops! Error loading mesh: " << Importer<MyMesh>::ErrorMsg(result) << endl;

		if (Importer<MyMesh>::ErrorCritical(result))
			return false;
	}

	if (debug) {
		std::cout << std::format("[{}] sucessfully loaded with [{}] faces & [{}] verts.\n",
			mesh_name, vcg_mesh->FN(), vcg_mesh->VN());
	}

	return true;
}

bool loadLabel(const string& label_filepath) {
	using json = nlohmann::json;

	// Default with all zeros
	vertex_labels = Eigen::VectorXi::Zero(vcg_mesh->VN());
	vertex_instances = Eigen::VectorXi::Zero(vcg_mesh->VN());

	// Load labels if file exists
	ifstream f(label_filepath);
	if (f.is_open()) {
		f >> json_data;

		vector<int> labels = json_data["labels"].get<vector<int>>();
		vector<int> instances = json_data["instances"].get<vector<int>>();
		if (labels.size() != vcg_mesh->VN()) {
			cerr << format("[Error] Label size {} not matching face number {}.\n", labels.size(), vcg_mesh->VN());
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

		if (debug) cout << format("Removed {} out of all {} components.\n", comp.second, comp.first);

	} while (comp.first - comp.second > 1);

	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);

	return;
}

void removeOuterRings() {
	using namespace vcg::tri;

	// === 1. Calculate each vertex area, and save per vertex quality as 
	//  quality = area / average_area * (label==0)
	Eigen::VectorXf vertex_stat = Eigen::VectorXf::Zero(vcg_mesh->VN());
	MeshToMatrix<MyMesh>::PerVertexArea(*vcg_mesh, vertex_stat);
	vertex_stat /= vertex_stat.mean();

	// if not has basin, i.e. max(area_ratio_max) < 20, return
	float max_ratio = vertex_stat.maxCoeff();
	if (max_ratio < 20) {
		if (debug) cout << format("max area ratio = {}, skip N-rings step.\n", max_ratio);
		return;
	}
	if (debug) cout << format("max area ratio = {}, will do N-rings step.\n", max_ratio);

	for (auto& v : vcg_mesh->vert) {
		size_t vi = vcg::tri::Index(*vcg_mesh, v);
		float q = vertex_stat(vi) * (vertex_labels(vi) == 0);
		vertex_stat(vi) = q;
		v.Q() = q;
	}

	// Set the source vertex to whom has the biggest quality
	int source_vid = -1;
	vertex_stat.maxCoeff(&source_vid);


	// === 2. Expand rings from source, based on quality,
	//  (basically, expand if q > 1) until non-expandable
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
	if (debug) cout << format("[{}-Rings] has {} faces & {} verts.\n",
		ring_k, vcg_rw->allF.size(), vcg_rw->allV.size());


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
	    if (debug) cout << format("Splited {} non-manifold vertices.\n", nmv);

		// "Remove Isolated pieces (wrt Diameter)"
		removeComponentsUntilOneRemained();
		//pair<int, int> comp = Clean<MyMesh>::RemoveSmallConnectedComponentsDiameter(*vcg_mesh, 0.05 * vcg_mesh->bbox.Diag());
		//if (debug) cout << format("Removed {} out of all {} components.\n", comp.second, comp.first);

		// "Repair non Manifold Edges"
		nmf = Clean<MyMesh>::RemoveNonManifoldFace(*vcg_mesh);
		if (debug) cout << format("Removed {} non-manifold faces.\n", nmf);

		// "Remove Unreferenced Vertices"
		urv = Clean<MyMesh>::RemoveUnreferencedVertex(*vcg_mesh);
		if (debug) cout << format("Removed {} unreferenced vertices.\n", urv);
		if (debug) cout << std::format("After cleaning: [{}] faces & [{}] vertices.\n", vcg_mesh->FN(), vcg_mesh->VN());

	} while (nmv > 0 || nmf > 0);


	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);
	updateTopology();

	return;
}

void updateAndSaveVertexLabels() {
	namespace fs = filesystem;

	vertex_labels = Eigen::VectorXi(vcg_mesh->VN());
	vertex_instances = Eigen::VectorXi(vcg_mesh->VN());
	vector<int> label_vec, instance_vec;
	
	MyMesh::PerVertexAttributeHandle<int> label_handle = vcg::tri::Allocator<MyMesh>::FindPerVertexAttribute<int>(*vcg_mesh, "vertex_labels");
	MyMesh::PerVertexAttributeHandle<int> instance_handle = vcg::tri::Allocator<MyMesh>::FindPerVertexAttribute<int>(*vcg_mesh, "vertex_instances");
	for (int i = 0; i < vcg_mesh->VN(); i++) {
		vertex_labels(i) = label_handle[i];
		label_vec.push_back(label_handle[i]);
		instance_vec.push_back(instance_handle[i]);
	}

	json_data["instances"] = instance_vec;
	json_data["labels"] = label_vec;

	// save to json file
	fs::path dir = output_dir;
	fs::path json_fn = mesh_name + ".json";
	ofstream f(dir / json_fn);
	f << json_data.dump();
	f.close();

	return;
}

void computeMeanCurvaturesAndClamp() {
	using namespace vcg::tri;

	updateTopology();  // NECASSARY!
	UpdateNormal<MyMesh>::PerVertexAngleWeighted(*vcg_mesh);
	UpdateNormal<MyMesh>::PerFaceNormalized(*vcg_mesh);
	UpdateNormal<MyMesh>::NormalizePerVertex(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);

	if (debug) cout << "updateCurvatureLocal()...";
	double start = clock();
	UpdateCurvatureFitting<MyMesh>::updateCurvatureLocal(*vcg_mesh, 0.01f * vcg_mesh->bbox.Diag());
	if (debug) cout << format("done with {}ms.\n", clock() - start);

	// Retrieve results
	int vn = vcg_mesh->VN(), i = 0;
	vK1.resize(vn);
	vK2.resize(vn);
	vK3.resize(vn);
	ForEachVertex(*vcg_mesh, [&i](const MyMesh::VertexType& v) {
		float k1 = v.cK1(), k2 = v.cK2();
		vK1(i) = k1;  // max curv
		vK2(i) = k2;  // min curv
		vK3(i) = sqrt((k1 * k1 + k2 * k2) / 2);   // curvedness
		i++;
		});

	// Clamp by percentile
	clampToPercentile(vK1, p_lower_bound, p_upper_bound);
	clampToPercentile(vK2, p_lower_bound, p_upper_bound);
	clampToPercentile(vK3, p_lower_bound, p_upper_bound);

	//if (debug) cout << "computeMeanCurvaturesAndClamp() done.\n";

	return;
}

void initGeometrycentral() {
	using namespace vcg::tri;

	// Get V, F
	V = Eigen::MatrixXf(vcg_mesh->VN(), 3);
	F = Eigen::MatrixXi(vcg_mesh->FN(), 3);

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
			fi++;
		}
	}

	// Init geometrycentral
	try {
	    gc_mesh = unique_ptr<ManifoldSurfaceMesh>(new ManifoldSurfaceMesh(F));
	    gc_geom = unique_ptr<VertexPositionGeometry>(new VertexPositionGeometry(*gc_mesh, V));
	}
	catch (const std::exception& e) {
		cout << e.what() << endl;
	}

	if (debug) cout << "initGeometrycentral() done.\n";

	return;
}

void map2xy() {
	if (debug) cout << "map2xy()...";

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

	// 计算相关系数，如果是正，则y取负；反之则x取负（使得图像和mesh上下左右一致）
	float corr = calcCorrelation();
	if (corr < 0) {
		XY.col(0).array() *= -1;
	}
	else {
		XY.col(1).array() *= -1;
	}

	//ps_mesh->addVertexScalarQuantity("X", XY.col(0));
	//ps_mesh->addVertexScalarQuantity("Y", XY.col(1));

	if (debug) cout << "done.\n";

	return;
}

void generateImageAndMask() {
	namespace fs = filesystem;

	// == Translate and scale XY to image's width x height
	float trans_x = -(XY.col(0).minCoeff());
	float trans_y = -(XY.col(1).minCoeff());
	float scale_x = width / (XY.col(0).maxCoeff() - XY.col(0).minCoeff() + 1e-6);
	float scale_y = height / (XY.col(1).maxCoeff() - XY.col(1).minCoeff() + 1e-6);
	XY.rowwise() += Eigen::RowVector2f(trans_x, trans_y);
	XY.col(0) *= scale_x;
	XY.col(1) *= scale_y;

	// darker -> more significant
	vK1.array() *= -1;
	vK3.array() *= -1;

	// == Translate and scale vK1 to [0, 1]
	float trans_1 = -(vK1.minCoeff());
	float scale_1 = 1 / (vK1.maxCoeff() - vK1.minCoeff() + 1e-6);
	vK1.array() += trans_1;
	vK1.array() *= scale_1;

	// == Translate and scale vK2 to [0, 1]
	float trans_2 = -(vK2.minCoeff());
	float scale_2 = 1 / (vK2.maxCoeff() - vK2.minCoeff() + 1e-6);
	vK2.array() += trans_2;
	vK2.array() *= scale_2;

	// == Translate and scale vK3 to [0, 1]
	float trans_3 = -(vK3.minCoeff());
	float scale_3 = 1 / (vK3.maxCoeff() - vK3.minCoeff() + 1e-6);
	vK3.array() += trans_3;
	vK3.array() *= scale_3;

	// 创建图像和 mask
	Eigen::MatrixXf image_r = Eigen::MatrixXf::Zero(height, width);  // R
	Eigen::MatrixXf image_g = Eigen::MatrixXf::Zero(height, width);  // G
	Eigen::MatrixXf image_b = Eigen::MatrixXf::Zero(height, width);  // B
	Eigen::MatrixXi mask = Eigen::MatrixXi::Constant(height, width, -1);  // Mask 图，默认值 -1 代表未覆盖

	// 遍历每个三角形面
	for (int i = 0; i < F.rows(); ++i) {
		int i0 = F(i, 0);
		int i1 = F(i, 1);
		int i2 = F(i, 2);

		Eigen::Vector2f p0 = XY.row(i0);
		Eigen::Vector2f p1 = XY.row(i1);
		Eigen::Vector2f p2 = XY.row(i2);

		float r0 = vK1(i0), r1 = vK1(i1), r2 = vK1(i2);  // max curv
		float b0 = vK2(i0), b1 = vK2(i1), b2 = vK2(i2);  // min curv
		float g0 = vK3(i0), g1 = vK3(i1), g2 = vK3(i2);  // curvedness

		int label0 = vertex_labels(i0);
		int label1 = vertex_labels(i1);
		int label2 = vertex_labels(i2);

		// 计算三角形的包围盒
		int min_x = std::max(0, (int)std::floor(std::min({ p0.x(), p1.x(), p2.x() })));
		int max_x = std::min(width - 1, (int)std::ceil(std::max({ p0.x(), p1.x(), p2.x() })));
		int min_y = std::max(0, (int)std::floor(std::min({ p0.y(), p1.y(), p2.y() })));
		int max_y = std::min(height - 1, (int)std::ceil(std::max({ p0.y(), p1.y(), p2.y() })));

		// 计算三角形的面积
		float area = (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p2.x() - p0.x()) * (p1.y() - p0.y());
		if (std::abs(area) < 1e-6) continue;

		// 遍历三角形包围盒内的每个像素
		for (int y = min_y; y <= max_y; ++y) {
			for (int x = min_x; x <= max_x; ++x) {
				Eigen::Vector2f p(x + 0.5f, y + 0.5f);  // 像素的中心点

				// 计算重心坐标
				float w0 = ((p1.x() - p.x()) * (p2.y() - p.y()) - (p2.x() - p.x()) * (p1.y() - p.y())) / area;
				float w1 = ((p2.x() - p.x()) * (p0.y() - p.y()) - (p0.x() - p.x()) * (p2.y() - p.y())) / area;
				float w2 = 1.0f - w0 - w1;

				// 如果像素在三角形内，进行插值
				if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
					// 计算插值后的灰度值
					float interpolated_r = w0 * r0 + w1 * r1 + w2 * r2;
					float interpolated_g = w0 * g0 + w1 * g1 + w2 * g2;
					float interpolated_b = w0 * b0 + w1 * b1 + w2 * b2;
					image_r(y, x) = std::max(image_r(y, x), interpolated_r);
					image_g(y, x) = std::max(image_g(y, x), interpolated_g);
					image_b(y, x) = std::max(image_b(y, x), interpolated_b);

					// 确定该像素的标签：选择最大 vK2 对应的标签
					int pixel_label = label0;
					if (vK2(i1) > vK2(i0) && vK2(i1) > vK2(i2)) {
						pixel_label = label1;
					}
					else if (vK2(i2) > vK2(i0) && vK2(i2) > vK2(i1)) {
						pixel_label = label2;
					}
					
					//// 选择最接近 (w最大) 的顶点对应的标签 但并不是全局最接近
					//if (w1 > w0 && w1 > w2) {
					//	pixel_label = label1;
					//}
					//else if (w2 > w0 && w2 > w1) {
					//	pixel_label = label2;
					//}
					
					// 更新 mask，保持最大 vK2 的标签
					mask(y, x) = pixel_label;
				}
			}
		}
	}

	// 由于 Eigen 默认按列存储，而大多数图像库按行存储，需要转换存储顺序
	vector<unsigned char> image_value_data(width * height * 3), 
		image_mask_data(width * height), 
		image_gray_data(width * height);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			image_value_data[(y * width + x) * 3]     = (unsigned char)(image_r(y, x) * 255);
			image_value_data[(y * width + x) * 3 + 1] = (unsigned char)(image_g(y, x) * 255);
			image_value_data[(y * width + x) * 3 + 2] = (unsigned char)(image_b(y, x) * 255);
			image_gray_data[y * width + x] = (unsigned char)(image_b(y, x) * 255); // b: min_curv
			image_mask_data[y * width + x] = (unsigned char)(mask(y, x));
		}
	}

	// 保存图像和 mask
	fs::path dir = output_dir;
	fs::path image_fn = mesh_name + ".png";
	fs::path gray_image_fn = mesh_name + "_gray.png";
	fs::path mask_fn = mesh_name + "_mask.png";
	stbi_write_png((dir / image_fn).string().c_str(),
		width, height, 3, image_value_data.data(), width * 3);
	stbi_write_png((dir / gray_image_fn).string().c_str(),
		width, height, 1, image_gray_data.data(), width);
	stbi_write_png((dir / mask_fn).string().c_str(),
		width, height, 1, image_mask_data.data(), width);
}

//void initPolyscope() {
//	polyscope::view::setNavigateStyle(polyscope::NavigateStyle::Free);
//	polyscope::init();
//	ps_mesh = std::unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(mesh_name, V, F));
//	auto ps_permutations = polyscopePermutations(*gc_mesh);
//	ps_mesh->setAllPermutations<std::vector<size_t>>(ps_permutations);
//	ps_mesh->addVertexScalarQuantity("vertex labels", vertex_labels);
//}
//
//void callback() {}



int main(int argc, char** argv) {
	cxxopts::Options options("MICCAI data prepare", "Used to preprocess MICCAI dataset.");
	options.add_options()
		("m,mesh", "Input mesh file name", cxxopts::value<string>())
		("l,label", "Input label file name", cxxopts::value<string>())
		("o,output_dir", "Output file directory", cxxopts::value<string>())
		("w,width", "Width of image_value to be mapped to", cxxopts::value<int>()->default_value("1024"))
		("height", "Height of image_value to be mapped to", cxxopts::value<int>()->default_value("256"))
		("lower_bound", "Lower bound percentile of min curv", cxxopts::value<float>()->default_value("0.005"))
		("upper_bound", "Upper bound percentile of min curv", cxxopts::value<float>()->default_value("0.995"))
		("i,ignore_nrings", "Ignore NRings removal step", cxxopts::value<bool>()->default_value("false"))
		("d,debug", "Print debug info", cxxopts::value<bool>()->default_value("false"))
		("h,help", "Print help information")
		;

	// parse args
	auto result = options.parse(argc, argv);

	// help
	if (argc < 2 || result.count("help")) {
		cout << options.help() << endl;
		return EXIT_SUCCESS;
	}

	// retrieve args
	mesh_fp = result["mesh"].as<string>();
	label_fp = result["label"].as<string>();
	if (result.count("output_dir")) {
	    output_dir = result["output_dir"].as<string>();
	}
	else {
		//TODO root_directory()好像不太对
		output_dir = filesystem::path(mesh_fp).parent_path().string();
	}
	width = result["width"].as<int>();
	height = result["height"].as<int>();
	p_lower_bound = result["lower_bound"].as<float>();
	p_upper_bound = result["upper_bound"].as<float>();
	ignore_nrings = result["ignore_nrings"].as<bool>();
	debug = result["debug"].as<bool>();


	// == Start
	auto t0 = clock();
	if (!loadMyMesh(mesh_fp)) return EXIT_FAILURE;
	if (!loadLabel(label_fp)) return EXIT_FAILURE;
	cout << format("Processing {}...", mesh_name);

	removeComponentsUntilOneRemained();

	if (needToRemoveOuterRings(mesh_name)) {
	    removeOuterRings();
	}

	makeMeshManifold();
	updateAndSaveVertexLabels();
	saveNewMesh();

	// Compute value image
	computeMeanCurvaturesAndClamp();

	// Compute mask image
	//initPolyscope();
	try {
	    initGeometrycentral();
	    map2xy();
	}
	catch (const std::exception& e) {
		cout << e.what();
	}
	generateImageAndMask();
	saveXYBinary();

	//polyscope::state::userCallback = callback;
	//polyscope::show();
	cout << format("{} seconds\n", int((clock() - t0) / 1000));
}