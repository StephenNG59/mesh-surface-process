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
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/curvature_fitting.h>
// == std
#include <format>

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
unique_ptr<MyMesh> vcg_mesh;
unique_ptr<ManifoldSurfaceMesh> gc_mesh;
unique_ptr<VertexPositionGeometry> gc_geom;
unique_ptr<VertexPositionGeometry> gc_geom2d;
unique_ptr<polyscope::SurfaceMesh> ps_mesh;
unique_ptr<polyscope::SurfaceMesh> ps_mesh2d;
string name="";
Eigen::MatrixXf V, VNormals, FNormals;
Eigen::MatrixXi F;
Eigen::VectorXi face_labels, edge_is_borders;
Eigen::VectorXf vK1, vK2;
Eigen::MatrixXf vPD1, vPD2;

// == Params
float regularize_lambda = 0.0;
float p_lower_bound = 0.001, p_upper_bound = 0.999;


bool loadMyMesh(const std::string& mesh_filepath) {
	using vcg::tri::io::Importer;

	// Import mesh
	vcg_mesh = make_unique<MyMesh>();
	int result = Importer<MyMesh>::Open(*vcg_mesh, mesh_filepath.c_str());
	if (result != Importer<MyMesh>::E_NOERROR) {
		cerr << "Error loading mesh: " << Importer<MyMesh>::ErrorMsg(result) << endl;
		return false;
	}

	name = mesh_filepath;
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

	// === Ensure Manifold & Update Topology ===
	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);

	// "Repair non Manifold Vertices by splitting"
	int num = Clean<MyMesh>::SplitNonManifoldVertex(*vcg_mesh, 0);
	cout << format("Splited {} non-manifold vertices.\n", num);
	// "Remove Isolated pieces (wrt Diameter)"
	pair<int, int> comp = Clean<MyMesh>::RemoveSmallConnectedComponentsDiameter(*vcg_mesh, 0.05 * vcg_mesh->bbox.Diag());
	cout << format("Removed {} out of all {} components.\n", comp.second, comp.first);
	// "Repair non Manifold Edges"
	num = Clean<MyMesh>::RemoveNonManifoldFace(*vcg_mesh);
	cout << format("Removed {} non-manifold faces.\n", num);
	// "Remove Unreferenced Vertices"
	num = Clean<MyMesh>::RemoveUnreferencedVertex(*vcg_mesh);
	cout << format("Removed {} unreferenced vertices.\n", num);

	// remove faces incident on deleted vertices
	num = Clean<MyMesh>::RemoveDegenerateVertex(*vcg_mesh);
	cout << format("Removed {} degenerate vertices.\n", num);

	// remove duplicate verts
	num = Clean<MyMesh>::RemoveDuplicateVertex(*vcg_mesh);
	cout << format("Removed {} duplicate verts.\n", num);
	// remove duplicate faces
	num = Clean<MyMesh>::RemoveDuplicateFace(*vcg_mesh);
	cout << format("Removed {} duplicate faces.\n", num);
	// remove duplicate edges
	num = Clean<MyMesh>::RemoveDuplicateEdge(*vcg_mesh);
	cout << format("Removed {} duplicate edges.\n", num);

	// compact vector
	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);

	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexEdge(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);
	// Normals - used by curvature
	UpdateNormal<MyMesh>::PerVertexAngleWeighted(*vcg_mesh);
	UpdateNormal<MyMesh>::PerFaceNormalized(*vcg_mesh);

	return true;
}

bool initPolyscopeAndGeometrycentral() {
	using namespace vcg::tri;

	// Get V, F & label
	V = Eigen::MatrixXf(vcg_mesh->VN(), 3);
	F = Eigen::MatrixXi(vcg_mesh->FN(), 3);
	face_labels = Eigen::VectorXi(vcg_mesh->FN());
	// Normals
	UpdateNormal<MyMesh>::PerVertexAngleWeighted(*vcg_mesh);
	UpdateNormal<MyMesh>::PerFaceNormalized(*vcg_mesh);
	VNormals = Eigen::MatrixXf(vcg_mesh->VN(), 3);
	FNormals = Eigen::MatrixXf(vcg_mesh->FN(), 3);

	// Face handle for labels
	MyMesh::PerFaceAttributeHandle<int> label_handle =
		Allocator<MyMesh>::FindPerFaceAttribute<int>(*vcg_mesh, "face_labels");

	cout << "+";
	int vi = 0, fi = 0;
	for (const auto& v : vcg_mesh->vert) {
		if (!v.IsD()) {
			V.row(vi) = Eigen::Vector3f(v.P()[0], v.P()[1], v.P()[2]);
			VNormals.row(vi) = Eigen::Vector3f(v.N()[0], v.N()[1], v.N()[2]);
			vi++;
		}
	}
	for (const auto& f : vcg_mesh->face) {
		if (!f.IsD()) {
			F.row(fi) = Eigen::Vector3i(
				Index(*vcg_mesh, f.V(0)),
				Index(*vcg_mesh, f.V(1)),
				Index(*vcg_mesh, f.V(2)));
			FNormals.row(fi) = Eigen::Vector3f(f.N()[0], f.N()[1], f.N()[2]);
			face_labels[fi] = label_handle[fi];
			fi++;
		}
	}

	// Print basic info
	try {
	    gc_mesh = unique_ptr<ManifoldSurfaceMesh>(new ManifoldSurfaceMesh(F));
	}
	catch (const std::exception& e) {
		cout << e.what();
	}
	gc_geom = unique_ptr<VertexPositionGeometry>(new VertexPositionGeometry(*gc_mesh, V/*, face_labels*/));
	//cout << format("Euler characteristic = {}, Genus = {}.\n", 
	//	gc_mesh->eulerCharacteristic(), gc_mesh->genus());

	// Init polyscope & geometry-central mesh
	polyscope::view::setNavigateStyle(polyscope::NavigateStyle::Free);
	polyscope::init();
	ps_mesh = std::unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(name, V, F));
	auto ps_permutations = polyscopePermutations(*gc_mesh);
    ps_mesh->setAllPermutations<std::vector<size_t>>(ps_permutations);
	ps_mesh->addFaceScalarQuantity("face_labels", face_labels);


	return true;
}

void execution1() {
	//gc_geom->requireFaceNormals();
	//gc_geom->faceNormals[23189] *= -1;
	gc_geom->requireVertexNormals();
	gc_geom->requireFaceNormals();
	gc_geom->requireFaceGradientOfEigenVector3D();
	gc_geom->requireVertexGradientOfEigenVector3D();

	// Get origin gradients, and rotate 90 degree ccw
	VertexData<Vector3> v_grads(*gc_mesh);
	VertexData<Vector3> v_rotated_grads(*gc_mesh);
	FaceData<Vector3> f_grads(*gc_mesh);
	FaceData<Vector3> f_rotated_grads(*gc_mesh);
	for (Vertex v : gc_mesh->vertices()) {
		v_grads[v] = gc_geom->vertexGradientOfEigenVector3D[v];
		v_rotated_grads[v] = cross(gc_geom->vertexNormals[v], v_grads[v]);
	}
	for (Face f : gc_mesh->faces()) {
		f_grads[f] = gc_geom->faceGradientOfEigenVector3D[f];
		f_rotated_grads[f] = cross(gc_geom->faceNormals[f], f_grads[f]);
	}

	// Compute scalar fields based on normalized gradients
	VertexData<double> x = computeSmoothestVertexScalarField(*gc_geom, v_grads, 0, regularize_lambda, true);
	VertexData<double> y = computeSmoothestVertexScalarField(*gc_geom, v_rotated_grads, 0, regularize_lambda, true);
	Eigen::MatrixXd xy(gc_mesh->nVertices(), 3);
	for (Vertex v : gc_mesh->vertices()) {
		xy.row(v.getIndex()) << x[v], y[v], 0;
	}

	// Visualize xy's 2th derivatives
	Eigen::VectorXd x_vec = x.toVector();
	Eigen::VectorXd y_vec = y.toVector();
	Eigen::VectorXd Lx = gc_geom->cotanLaplacian * x_vec;
	Eigen::VectorXd Ly = gc_geom->cotanLaplacian * y_vec;
	Eigen::VectorXd LN_norm = (gc_geom->cotanLaplacian * VNormals.cast<double>()).rowwise().norm();
	ps_mesh->addVertexScalarQuantity("L * x(normalized)", Lx);
	ps_mesh->addVertexScalarQuantity("L * y(normalized)", Ly);
	ps_mesh->addVertexScalarQuantity("L * Normals", LN_norm);

    // Right-hand-side visualization
	Eigen::SparseVector<double> rhs1(gc_mesh->nVertices()), rhs2(gc_mesh->nVertices());
	for (Vertex vi : gc_mesh->vertices()) {
		Vector3 grad = v_grads[vi], rot_grad = v_rotated_grads[vi];
		for (Halfedge he : vi.outgoingHalfedges()) {
			Vertex vj = he.tipVertex();

			double w = gc_geom->edgeCotanWeights[he.edge()];
			Vector3 x = gc_geom->vertexPositions[vj] - gc_geom->vertexPositions[vi];

			double coef1 = w * dot(x, grad) * 0.5, coef2 = w * dot(x, rot_grad) * 0.5;
			rhs1.coeffRef(vj.getIndex()) += coef1;
			rhs2.coeffRef(vj.getIndex()) += coef2;
			rhs1.coeffRef(vi.getIndex()) -= coef1;
			rhs2.coeffRef(vi.getIndex()) -= coef2;
		}
	}
	ps_mesh->addVertexScalarQuantity("rhs1", rhs1.toDense());
	ps_mesh->addVertexScalarQuantity("rhs2", rhs2.toDense());

	// Translate and scale to similar size
	double scale = (xy.colwise().maxCoeff() - xy.colwise().minCoeff()).squaredNorm() /
		(V.leftCols(2).colwise().maxCoeff() - V.leftCols(2).colwise().minCoeff()).squaredNorm();
	xy /= sqrt(scale);
	Eigen::VectorXd translate = xy.colwise().mean();
	translate(1) -= V.col(1).maxCoeff();
	xy.rowwise() -= translate.transpose();

	// Identify the 'border edges'
	edge_is_borders = Eigen::VectorXi::Zero(gc_mesh->nEdges());
	for (Edge e : gc_mesh->edges()) {
		Halfedge heA = e.halfedge(), heB = heA.twin();
		if (heB == heA) continue;
		
		Face fA = heA.face(), fB = heB.face();
		if (face_labels(fA.getIndex()) != face_labels(fB.getIndex())) {
			edge_is_borders(e.getIndex()) = 1;
		}
	}

	// Visualize
	ps_mesh->addFaceVectorQuantity("Face Normals", gc_geom->faceNormals)->setVectorRadius(0.00005)->setVectorLengthScale(0.001);
	ps_mesh->addVertexVectorQuantity("Vertex Normals", gc_geom->vertexNormals)->setVectorRadius(0.00005)->setVectorLengthScale(0.001);
	ps_mesh->addVertexScalarQuantity("Un-normalized X", gc_geom->cotanLaplacianSmallestEigenvector)->setIsolinesEnabled(true);
	ps_mesh->addVertexScalarQuantity("Normalized X", x)->setIsolinesEnabled(true);
	ps_mesh->addVertexScalarQuantity("Normalized Y", y)->setIsolinesEnabled(true);
	ps_mesh->addVertexVectorQuantity("Gradients V", v_grads)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);
	ps_mesh->addFaceVectorQuantity("Gradients F", f_grads)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);
	ps_mesh->addVertexVectorQuantity("Gradients V rot-90", v_rotated_grads)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);
	ps_mesh->addFaceVectorQuantity("Gradients F rot-90", f_rotated_grads)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);

	// Register new polyscope mesh2d and set permutations
	ps_mesh2d = unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh2D(
		name + " mapped to XY", xy.leftCols(2), gc_mesh->getFaceVertexList()));
	auto ps_permutations = polyscopePermutations(*gc_mesh);
	ps_mesh2d->setAllPermutations<std::vector<size_t>>(ps_permutations);
	ps_mesh2d->setBackFacePolicy(polyscope::BackFacePolicy::Custom);
	//ps_mesh2d->setEdgeColor(glm::vec3(0.2, 0.3, 0.4));
	//ps_mesh2d->setEdgeWidth(0.88);

	// Add quantities to mesh2d
	ps_mesh2d->addFaceScalarQuantity("face labels", face_labels);
	ps_mesh2d->addEdgeScalarQuantity("edge is borders", edge_is_borders, polyscope::DataType::MAGNITUDE);

	// Dual areas ratio
	gc_geom2d = unique_ptr<VertexPositionGeometry>(new VertexPositionGeometry(*gc_mesh, xy));
	gc_geom2d->requireVertexDualAreas();
	gc_geom->requireVertexDualAreas();
	VertexData<double> dualAreasRatio1 = gc_geom->vertexDualAreas / gc_geom2d->vertexDualAreas;
	VertexData<double> dualAreasRatio2 = gc_geom2d->vertexDualAreas / gc_geom->vertexDualAreas;
	// square root
	for (Vertex v : gc_mesh->vertices()) {
		dualAreasRatio1[v] = sqrt(dualAreasRatio1[v]);
		dualAreasRatio2[v] = sqrt(dualAreasRatio2[v]);
	}
	ps_mesh2d->addVertexScalarQuantity("Dual Areas Ratio (3d/2d)", dualAreasRatio1, polyscope::DataType::STANDARD);
	ps_mesh2d->addVertexScalarQuantity("Dual Areas Ratio (2d/3d)", dualAreasRatio2, polyscope::DataType::STANDARD);

	// Face normals as colors
	ps_mesh->addFaceColorQuantity("Normals", gc_geom->faceNormals);
	ps_mesh2d->addFaceColorQuantity("Normals", gc_geom->faceNormals);
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

void computeMeanCurvaturesAndShow() {
	using namespace vcg::tri;

	// compact vector
	Allocator<MyMesh>::CompactVertexVector(*vcg_mesh);
	Allocator<MyMesh>::CompactFaceVector(*vcg_mesh);

	UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
	UpdateTopology<MyMesh>::VertexEdge(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);
	// Normals - used by curvature
	UpdateNormal<MyMesh>::PerVertexAngleWeighted(*vcg_mesh);
	UpdateNormal<MyMesh>::PerFaceNormalized(*vcg_mesh);

	// Calculate
	UpdateNormal<MyMesh>::NormalizePerVertex(*vcg_mesh);
	UpdateBounding<MyMesh>::Box(*vcg_mesh);
	double start = clock();
	UpdateCurvatureFitting<MyMesh>::updateCurvatureLocal(*vcg_mesh, 0.01f * vcg_mesh->bbox.Diag());
	cout << format("{}ms cost by `updateCurvatureLocal()`", clock() - start);

	// Retrieve results
	int vn = vcg_mesh->VN(), i = 0;
	vK1.resize(vn);
	vK2.resize(vn);
	vPD1.resize(vn, 3);
	vPD2.resize(vn, 3);
	ForEachVertex(*vcg_mesh, [&i](const MyMesh::VertexType& v) {
		vK1(i) = v.cK1();
		vK2(i) = v.cK2();
		vPD1.row(i) = v.cPD1().ToEigenVector<Eigen::Vector3f>().normalized();
		vPD2.row(i) = v.cPD2().ToEigenVector<Eigen::Vector3f>().normalized();
		i++;
		});

	// Clamp by percentile
	clampToPercentile(vK1, p_lower_bound, p_upper_bound);
	clampToPercentile(vK2, p_lower_bound, p_upper_bound);

	// Visualize
	ps_mesh->addVertexScalarQuantity("Max curvature", vK1);
	ps_mesh->addVertexScalarQuantity("Min curvature", vK2)->setEnabled(true);
	ps_mesh2d->addVertexScalarQuantity("Max curvature", vK1);
	ps_mesh2d->addVertexScalarQuantity("Min curvature", vK2)->setEnabled(true);
	ps_mesh->addVertexVectorQuantity("Max principal direction", vPD1)->setVectorRadius(0.0005)->setVectorLengthScale(0.005);
	ps_mesh->addVertexVectorQuantity("Min principal direction", vPD2)->setVectorRadius(0.0005)->setVectorLengthScale(0.005);

	// 2nd derivatives
	Eigen::VectorXf LK1 = gc_geom->cotanLaplacian.cast<float>() * vK1, LK2 = gc_geom->cotanLaplacian.cast<float>() * vK2;
	ps_mesh->addVertexScalarQuantity("L * max curv", LK1);
	ps_mesh->addVertexScalarQuantity("L * min curv", LK2);
}

void callback() {
	using namespace ImGui;

	Text("* this must be executed first before any other operations");
	if (Button("Setup Execution")) {
		execution1();
	} SameLine();
	PushItemWidth(130);
	SliderFloat("regularize coef", &regularize_lambda, 0.01, 10.0, "%.1f", ImGuiSliderFlags_Logarithmic);
	PopItemWidth();
	Text("Use 1st eigen vector and its 'perpendicular' component to map to XY coord.");
	Separator();

	if (Button("Compute Mean Curvatures")) {
		computeMeanCurvaturesAndShow();
	} SameLine();
	PushItemWidth(100);
	SliderFloat("lower", &p_lower_bound, 0.0, 0.2, "%.3f", ImGuiSliderFlags_Logarithmic);
	SameLine();
	SliderFloat("upper", &p_upper_bound, 0.8, 1.0, "%.3f", ImGuiSliderFlags_Logarithmic);
	PopItemWidth();
	Separator();

}

int main(int argc, char** argv) {
	if (argc < 2) {
		cerr << format("Usage: {} <mesh file> [<face_labels file>]\n", argv[0]);
		return EXIT_FAILURE;
	}

	const string mesh_fp = string(argv[1]);
	const string label_fp = argc >= 3 ? string(argv[2]) : "";

	if (!loadMyMesh(mesh_fp)) return EXIT_FAILURE;
	if (!loadLabel(label_fp)) return EXIT_FAILURE;
	//if (!repairMesh()) return EXIT_FAILURE;
	if (!initPolyscopeAndGeometrycentral()) return EXIT_FAILURE;

	polyscope::state::userCallback = callback;
	polyscope::show();

	return EXIT_SUCCESS;
}