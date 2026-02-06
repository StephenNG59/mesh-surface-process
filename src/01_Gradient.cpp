// === Eigen ===
#include <Eigen/core>
// === polyscope ===
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
// === geometry-central ===
#include <geometrycentral/surface/meshio.h>
#include <geometrycentral/surface/global_semantic_geometry.h>
#include <geometrycentral/surface/direction_fields.h>
#include <geometrycentral/surface/vector_heat_method.h>
#include <geometrycentral/surface/scalar_fields.h>
#include <geometrycentral/numerical/linear_solvers.h>


using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

unique_ptr<ManifoldSurfaceMesh> gc_mesh;
unique_ptr<VertexPositionGeometry> gc_geom_origin;
unique_ptr<VertexPositionGeometry> gc_geom_new;
unique_ptr<polyscope::SurfaceMesh> ps_mesh;
unique_ptr<polyscope::SurfaceMesh> ps_mesh_1;
unique_ptr<polyscope::SurfaceMesh> ps_mesh_2;
unique_ptr<polyscope::SurfaceMesh> ps_mesh_3;
unique_ptr<polyscope::SurfaceMesh> ps_mesh_4;

// == Global params
int direction_field_n_sym = 1;
int transport_source_vid = 123;
int kth_eigen_vector = 1;


void computeEigenvectorAndGradient() {
	// kth smallest eigen vector
	gc_geom_origin->requireCotanLaplacianSmallestEigenvector();
	ps_mesh->addVertexScalarQuantity("Eigen Vector", gc_geom_origin->cotanLaplacianSmallestEigenvector)->setIsolinesEnabled(true)->setEnabled(true);

	// Gradient of eigen vector
	gc_geom_origin->requireVertexGradientOfEigenVector3D();
	ps_mesh->addVertexVectorQuantity("Gradient of Eigen Vector", gc_geom_origin->vertexGradientOfEigenVector3D)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);
}

void computeSmoothestDirectionFieldAndParallelTransport() {
	// Tangent basis
	gc_geom_origin->requireVertexTangentBasis();
	VertexData<Vector3> v_basis_x(*gc_mesh), v_basis_y(*gc_mesh);
	for (Vertex v : gc_mesh->vertices()) {
		v_basis_x[v] = gc_geom_origin->vertexTangentBasis[v][0];
		v_basis_y[v] = gc_geom_origin->vertexTangentBasis[v][1];
	}

	transport_source_vid = min(int(gc_mesh->nVertices() - 1), transport_source_vid);
	Vertex source_v = gc_mesh->vertex(transport_source_vid);
	

	// == Using origin transport vectors along halfedge
	
	// Smoothest direction field
	VertexData<Vector2> v_smooth_dirs = computeSmoothestVertexDirectionField(*gc_geom_origin, direction_field_n_sym);
	ps_mesh->addVertexTangentVectorQuantity("origin|Smoothest Direction Field",
		v_smooth_dirs, v_basis_x, v_basis_y, direction_field_n_sym
	)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);

	// Parallel transport with specified source
	VectorHeatMethodSolver vhm_solver(*gc_geom_origin);
	VertexData<Vector2> v_transported_vecs = vhm_solver.transportTangentVector(
		source_v, v_smooth_dirs[source_v]);
	ps_mesh->addVertexTangentVectorQuantity("origin|Parallel Transported Vectors",
		v_transported_vecs, v_basis_x, v_basis_y, direction_field_n_sym
	)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);

	// Difference between two
	VertexData<float> v_diff_angle(*gc_mesh, 0.f);
	for (Vertex v : gc_mesh->vertices()) {
		float phi1 = atan2(v_smooth_dirs[v][1], v_smooth_dirs[v][0]);
		float phi2 = atan2(v_transported_vecs[v][1], v_transported_vecs[v][0]);
		v_diff_angle[v] = regularizeAngle(phi2 - phi1);
	}
	ps_mesh->addVertexScalarQuantity("origin|Angle Difference",
		v_diff_angle, polyscope::DataType::MAGNITUDE
	)->setColorMap("phase");


	// == Using new transport vectors along halfedge

	// Smoothest direction field
	v_smooth_dirs = computeSmoothestVertexDirectionField(*gc_geom_new, direction_field_n_sym);
	ps_mesh->addVertexTangentVectorQuantity("new|Smoothest Direction Field",
		v_smooth_dirs, v_basis_x, v_basis_y, direction_field_n_sym
	)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);

	// Parallel transport with specified source
	VectorHeatMethodSolver vhm_solver_new(*gc_geom_new);
	v_transported_vecs = vhm_solver_new.transportTangentVector(
		source_v, v_smooth_dirs[source_v]);
	ps_mesh->addVertexTangentVectorQuantity("new|Parallel Transported Vectors",
		v_transported_vecs, v_basis_x, v_basis_y, direction_field_n_sym
	)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);

	// Difference between two
	for (Vertex v : gc_mesh->vertices()) {
		float phi1 = atan2(v_smooth_dirs[v][1], v_smooth_dirs[v][0]);
		float phi2 = atan2(v_transported_vecs[v][1], v_transported_vecs[v][0]);
		v_diff_angle[v] = regularizeAngle(phi2 - phi1);
	}
	ps_mesh->addVertexScalarQuantity("new|Angle Difference",
		v_diff_angle, polyscope::DataType::MAGNITUDE
	)->setColorMap("phase");


	// Show the source point
	vector<Vector3> cloud;
	cloud.push_back(gc_geom_origin->vertexPositions[transport_source_vid]);
	polyscope::PointCloud* pointnet = polyscope::registerPointCloud("source vertex", cloud);
	pointnet->setPointRadius(.005)->setEnabled(true);
}

void computeSmoothestScalarFieldAndMapToXY() {
	gc_geom_origin->requireVertexNormals();
	gc_geom_origin->requireVertexGradientOfEigenVector3D();

	// Rotate gradient 90 degree ccw
	VertexData<Vector3> gradients(*gc_mesh);
	for (Vertex v : gc_mesh->vertices()) {
		gradients[v] = cross(
			gc_geom_origin->vertexNormals[v], gc_geom_origin->vertexGradientOfEigenVector3D[v]
		);
	}
	ps_mesh->addVertexVectorQuantity("Rot-90 gradients", gradients)->setVectorRadius(0.0005)->setVectorLengthScale(0.015);

	// Compute scalar based on gradients
	VertexData<double> y = computeSmoothestVertexScalarField(*gc_geom_origin, gradients, 0, false);
	ps_mesh->addVertexScalarQuantity("Rot-90 scalars (un-normalized)", y)->setIsolinesEnabled(true);

	VertexData<double> y_even = computeSmoothestVertexScalarField(*gc_geom_origin, gradients, 0, true);
	ps_mesh->addVertexScalarQuantity("Rot-90 scalars (normalized)", y_even)->setIsolinesEnabled(true);

	// 
	VertexData<double> x = computeSmoothestVertexScalarField(*gc_geom_origin, gc_geom_origin->vertexGradientOfEigenVector3D, 0, false);
	ps_mesh->addVertexScalarQuantity("Remade scalars (un-normalized)", x)->setIsolinesEnabled(true);

	VertexData<double> x_even = computeSmoothestVertexScalarField(*gc_geom_origin, gc_geom_origin->vertexGradientOfEigenVector3D, 0, true);
	ps_mesh->addVertexScalarQuantity("Remade scalars (normalized)", x_even)->setIsolinesEnabled(true);


	// Reposition mesh to x-y
	Eigen::MatrixXd V0(gc_mesh->nVertices(), 3), V1(gc_mesh->nVertices(), 3), V2(gc_mesh->nVertices(), 3);
	for (Vertex v : gc_mesh->vertices()) {
		size_t vi = v.getIndex();
		Vector3 pos = gc_geom_origin->vertexPositions[v];
		V0.row(vi) << pos.x, pos.y, 0;
		V1.row(vi) << x[v], y[v], 0;
		V2.row(vi) << x_even[v], y_even[v], 0;
	}

	// Translate to origin & Scale to similar size
	Eigen::VectorXd translate0 = V0.colwise().mean(), translate1 = V1.colwise().mean(), translate2 = V2.colwise().mean();
	V1.rowwise() -= translate1.transpose();
	V2.rowwise() -= translate2.transpose();

	double scale0 = (V0.colwise().maxCoeff() - V0.colwise().minCoeff()).norm();
	double scale1 = (V1.colwise().maxCoeff() - V1.colwise().minCoeff()).norm() / scale0;
	double scale2 = (V2.colwise().maxCoeff() - V2.colwise().minCoeff()).norm() / scale0;
	V1 /= scale1;
	V2 /= scale2;

	//// Add the z-coord
	//for (Vertex v : gc_mesh->vertices()) {
	//	size_t vi = v.getIndex();
	//	Vector3 pos = gc_geom_origin->vertexPositions[v];
	//	V1(vi, 2) = pos.z;
	//	V2(vi, 2) = pos.z;
	//}

	//// Stretch V1
	//for (int i = 0; i < V1.rows(); i++) {
	//	V1(i, 0) *= sqrt(abs(V1(i, 0)) + 1);
	//	V1(i, 1) *= sqrt(abs(V1(i, 0)) + 1);
	//}

	ps_mesh_1 = unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(
		"Direct Projected to XY", V0, gc_mesh->getFaceVertexList()));
	ps_mesh_2 = unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(
		"Eigen Mapped to XY (un-normalized)", V1, gc_mesh->getFaceVertexList()));
	ps_mesh_3 = unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(
		"Eigen Mapped to XY (normalized)", V2, gc_mesh->getFaceVertexList()));
	ps_mesh_1->setEnabled(false);
	ps_mesh_2->setEnabled(false);
	ps_mesh_3->setEnabled(false);
}

void computeSimpleEigenMapToXY() {
	gc_geom_origin->requireCotanLaplacian();
	gc_geom_origin->requireVertexGalerkinMassMatrix();

	std::vector<Vector<double>> smallestKEigenvectors =
		geometrycentral::smallestKEigenvectorsPositiveDefinite(
			gc_geom_origin->cotanLaplacian, 
			gc_geom_origin->vertexGalerkinMassMatrix,
			3);
	Vector<double> eval_2 = smallestKEigenvectors[1];
	Vector<double> eval_3 = smallestKEigenvectors[2];

	// Visualize eigen vectors
	ps_mesh->addVertexScalarQuantity("2nd eigen vector", eval_2)->setIsolinesEnabled(true);
	ps_mesh->addVertexScalarQuantity("3rd eigen vector", eval_3)->setIsolinesEnabled(true);

	// Reposition mesh to x-y
	Eigen::MatrixXd V0(gc_mesh->nVertices(), 3);
	for (Vertex v : gc_mesh->vertices()) {
		size_t vi = v.getIndex();
		Vector3 pos = gc_geom_origin->vertexPositions[v];
		V0.row(vi) << pos.x, pos.y, 0;
	}

	double scale = max(eval_2.maxCoeff() - eval_2.minCoeff(), eval_3.maxCoeff() - eval_3.minCoeff()) / 
		(V0.colwise().maxCoeff() - V0.colwise().minCoeff()).maxCoeff();

	// Reposition mesh to x-y
	Eigen::MatrixXd V(gc_mesh->nVertices(), 3);
	for (Vertex v : gc_mesh->vertices()) {
		size_t vi = v.getIndex();
		V.row(vi) << eval_2[vi] / scale, eval_3[vi] / scale, 0;
	}

	ps_mesh_4 = unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(
		"Simple Eigen Mapped to XY", V, gc_mesh->getFaceVertexList()));

}

void callback() {
	using namespace ImGui;

	Text("These settings should be specified *before* any executions.");
	PushItemWidth(240);
	if (SliderInt("smallest eigen vector", &kth_eigen_vector, 0, 7, "Use the %dth", ImGuiSliderFlags_AlwaysClamp)) {
		gc_geom_origin->useKthSmallestEigenvector = kth_eigen_vector;
		gc_geom_new->useKthSmallestEigenvector = kth_eigen_vector;
	}
	PopItemWidth();
	Separator();


	if (Button("Execution 1")) {
		computeEigenvectorAndGradient();
	}
	Text("1) Compute the kth smallest eigen vector of Laplacian matrix;");
	Text("2) Compute the gradient of this eigen vector.");
	Separator();


	if (Button("Execution 2")) {
		computeSmoothestDirectionFieldAndParallelTransport();
	} SameLine();
	PushItemWidth(160);
	SliderInt("source vertex", &transport_source_vid, 0, gc_mesh->nVertices(), "%d", ImGuiSliderFlags_AlwaysClamp);
	PopItemWidth();
	Text("For origin and new transportVectorsAlongHalfedge(), do the following respectively:");
	Text("1) Compute smoothest direction field;");
	Text("2) Pick a source vertex and parallel transport it;");
    Text("3) Visualize the angle difference between 2 fields.");
	Text("* origin - use surface geometry to transport vectors along halfedge");
	Text("* new - use the gradient of eigen vector to transport vectors along halfedge");
	Separator();


	if (Button("Execution 3")) {
		computeSmoothestScalarFieldAndMapToXY();
	}
	Text("1) Compute gradient of the kth smallest eigen vector of Laplacian matrix;");
	Text("2) Compute a scalar field that follows the gradient best;");
	Text("   (basically, something like the 'normalized' eigen vector')");
	Text("3) Rotate gradient vectors 90 degrees ccw;");
	Text("4) Compute a scalar field that follows the rotated gradient best;");
	Text("5) Use scalars in 2) as x & scalars in 4) as y, mapping mesh surface to x-y plane.");
	Text("   (visualize another mapping for comparison: directly eliminate origin z-component)");
	Separator();


	if (Button("Execution 4")) {
		computeSimpleEigenMapToXY();
	}
	Text("Use 1st and 2nd (meaningful)eigen vector as xy coord.");
	Separator();
}


int main(int argc, char** argv) {
	if (argc < 2) {
		cerr << format("Usage: {} <mesh file>", argv[0]);
		return EXIT_FAILURE;
	}

	const string mesh_fp(argv[1]);
	tie(gc_mesh, gc_geom_origin) = readManifoldSurfaceMesh(mesh_fp);
	gc_geom_new = gc_geom_origin->copy();
	gc_geom_new->useOriginTransportVectorsAlongHalfedge = false;

	polyscope::view::setNavigateStyle(polyscope::NavigateStyle::Free);
	polyscope::init();
	ps_mesh = unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(mesh_fp, 
			gc_geom_origin->inputVertexPositions, 
			gc_mesh->getFaceVertexList()));
	//auto ps_permutations = polyscopePermutations(*gc_mesh);
	//ps_mesh->setAllPermutations<std::vector<size_t>>(ps_permutations);

	polyscope::state::userCallback = callback;
	polyscope::show();

	return EXIT_SUCCESS;
}
