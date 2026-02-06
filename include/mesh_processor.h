#ifndef MESH_PROCESSOR_H
#define MESH_PROCESSOR_H

#include <string>
#include <iostream>
#include <format>
#include <complex>
// === Eigen ===
#include <Eigen/core>
// === polyscope ===
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
// === geometry-central ===
#include <geometrycentral/surface/manifold_surface_mesh.h>
//#include <geometrycentral/surface/global_semantic_geometry.h>
#include <geometrycentral/surface/vertex_position_geometry.h>
#include <geometrycentral/surface/direction_fields.h>
#include <geometrycentral/surface/vector_heat_method.h>
#include <geometrycentral/surface/surface_centers.h>
// === vcglib ===
#include <wrap/io_trimesh/import.h>
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/curvature_fitting.h>
//// === libigl ===
//#include <igl/massmatrix.h>
//#include <igl/grad.h>
//#include <igl/per_vertex_normals.h>
//// === Spectra ===
//#include <Spectra/SymGEigsShiftSolver.h>


class MyMesh; 

class MeshProcessor {
public:
	MeshProcessor();
	MeshProcessor(const std::string& mesh_filepath, const std::string& label_filepath);
	~MeshProcessor();

	void uiCallback();

	//void computeVertexCurvature(const bool show);
	void computeFaceVertexGradient();
	void compute1SourceLogMap();
	//void compute1LabelKarcherMean();
	void computeLabelMedianAndMean(const int label);

public:
	Eigen::MatrixXf V;
	Eigen::MatrixXi F;
	Eigen::VectorXi face_labels;


private:
	std::string name;
	std::unique_ptr<MyMesh> vcg_mesh;
	std::unique_ptr<polyscope::SurfaceMesh> ps_mesh;
	std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh> gc_mesh;
	std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> gc_geom;

	bool loadMesh(const std::string& mesh_filepath);
	bool loadLabel(const std::string& label_filepath);
	bool repairMesh();
	bool initPolyscopeAndGeometrycentral();


private:

	// == ui related
	int center_label = 0;
	int unrequire_times = 2;
	void showPoint(const geometrycentral::surface::SurfacePoint& sp, const std::string& name);
};

#endif // !MESH_PROCESSOR_H