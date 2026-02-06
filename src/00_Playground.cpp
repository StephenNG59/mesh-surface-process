//// === Polyscope ===
//#include "polyscope/polyscope.h"
//
//// === Geometry-Central ===
//#include "geometrycentral/surface/manifold_surface_mesh.h"
//#include "geometrycentral/surface/vertex_position_geometry.h"
//#include "geometrycentral/surface/meshio.h"
//
//// === Eigen ===
//#include <Eigen/Core>
//
//// === libigl ===
//#include <igl/writeOBJ.h>
//
//// === Spectra ===
//#include <Spectra/SymGEigsShiftSolver.h>

#include "mesh_processor.h"
#include <iostream>
#include <functional>
#include <polyscope/polyscope.h>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << std::format("Usage: {} <mesh file> [<face_labels file>]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string mesh_fp = std::string(argv[1]);
    const std::string label_fp = argc >= 3 ? std::string(argv[2]) : "";
    MeshProcessor mp(mesh_fp, label_fp);

    std::function<void()> myCallback = std::bind(&MeshProcessor::uiCallback, &mp);
    polyscope::state::userCallback = myCallback;
    polyscope::show();

    return EXIT_SUCCESS;
}