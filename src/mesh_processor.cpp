#include "mesh_processor.h"

using namespace geometrycentral::surface;

// Define the MyMesh structure
class  MyVertex;
class  MyFace;
class  MyEdge;
struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>     ::AsVertexType,
                                           vcg::Use<MyFace>       ::AsFaceType,
                                           vcg::Use<MyEdge>       ::AsEdgeType> {};
class  MyVertex : public vcg::Vertex<MyUsedTypes,
                                     vcg::vertex::Coord3f,
                                     vcg::vertex::Normal3f,
                                     vcg::vertex::VFAdj,
                                     vcg::vertex::VEAdj,
                                     vcg::vertex::CurvatureDirf,
                                     vcg::vertex::Qualityf,
                                     vcg::vertex::Mark,
                                     vcg::vertex::BitFlags> {};
class  MyFace : public vcg::Face<MyUsedTypes,
                                 vcg::face::Normal3f,
                                 vcg::face::VFAdj,
                                 vcg::face::FFAdj,
                                 vcg::face::VertexRef,
                                 vcg::face::Mark,
                                 vcg::face::BitFlags> {};
class  MyEdge : public vcg::Edge<MyUsedTypes,
                                 vcg::edge::VEAdj,
                                 vcg::edge::VertexRef,
                                 vcg::edge::Mark,
                                 vcg::edge::BitFlags> {};
class  MyMesh : public vcg::tri::TriMesh<std::vector<MyVertex>,
                                         std::vector<MyFace>,
                                         std::vector<MyEdge> > {};


MeshProcessor::MeshProcessor(const std::string& mesh_filepath, const std::string& label_filepath="") {
    vcg_mesh = std::make_unique<MyMesh>();

    if (!loadMesh(mesh_filepath)) return;

    if (!loadLabel(label_filepath)) return;

    if (!repairMesh()) return;

    if (!initPolyscopeAndGeometrycentral()) return;

}
MeshProcessor::~MeshProcessor() = default;

bool MeshProcessor::loadMesh(const std::string& mesh_filepath) {
    using vcg::tri::io::Importer;

    // Import mesh
    int result = Importer<MyMesh>::Open(*vcg_mesh, mesh_filepath.c_str());
    if (result != Importer<MyMesh>::E_NOERROR) {
        std::cerr << "Error loading mesh: " << Importer<MyMesh>::ErrorMsg(result) << std::endl;
        return false;
    }

    name = mesh_filepath;
    std::cout << std::format("{} sucessfully loaded with {} faces & {} verts.\n",
        mesh_filepath, vcg_mesh->FN(), vcg_mesh->VN());

    return true;
}

bool MeshProcessor::loadLabel(const std::string& label_filepath) {
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

bool MeshProcessor::repairMesh() {
    // === Ensure Manifold & Update Topology ===
    vcg::tri::UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
    vcg::tri::UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
    vcg::tri::UpdateBounding<MyMesh>::Box(*vcg_mesh);

    // "Repair non Manifold Vertices by splitting"
    int num = vcg::tri::Clean<MyMesh>::SplitNonManifoldVertex(*vcg_mesh, 0);
    std::cout << std::format("Splited {} non-manifold vertices.\n", num);
    // "Remove Isolated pieces (wrt Diameter)"
    std::pair<int, int> comp = vcg::tri::Clean<MyMesh>::RemoveSmallConnectedComponentsDiameter(*vcg_mesh, 0.05 * vcg_mesh->bbox.Diag());
    std::cout << std::format("Removed {} out of all {} components.\n", comp.second, comp.first);
    // "Repair non Manifold Edges"
    num = vcg::tri::Clean<MyMesh>::RemoveNonManifoldFace(*vcg_mesh);
    std::cout << std::format("Removed {} non-manifold faces.\n", num);
    // "Remove Unreferenced Vertices"
    num = vcg::tri::Clean<MyMesh>::RemoveUnreferencedVertex(*vcg_mesh);
    std::cout << std::format("Removed {} unreferenced vertices.\n", num);

    vcg::tri::UpdateTopology<MyMesh>::FaceFace(*vcg_mesh);
    vcg::tri::UpdateTopology<MyMesh>::VertexFace(*vcg_mesh);
    vcg::tri::UpdateTopology<MyMesh>::VertexEdge(*vcg_mesh);
    vcg::tri::UpdateBounding<MyMesh>::Box(*vcg_mesh);
    // Normals - used by curvature
    vcg::tri::UpdateNormal<MyMesh>::PerVertexAngleWeighted(*vcg_mesh);
    vcg::tri::UpdateNormal<MyMesh>::PerFaceNormalized(*vcg_mesh);

    return true;
}

bool MeshProcessor::initPolyscopeAndGeometrycentral() {
    // Get V, F & label
    V = Eigen::MatrixXf(vcg_mesh->VN(), 3);
    F = Eigen::MatrixXi(vcg_mesh->FN(), 3);
    face_labels = Eigen::VectorXi(vcg_mesh->FN());

    // Face handle for labels
    MyMesh::PerFaceAttributeHandle<int> label_handle =
        vcg::tri::Allocator<MyMesh>::FindPerFaceAttribute<int>(*vcg_mesh, "face_labels");

    int vi = 0, fi = 0;
    for (const auto& v : vcg_mesh->vert) {
        if (!v.IsD()) {
            V.row(vi++) = Eigen::Vector3f(v.P()[0], v.P()[1], v.P()[2]);
        }
    }
    for (const auto& f : vcg_mesh->face) {
        if (!f.IsD()) {
            F.row(fi) = Eigen::Vector3i(
                vcg::tri::Index(*vcg_mesh, f.V(0)),
                vcg::tri::Index(*vcg_mesh, f.V(1)),
                vcg::tri::Index(*vcg_mesh, f.V(2)));
            face_labels[fi] = label_handle[fi];
            fi++;
        }
    }

    // Init polyscope & geometry-central mesh
    polyscope::view::setNavigateStyle(polyscope::NavigateStyle::Free);
    polyscope::init();
    gc_mesh = std::unique_ptr<ManifoldSurfaceMesh>(new ManifoldSurfaceMesh(F));
    gc_geom = std::unique_ptr<VertexPositionGeometry>(new VertexPositionGeometry(*gc_mesh, V/*, face_labels*/));
    ps_mesh = std::unique_ptr<polyscope::SurfaceMesh>(polyscope::registerSurfaceMesh(name, V, F));
    ps_mesh->addFaceScalarQuantity("face_labels", face_labels);

    return true;
}


void MeshProcessor::uiCallback() {
    using namespace ImGui;
    //if (Button("Calculate Curvature")) {
    //    computeVertexCurvature(true);
    //}

    if (Button("Calculate Gradient of kth Eigen Vector")) {
        computeFaceVertexGradient();
    }

    if (Button("Smoothest direction field")) {
        computeSmoothesDirectionField();
    }

    if (Button("Transport 1 vector")) {
        computeTransport1Vector();
    }
    
    InputInt("unrequire times", &unrequire_times, 1, 2);
    if (Button("Logarithmic map")) {
        compute1SourceLogMap();
    }

    //if (Button("Karcher mean of random label")) {
    //    compute1LabelKarcherMean();
    //}

    InputInt("label", &center_label, 1, 2);
    if (Button("Compute specified label centers (p=1&2)")) {
        computeLabelMedianAndMean(center_label);
    }

}


//void MeshProcessor::computeVertexCurvature(const bool show) {
//    // Calculate
//    vcg::tri::UpdateNormal<MyMesh>::NormalizePerVertex(*vcg_mesh);
//    vcg::tri::UpdateBounding<MyMesh>::Box(*vcg_mesh);
//    vcg::tri::UpdateCurvatureFitting<MyMesh>::updateCurvatureLocal(*vcg_mesh, 0.01f * vcg_mesh->bbox.Diag());
//
//    // Retrieve results
//    const int vn = vcg_mesh->VN();
//    std::vector<float> Q, K1, K2;
//    Eigen::MatrixXf PD1(vn, 3), PD2(vn, 3);
//
//    int i = 0;
//    vcg::tri::ForEachVertex(*vcg_mesh, [&Q, &K1, &K2, &i, &PD1, &PD2](const MyMesh::VertexType& v) {
//        Q.push_back(v.cQ());
//        K1.push_back(v.cK1());
//        K2.push_back(v.cK2());
//        PD1.row(i) = v.cPD1().ToEigenVector<Eigen::Vector3f>().normalized();
//        PD2.row(i) = v.cPD2().ToEigenVector<Eigen::Vector3f>().normalized();
//        i++;
//        });
//
//    // Add quantities
//    if (show) {
//        ps_mesh->addVertexScalarQuantity("V Curv-max", K1);
//        ps_mesh->addVertexScalarQuantity("V Curv-min", K2);
//        ps_mesh->addVertexScalarQuantity("V Curv", Q);
//        ps_mesh->addVertexVectorQuantity("V Curv-max-PD", PD1)->setVectorRadius(.0005);
//        ps_mesh->addVertexVectorQuantity("V Curv-min-PD", PD2)->setVectorRadius(.0005);
//    }
//}

void MeshProcessor::computeFaceVertexGradient() {

    // === face 3d
    gc_geom->requireFaceGradientOfEigenVector3D();
    ps_mesh->addFaceVectorQuantity("face gradient of eigen vector 3d",
        gc_geom->faceGradientOfEigenVector3D)->setVectorRadius(0.001);

    // === face 2d
    gc_geom->requireFaceTangentBasis();
    FaceData<geometrycentral::Vector3> f_basis_x(*gc_mesh), f_basis_y(*gc_mesh);
    for (Face f : gc_mesh->faces()) {
        f_basis_x[f] = gc_geom->faceTangentBasis[f][0];
        f_basis_y[f] = gc_geom->faceTangentBasis[f][1];
    }
    gc_geom->requireFaceGradientOfEigenVector2D();
    ps_mesh->addFaceTangentVectorQuantity("face gradient of eigen vector 2d", 
        gc_geom->faceGradientOfEigenVector2D, f_basis_x, f_basis_y)->setVectorRadius(0.001);

    // === vertex 3d
    gc_geom->requireVertexGradientOfEigenVector3D();
    ps_mesh->addVertexVectorQuantity("vertex gradient of eigen vector 3d",
        gc_geom->vertexGradientOfEigenVector3D)->setVectorRadius(0.001);

    // === vertex 2d
    gc_geom->requireVertexTangentBasis();
    VertexData<geometrycentral::Vector3> v_basis_x(*gc_mesh), v_basis_y(*gc_mesh);
    for (Vertex v : gc_mesh->vertices()) {
        v_basis_x[v] = gc_geom->vertexTangentBasis[v][0];
        v_basis_y[v] = gc_geom->vertexTangentBasis[v][1];
    }
    gc_geom->requireVertexGradientOfEigenVector2D();
    ps_mesh->addVertexTangentVectorQuantity("vertex gradient of eigen vector 2d",
        gc_geom->vertexGradientOfEigenVector2D, v_basis_x, v_basis_y)->setVectorRadius(0.001);

    // === visualize eigen vector
    ps_mesh->addVertexScalarQuantity("eigen vector", gc_geom->cotanLaplacianSmallestEigenvector);
}


void MeshProcessor::compute1SourceLogMap() {

    VectorHeatMethodSolver vhm_solver(*gc_geom);

    // Pick a random vertex as source and compute logarithmic map
    size_t vid = geometrycentral::randomIndex(gc_mesh->nVertices());
    
    // 1) origin
    gc_geom->useOriginTransportVectorsAlongHalfedge = false;
    VertexData<geometrycentral::Vector2> log_map_origin = 
        vhm_solver.computeLogMap(gc_mesh->vertex(vid));
    // 2) new
    for (int i = 0; i < unrequire_times; i++) {
        gc_geom->unrequireTransportVectorsAlongHalfedge();
        std::cout << i << " unrequire done." << std::endl;
    }
    gc_geom->useOriginTransportVectorsAlongHalfedge = true;
    gc_geom->requireTransportVectorsAlongHalfedge();
    VertexData<geometrycentral::Vector2> log_map_new =
        vhm_solver.computeLogMap(gc_mesh->vertex(vid));

    // Convert to angle & distance
    VertexData<float> angle_origin(*gc_mesh), distance_origin(*gc_mesh);
    VertexData<float> angle_new(*gc_mesh), distance_new(*gc_mesh);
    for (Vertex v : gc_mesh->vertices()) {
        angle_origin[v] = atan2(log_map_origin[v][1], log_map_origin[v][0]) * 180 / M_PI;
        if (angle_origin[v] < 0) angle_origin[v] += 360;
        distance_origin[v] = log_map_origin[v].norm();

        angle_new[v] = atan2(log_map_new[v][1], log_map_new[v][0]) * 180 / M_PI;
        if (angle_new[v] < 0) angle_new[v] += 360;
        distance_new[v] = log_map_new[v].norm();
    }
    ps_mesh->addVertexScalarQuantity("log map angle-origin", angle_origin)->setColorMap("phase")->setEnabled(true);
    ps_mesh->addVertexScalarQuantity("log map angle-new", angle_new)->setColorMap("phase")->setEnabled(true);
    ps_mesh->addVertexScalarQuantity("log map distance-origin", distance_origin);
    ps_mesh->addVertexScalarQuantity("log map distance-new", distance_new);

    // Show the source point
    std::vector<geometrycentral::Vector3> cloud;
    cloud.push_back(gc_geom->vertexPositions[vid]);
    polyscope::PointCloud* pointnet = polyscope::registerPointCloud("log map source", cloud);
    pointnet->setPointRadius(.005)->setEnabled(true);
}

//void MeshProcessor::compute1LabelKarcherMean() {
//    // Pick a random face with label (label probably != 0)
//    size_t fid = geometrycentral::randomIndex(gc_mesh->nFaces()), attempts = 0;
//    while (face_labels[fid] == 0 && attempts <= 5) {
//        fid = geometrycentral::randomIndex(gc_mesh->nFaces());
//        attempts++;
//    }
//    const int label = face_labels[fid];
//    std::cout << std::format("source face #{} with label {}", fid, label);
//
//    // Get all vertices that touches any face of this label
//    std::vector<Vertex> source_verts;
//    for (Vertex v : gc_mesh->vertices()) {
//        bool is_same_label = false;
//        for (Face f : v.adjacentFaces()) {
//            if (face_labels[f.getIndex()] == label) is_same_label = true;
//        }
//        
//        if (is_same_label) {
//            source_verts.push_back(v);
//        }
//    }
//
//    SurfacePoint pt = findCenter(*gc_mesh, *gc_geom, source_verts, 2);
//
//    // Show center point
//    std::vector<geometrycentral::Vector3> cloud;
//    geometrycentral::Vector3 pt_coord = pt.interpolate(gc_geom->inputVertexPositions);
//    cloud.push_back(pt_coord);
//    polyscope::PointCloud* pointnet = polyscope::registerPointCloud("karcher mean", cloud);
//    pointnet->setPointRadius(.005)->setEnabled(true);
//
//
//    // == Compute log map from karcher mean
//    VectorHeatMethodSolver vhm_solver(*gc_geom);
//    VertexData<geometrycentral::Vector2> log_map = vhm_solver.computeLogMap(pt);
//
//    // Convert to angle & distance
//    VertexData<float> angle(*gc_mesh), distance(*gc_mesh);
//    for (Vertex v : gc_mesh->vertices()) {
//        angle[v] = atan2(log_map[v][1], log_map[v][0]) * 180 / M_PI;
//        distance[v] = log_map[v].norm();
//    }
//    ps_mesh->addVertexScalarQuantity("log map angle", angle)->setColorMap("phase")->setEnabled(true);
//    ps_mesh->addVertexScalarQuantity("log map distance", distance);
//}


void MeshProcessor::computeLabelMedianAndMean(const int label) {
    if (!(face_labels.array() == label).any()) {
        std::cerr << std::format("No face has label {}, computation breaks.", label);
        return;
    }

    // Get all vertices that touches any face of this label
    std::vector<Vertex> source_verts;
    for (Vertex v : gc_mesh->vertices()) {
        bool is_same_label = false;
        for (Face f : v.adjacentFaces()) {
            if (face_labels[f.getIndex()] == label) {
                is_same_label = true;
                break;
            }
        }
        if (is_same_label) {
            source_verts.push_back(v);
        }
    }

    // Compute median & mean
    SurfacePoint sp_median = findCenter(*gc_mesh, *gc_geom, source_verts, 1);
    SurfacePoint sp_mean   = findCenter(*gc_mesh, *gc_geom, source_verts, 2);
    showPoint(sp_median, std::format("median of [{}]", label));
    showPoint(sp_mean,   std::format("mean of [{}]",   label));
}


void MeshProcessor::showPoint(const geometrycentral::surface::SurfacePoint& sp, const std::string& name) {
    std::vector<geometrycentral::Vector3> cloud;
    cloud.push_back(sp.interpolate(gc_geom->inputVertexPositions));

    polyscope::PointCloud* pointnet = polyscope::registerPointCloud(name, cloud);
    pointnet->setPointRadius(.005)->setEnabled(true);
}