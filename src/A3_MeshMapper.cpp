#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <igl/readOBJ.h>
#include <igl/arap.h>
#include <igl/per_vertex_normals.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/principal_curvature.h>

#include <cnpy.h>

#include "log.h"
#include "cxxopts.hpp"

using namespace std;

/*
//#include "json.hpp"
//using json = nlohmann::json;
json to_json_matrix(const Eigen::MatrixXd& M) {
    json j = json::array();
    for (int i = 0; i < M.rows(); ++i) {
        json row = json::array();
        for (int jcol = 0; jcol < M.cols(); ++jcol) {
            row.push_back(M(i, jcol));
        }
        j.push_back(row);
    }
    return j;
}

json to_json_matrix(const Eigen::MatrixXi& M) {
    json j = json::array();
    for (int i = 0; i < M.rows(); ++i) {
        json row = json::array();
        for (int jcol = 0; jcol < M.cols(); ++jcol) {
            row.push_back(M(i, jcol));
        }
        j.push_back(row);
    }
    return j;
}

json to_json_vector(const Eigen::VectorXi& v) {
    json j = json::array();
    for (int i = 0; i < v.size(); ++i) j.push_back(v[i]);
    return j;
}
*/

// == Params
string mesh_fp;
string output_dir;
string method;
unsigned int curv_ring;


bool readMFile(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F,
    Eigen::VectorXi& labels, Eigen::VectorXi& instances) {

    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> faces;
    std::vector<int> label_list, instance_list;
    std::unordered_map<int, int> id_to_index; // maps vertex ID to vector index

    std::string line;
    int vertex_index = 0;

    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "Vertex") {
            int id;
            double x, y, z;
            iss >> id >> x >> y >> z;

            int label = -1, instance = -1;
            size_t label_pos = line.find("label=");
            if (label_pos != std::string::npos) {
                label = std::stoi(line.substr(label_pos + 6));
            }
            size_t inst_pos = line.find("instance=");
            if (inst_pos != std::string::npos) {
                instance = std::stoi(line.substr(inst_pos + 9));
            }

            vertices.emplace_back(x, y, z);
            label_list.push_back(label);
            instance_list.push_back(instance);
            id_to_index[id] = vertex_index++;
        }
        else if (type == "Face") {
            int id, v1, v2, v3;
            iss >> id >> v1 >> v2 >> v3;

            if (id_to_index.count(v1) == 0 || id_to_index.count(v2) == 0 || id_to_index.count(v3) == 0) {
                std::cerr << "Face references unknown vertex ID in line: " << line << std::endl;
                return false;
            }

            faces.emplace_back(
                id_to_index[v1],
                id_to_index[v2],
                id_to_index[v3]
            );
        }
    }

    V.resize(vertices.size(), 3);
    for (size_t i = 0; i < vertices.size(); ++i)
        V.row(i) = vertices[i];

    F.resize(faces.size(), 3);
    for (size_t i = 0; i < faces.size(); ++i)
        F.row(i) = faces[i];

    labels = Eigen::Map<Eigen::VectorXi>(label_list.data(), label_list.size());
    instances = Eigen::Map<Eigen::VectorXi>(instance_list.data(), instance_list.size());

    return true;
}

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

void normalizeUV(Eigen::MatrixXd& UV) {
    for (int i = 0; i < UV.cols(); ++i) {
        double min_val = UV.col(i).minCoeff();
        double max_val = UV.col(i).maxCoeff();
        if (max_val != min_val) {
            UV.col(i) = (UV.col(i).array() - min_val) / (max_val - min_val);
        }
        else {
            UV.col(i).setZero();
        }
    }
}

void writeToNPZ(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& N,
    const Eigen::MatrixXd& UV,
    const Eigen::VectorXd& K1,
    const Eigen::VectorXd& K2,
    const Eigen::VectorXi& labels,
    const Eigen::VectorXi& instances) {

    string filename = output_dir + "/" + filesystem::path(mesh_fp).stem().stem().string() + ".npz";

    vector<double> v_vec      = toStdVector(V);
    vector<int>    f_vec      = toStdVector(F);
    vector<double> n_vec      = toStdVector(N);
    vector<double> uv_vec     = toStdVector(UV);
    vector<double> k1_vec     = toStdVector(K1);
    vector<double> k2_vec     = toStdVector(K2);
    vector<int> labels_vec    = toStdVector(labels);
    vector<int> instances_vec = toStdVector(instances);

    cnpy::npz_save(filename, "V", v_vec.data(), {static_cast<size_t>(V.rows()), static_cast<size_t>(V.cols())}, "w");
    cnpy::npz_save(filename, "F", f_vec.data(), {static_cast<size_t>(F.rows()), static_cast<size_t>(F.cols())}, "a");
    cnpy::npz_save(filename, "N", n_vec.data(), { static_cast<size_t>(N.rows()), static_cast<size_t>(N.cols()) }, "a");
    cnpy::npz_save(filename, "UV", uv_vec.data(), { static_cast<size_t>(UV.rows()), static_cast<size_t>(UV.cols()) }, "a");
    cnpy::npz_save(filename, "K1", k1_vec.data(), { static_cast<size_t>(k1_vec.size()) }, "a");
    cnpy::npz_save(filename, "K2", k2_vec.data(), { static_cast<size_t>(k2_vec.size()) }, "a");
    cnpy::npz_save(filename, "labels", labels_vec.data(), { static_cast<size_t>(labels.size()) }, "a");
    cnpy::npz_save(filename, "instances", instances_vec.data(), { static_cast<size_t>(instances.size()) }, "a");
}

int main(int argc, char* argv[]) {
    using namespace std;

    string help_string("Unwrap a mesh and output enriched data.");
    help_string += "\n\t[Input]  mesh(.m) file";
    help_string += "\n\t[Output] .npz file";
    cxxopts::Options options("UVUnwrap", help_string);
    options.add_options()
        ("m, mesh", "Input .m mesh file", cxxopts::value<string>())
        ("o, output-dir", "Output file directory", cxxopts::value<string>())
        ("d, method", "UV Unwrap method (arap/harmonic)", cxxopts::value<string>()->default_value("arap"))
        ("r, curv-ring", "Ring number used to calculate quadric fitting curvatures", cxxopts::value<unsigned int>()->default_value("5"))
        ("l, log-level", "Log level (error|warning|info|debug|trace)", cxxopts::value<string>()->default_value("debug"))
        ("h, help", "Print help information");


    // == Parse args
    auto result = options.parse(argc, argv);
    // help
    if (argc < 2 || result.count("help")
        || !result.count("mesh")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }
    // retrieve args
    mesh_fp = result["mesh"].as<string>();
    method = result["method"].as<string>();
    curv_ring = result["curv-ring"].as<unsigned int>();
    if (result.count("output-dir")) {
        output_dir = result["output-dir"].as<string>();
        filesystem::create_directories(output_dir);
    }
    else {
        output_dir = filesystem::path(mesh_fp).parent_path().string();
    }
    // log level
    const string lvl = result["log-level"].as<string>();
    if (lvl == "error")   current_log_level() = LogLevel::error;
    else if (lvl == "warning") current_log_level() = LogLevel::warning;
    else if (lvl == "info")    current_log_level() = LogLevel::info;
    else if (lvl == "debug")   current_log_level() = LogLevel::debug;
    else if (lvl == "trace")   current_log_level() = LogLevel::trace;


    // == Read .m file
    auto t00 = clock();
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::VectorXi labels, instances;
    if (!readMFile(mesh_fp, V, F, labels, instances)) return 1;
    LOG(LogLevel::debug, format("[{}]:{} faces, {} verts", filesystem::path(mesh_fp).stem().stem().string(), F.rows(), V.rows()));


    // == Compute normals
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);


    // == Compute curvatures
    // Compute curvature directions via quadric fitting
    auto t0 = clock();
    Eigen::MatrixXd PD1, PD2;
    Eigen::VectorXd K1, K2;
    igl::principal_curvature(V, F, PD1, PD2, K1, K2, curv_ring);
    auto t1 = clock();
    LOG(LogLevel::debug, format("1) curvature: {} ms", t1 - t0));


    // == Compute uv mapping
    // the initial solution for ARAP (harmonic parametrization)
    Eigen::VectorXi bnd;
    igl::boundary_loop(F, bnd);
    Eigen::MatrixXd bnd_uv;
    igl::map_vertices_to_circle(V, bnd, bnd_uv);

    Eigen::MatrixXd initial_guess;
    igl::harmonic(V, F, bnd, bnd_uv, 1, initial_guess);

    Eigen::MatrixXd UV;
    if (method == "arap") {
        
        igl::ARAPData arap_data;

        Eigen::Vector<int, 1> b;
        b(0) = bnd(int(bnd.rows() / 2));
        Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(1, 2);

        //arap_data.with_dynamics = true;
        //Eigen::VectorXi b = Eigen::VectorXi::Zero(0);
        //Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0, 0);
        
        // Initialize ARAP
        arap_data.max_iter = 100;
        // 2 means that we're going to *solve* in 2d
        arap_precomputation(V, F, 2, b, arap_data);

        // Solve arap using the harmonic map as initial guess
        UV = initial_guess;

        arap_solve(bc, arap_data, UV);
    }
    else {
        UV = initial_guess;
    }

    normalizeUV(UV);
    auto t2 = clock();
    LOG(LogLevel::debug, format("2) uv-map: {} ms", t2 - t1));

    // Output as JSON
    /*
    json j;
    j["V"] = to_json_matrix(V);
    j["F"] = to_json_matrix(F);
    j["UV"] = to_json_matrix(UV);
    j["N"] = to_json_matrix(N);
    j["labels"] = to_json_vector(labels);
    j["instances"] = to_json_vector(instances);

    std::ofstream out("mesh_data.json");
    out << j.dump();
    out.close();
    std::cout << "Mesh data exported to mesh_data.json" << std::endl;*/

    // Output as npz file
    writeToNPZ(V, F, N, UV, K1, K2, labels, instances);
    auto t3 = clock();
    LOG(LogLevel::debug, format("TOTAL: {} s", int((t3 - t00)/1000)));

    return 0;
}
