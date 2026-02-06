/* 
输入包含V、F、N、K、labels、instances等信息的.npz，以及包含YOLO seg预测instances的.txt；
输出每个instance对应一个.obj，一个.txt（记录每个顶点的GT label），一个.npy（每个顶点的attributes/features）；
*/

#include <format>
#include <fstream>
#include <filesystem>
#include <set>
#include <random>

#include <Eigen/Core>

#include <igl/writeOBJ.h>

#include <cnpy.h>

#include "log.h"
#include "cxxopts.hpp"

using namespace std;

// == Params
string input_fp;
string pred_result_fp;
string output_dir;
float clamp_lower = 0.005, clamp_upper = 0.995;
float remove_ratio = 0.8;
float expand_pred_scale = 1.1;


// == Miscs
void clampToPercentile(Eigen::VectorXd& vec, float low_perc, float up_perc) {
    vector<double> sorted_vec(vec.data(), vec.data() + vec.size());
    sort(sorted_vec.begin(), sorted_vec.end());

    size_t low_idx = static_cast<size_t>(low_perc * sorted_vec.size());
    size_t up_idx = static_cast<size_t>(up_perc * sorted_vec.size()) - 1;

    double low_bound = sorted_vec[low_idx];
    double up_bound = sorted_vec[up_idx];

    vec = vec.unaryExpr([low_bound, up_bound](double val) {
        return std::clamp(val, low_bound, up_bound);
        });
}

template<typename MatType>
MatType npyToEigen(const cnpy::NpyArray& arr);

template<>
Eigen::MatrixXd npyToEigen<Eigen::MatrixXd>(const cnpy::NpyArray& arr) {
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        reinterpret_cast<const double*>(arr.data<double>()), arr.shape[0], arr.shape[1]);
}

template<>
Eigen::MatrixXi npyToEigen<Eigen::MatrixXi>(const cnpy::NpyArray& arr) {
    return Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        reinterpret_cast<const int*>(arr.data<int>()), arr.shape[0], arr.shape[1]);
}

template<typename VecType>
VecType npyToEigenVec(const cnpy::NpyArray& arr);

template<>
Eigen::VectorXd npyToEigenVec<Eigen::VectorXd>(const cnpy::NpyArray& arr) {
    return Eigen::Map<const Eigen::VectorXd>(
        reinterpret_cast<const double*>(arr.data<double>()), arr.shape[0]);
}

template<>
Eigen::VectorXi npyToEigenVec<Eigen::VectorXi>(const cnpy::NpyArray& arr) {
    return Eigen::Map<const Eigen::VectorXi>(
        reinterpret_cast<const int*>(arr.data<int>()), arr.shape[0]);
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

void normalizeAttribute(Eigen::VectorXd& vec) {
    vec.array() -= vec.minCoeff();
    vec.array() /= (vec.maxCoeff() - vec.minCoeff() + 1e-6);
}

void expandBBox(
    Eigen::Vector4d& bbox,
    const float expand_scale
) {
    double width = bbox(2) - bbox(0);
    double height = bbox(3) - bbox(1);
    Eigen::Vector2d center((bbox(0) + bbox(2)) * 0.5, (bbox(1) + bbox(3)) * 0.5);

    width *= expand_scale;
    height *= expand_scale;

    bbox(0) = center.x() - width / 2.0;
    bbox(1) = center.y() - height / 2.0;
    bbox(2) = center.x() + width / 2.0;
    bbox(3) = center.y() + height / 2.0;

    return;
}

bool insideBbox(const Eigen::Vector4d& bbox, const Eigen::Vector2d& xy) {
    bool result = (
        xy.x() >= bbox.x() &&
        xy.y() >= bbox.y() &&
        xy.x() <= bbox.z() &&
        xy.y() <= bbox.w());
    return result;
}

// Ray casting
bool pointInPolygon(const Eigen::Vector2d& pt, const vector<Eigen::Vector2d>& polygon) {
    int n = polygon.size();
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const auto& pi = polygon[i];
        const auto& pj = polygon[j];
        if (((pi.y() > pt.y()) != (pj.y() > pt.y())) &&
            (pt.x() < (pj.x() - pi.x()) * (pt.y() - pi.y()) / (pj.y() - pi.y() + 1e-12) + pi.x())) {
            inside = !inside;
        }
    }
    return inside;
}


// == Functions
void readNpzFile(
    const string& fp,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::MatrixXd& N,
    Eigen::MatrixXd& UV,
    Eigen::VectorXd& K1,
    Eigen::VectorXd& K2,
    Eigen::VectorXi& labels,
    Eigen::VectorXi& instances) {
    LOG(LogLevel::debug, format("Loading {}...", fp));

    cnpy::npz_t npz = cnpy::npz_load(fp);
    V = npyToEigen<Eigen::MatrixXd>(npz["V"]);
    F = npyToEigen<Eigen::MatrixXi>(npz["F"]);
    N = npyToEigen<Eigen::MatrixXd>(npz["N"]);
    UV = npyToEigen<Eigen::MatrixXd>(npz["UV"]);
    K1 = npyToEigenVec<Eigen::VectorXd>(npz["K1"]);
    K2 = npyToEigenVec<Eigen::VectorXd>(npz["K2"]);
    labels = npyToEigenVec<Eigen::VectorXi>(npz["labels"]);
    instances = npyToEigenVec<Eigen::VectorXi>(npz["instances"]);
}

void readPredInstances(
    const string& fp,
    map<int, int>& instance_label_map,
    map<int, Eigen::Vector4d>& uv_bboxes,
    map<int, vector<Eigen::Vector2d>>& uv_masks) {
    ifstream infile(fp);
    if (!infile.is_open()) {
        LOG(LogLevel::error, "Failed to open file: " + fp);
        return;
    }

    string line;
    int instance_id = 1;  // instance(other than BG) starts from 1

    // For each line: class_id x1 y1 x2 y2 ... xk yk confidence
    while (getline(infile, line)) {
        istringstream iss(line);
        vector<double> values;
        double val;
        while (iss >> val) {
            values.push_back(val);
        }

        if (values.size() < 3) continue;  // at least: class_id x1 y1

        // Retrieve class label
        instance_label_map[instance_id] = int(values[0]);
        size_t start_idx = 1, end_idx = values.size();

        // Ignore confidence
        bool has_conf = (values.size() % 2 == 0);
        if (has_conf) --end_idx;

        // Bbox & mask
        double min_x = 1e9, min_y = 1e9, max_x = -1e9, max_y = -1e9;
        vector<Eigen::Vector2d> mask;
        for (size_t i = start_idx; i < end_idx; i += 2) {
            double x = values[i];
            double y = 1 - values[i + 1];  // y axis in YOLO is from-up-to-down!
            mask.emplace_back(x, y);
            min_x = min(min_x, x);
            min_y = min(min_y, y);
            max_x = max(max_x, x);
            max_y = max(max_y, y);
        }

        uv_bboxes[instance_id] = Eigen::Vector4d(min_x, min_y, max_x, max_y);
        uv_masks[instance_id] = mask;
        instance_id++;
    }

    infile.close();
}


int main(int argc, char* argv[]) {
    cxxopts::Options options("Single Tooth Extractor", "Read a npz file and generate one mesh for each tooth.");
    options.add_options()
        ("i,input", "Input .npz file", cxxopts::value<string>())
        ("p,pred-result", "The file path containing predicted instances in YOLO segment format", cxxopts::value<string>())
        ("o,output-dir", "Output file directory", cxxopts::value<string>())
        ("lower", "Lower percentile to clamp", cxxopts::value<float>()->default_value("0.005"))
        ("upper", "Upper percentile to clamp", cxxopts::value<float>()->default_value("0.995"))
        ("expand-scale", "Fixed the expanding scale of per instance uv bbox", cxxopts::value<float>()->default_value("1.1"))
        ("r,remove-ratio", "Randomly remove some instances to generate", cxxopts::value<float>()->default_value("0.8"))
        ("l,log-level", "Log level (error|warning|info|debug|trace)", cxxopts::value<string>()->default_value("debug"))
        ("help", "Print help");

    // == Parse args
    auto result = options.parse(argc, argv);
    if (result.count("help")
        || argc < 3
        || !(result.count("input"))
        || !(result.count("pred-result"))) {
        cout << options.help() << endl;
        return 0;
    }
    // retrieve args
    input_fp = result["input"].as<string>();
    pred_result_fp = result["pred-result"].as<string>();
    clamp_lower = result["lower"].as<float>();
    clamp_upper = result["upper"].as<float>();
    expand_pred_scale = result["expand-scale"].as<float>();
    remove_ratio = result["remove-ratio"].as<float>();
    if (result.count("output-dir")) {
        output_dir = result["output-dir"].as<string>();
        filesystem::create_directories(output_dir);
    }
    else {
        output_dir = filesystem::path(input_fp).parent_path().string();
    }
    // log level
    string lvl = result["log-level"].as<string>();
    if (lvl == "error") current_log_level() = LogLevel::error;
    else if (lvl == "warning") current_log_level() = LogLevel::warning;
    else if (lvl == "info") current_log_level() = LogLevel::info;
    else if (lvl == "debug") current_log_level() = LogLevel::debug;
    else if (lvl == "trace") current_log_level() = LogLevel::trace;


    // == Read .npz file
    Eigen::MatrixXd V, N, UV;
    Eigen::MatrixXi F;
    Eigen::VectorXd K1, K2;
    Eigen::VectorXi labels, instances;
    readNpzFile(input_fp, V, F, N, UV, K1, K2, labels, instances);
    LOG(LogLevel::info, format("Loaded {} with {} vertices and {} faces",
        filesystem::path(input_fp).stem().string(), V.rows(), F.rows()));


    normalizeUV(UV);  // IMPORTANT!


    // 6D features: Normals + K1,K2,K3
    Eigen::MatrixXd attr;
    normalizeUV(N);
    clampToPercentile(K1, clamp_lower, clamp_upper);
    clampToPercentile(K2, clamp_lower, clamp_upper);
    normalizeAttribute(K1);
    normalizeAttribute(K2);
    Eigen::VectorXd K3 = sqrt((K1.array() * K1.array() + K2.array() * K2.array()) / 2);
    attr = Eigen::MatrixXd(K1.rows(), 6);
    attr.leftCols(3) = N;
    attr.col(3) = K1;
    attr.col(4) = K2;
    attr.col(5) = K3;


    // === Read .txt(pred instances) and find each instance's uv bbox
    map<int, Eigen::Vector4d> uv_bboxes;
    map<int, vector<Eigen::Vector2d>> uv_masks;
    map<int, int> instance_label_map;
    set<int> unique_instances;
    readPredInstances(pred_result_fp, instance_label_map, uv_bboxes, uv_masks);
    for (const auto& [key, _] : uv_bboxes) {
        unique_instances.insert(key);
    }

    // Random remove some instances with labels 0~6 (due to large amounts)
    if (remove_ratio > 0) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> prob_dist(0., 1.);
        for (auto it = unique_instances.begin(); it != unique_instances.end(); ) {
            int current_label = instance_label_map[*it];

            if (prob_dist(gen) < remove_ratio && current_label != 7 && current_label != 15) {
                it = unique_instances.erase(it);
            }
            else {
                it++;
            }
        }
    }

    // For each instance
    for (int instance : unique_instances) {
        int current_label = instance_label_map[instance];

        // Expand bbox by expand_pred_scale
        expandBBox(uv_bboxes[instance], expand_pred_scale);

        // Get all faces with labels inside bbox
        Eigen::Vector4d uv_bbox = uv_bboxes[instance];
        vector<Eigen::Vector2d> uv_mask = uv_masks[instance];
        Eigen::VectorXi labels_new(labels);

        std::vector<int> valid_faces;
        std::set<int> valid_verts_set;
        for (int fi = 0; fi < F.rows(); fi++) {
            Eigen::Vector3i vids = F.row(fi);
            if (insideBbox(uv_bbox, UV.row(vids[0])) ||
                insideBbox(uv_bbox, UV.row(vids[1])) ||
                insideBbox(uv_bbox, UV.row(vids[2]))) {

                valid_faces.push_back(fi);
                for (size_t i = 0; i < 3; i++) {
                    int vi = vids[i];
                    valid_verts_set.insert(vi);
                    if (pointInPolygon(UV.row(vi), uv_mask))
                        labels_new[vi] = current_label+1;  // 0~7 -> 1~8
                    else
                        labels_new[vi] = 0;
                }
            }
        }

        // 将set映射为vector并建立全局索引到局部索引的映射表
        std::vector<int> valid_verts(valid_verts_set.begin(), valid_verts_set.end());
        std::unordered_map<int, int> global_to_local;
        for (size_t i = 0; i < valid_verts.size(); ++i) {
            global_to_local[valid_verts[i]] = static_cast<int>(i);
        }


        // == Construct sub-mesh
        // 构建新的 V, F, attr, label
        Eigen::MatrixXd V_sub(valid_verts.size(), 3);
        Eigen::MatrixXd attr_sub(valid_verts.size(), attr.cols());
        Eigen::VectorXi pred_mask_sub(valid_verts.size());
        Eigen::VectorXi gt_mask_sub(valid_verts.size());

        for (size_t i = 0; i < valid_verts.size(); ++i) {
            int vi = valid_verts[i];
            V_sub.row(i) = V.row(vi);
            attr_sub.row(i) = attr.row(vi);
            pred_mask_sub(i) = labels_new(vi);  // 使用labels_new：即根据uv_mask筛选后的label
            gt_mask_sub(i) = labels(vi);        // 注意gt_mask是11~48并且是所有牙齿，pred_mask是1~8并且只是单一牙齿
        }

        // 新的face索引
        Eigen::MatrixXi F_sub(valid_faces.size(), 3);
        for (size_t i = 0; i < valid_faces.size(); ++i) {
            Eigen::Vector3i old_face = F.row(valid_faces[i]);
            F_sub.row(i) = Eigen::Vector3i(
                global_to_local[old_face[0]],
                global_to_local[old_face[1]],
                global_to_local[old_face[2]]
            );
        }

        // 保存路径构造
        string filename = filesystem::path(input_fp).stem().string() + "_" + to_string(instance);
        auto output_root = filesystem::path(output_dir);
        filesystem::path mesh_dir("obj"), attr_dir("npy"), pred_mask_dir("pred_mask"), gt_mask_dir("gt_mask");
        filesystem::create_directories(output_root / mesh_dir);
        filesystem::create_directories(output_root / attr_dir);
        filesystem::create_directories(output_root / pred_mask_dir);
        filesystem::create_directories(output_root / gt_mask_dir);
        string mesh_path = (output_root / mesh_dir / (filename + ".obj")).string();
        string attr_path = (output_root / attr_dir / (filename + ".npy")).string();
        string pred_mask_path = (output_root / pred_mask_dir / (filename + ".txt")).string();
        string gt_mask_path = (output_root / gt_mask_dir / (filename + ".txt")).string();

        // 保存 mesh（使用 igl::writeOBJ）
        igl::writeOBJ(mesh_path, V_sub, F_sub);

        // 保存 attr（注意Eigen是column major存储，需要先转置再存到npy中，以便后续numpy读取）
        Eigen::MatrixXd mat_transposed = attr_sub.transpose();
        cnpy::npy_save(attr_path, mat_transposed.data(), { (size_t)attr_sub.rows(), (size_t)attr_sub.cols() }, "w");

        // 保存 labels（保存为 txt 或 npy）
        std::ofstream pred_mask_out(pred_mask_path);
        std::ofstream gt_mask_out(gt_mask_path);
        for (int i = 0; i < valid_verts.size(); ++i) {
            pred_mask_out << pred_mask_sub(i) << " ";
            gt_mask_out << gt_mask_sub(i) << " ";
        }
        
        pred_mask_out.close();
    }
}