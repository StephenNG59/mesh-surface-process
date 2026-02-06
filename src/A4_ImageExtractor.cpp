#include <format>
#include <array>
#include <tuple>
#include <set>
#include <random>
#include <filesystem>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cnpy.h>

#include "log.h"
#include "cxxopts.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

// == Params
string input_fp;
string pred_result_fp = "";
string output_dir;
string attribute;
size_t img_width, img_height;
float clamp_lower = 0.005, clamp_upper = 0.995;
bool per_instance = false;
float remove_ratio = 0.8;
bool random_jitter = false;
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

// Check if point is in triangle using barycentric coordinates
bool insideTriangle(const Eigen::Vector2d& p, const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c, Eigen::Vector3d& bary) {
    Eigen::Vector2d v0 = b - a;
    Eigen::Vector2d v1 = c - a;
    Eigen::Vector2d v2 = p - a;
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom = d00 * d11 - d01 * d01;
    if (denom == 0) return false;

    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;
    bary = Eigen::Vector3d(u, v, w);
    return (u >= 0 && v >= 0 && w >= 0);
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

void extendBbox(Eigen::Vector4d& bbox, const Eigen::Vector2d& xy) {
    // if < 0, initialize it
    if (bbox.x() < 0) {
        bbox.x() = xy.x();
        bbox.y() = xy.y();
        bbox.z() = xy.x();
        bbox.w() = xy.y();
    }
    else {
        bbox.x() = min(bbox.x(), xy.x());
        bbox.y() = min(bbox.y(), xy.y());
        bbox.z() = max(bbox.z(), xy.x());
        bbox.w() = max(bbox.w(), xy.y());
    }
}

bool insideBbox(const Eigen::Vector4d& bbox, const Eigen::Vector2d& xy) {
    bool result = (
        xy.x() >= bbox.x() &&
        xy.y() >= bbox.y() &&
        xy.x() <= bbox.z() &&
        xy.y() <= bbox.w());
    return result;
}

Eigen::Vector4d randomExpandOrShrinkBBox(
    const Eigen::Vector4d& bbox,
    double max_shift_ratio = 0.02,
    double scale_expand_prob = 0.9,
    double scale_min = 0.95,
    double scale_max = 1.1)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    double width = bbox(2) - bbox(0);
    double height = bbox(3) - bbox(1);
    Eigen::Vector2d center((bbox(0) + bbox(2)) * 0.5, (bbox(1) + bbox(3)) * 0.5);

    // 随机平移bbox中心，范围 [-max_shift_ratio * width, max_shift_ratio * width]
    std::uniform_real_distribution<> shift_dist_x(-max_shift_ratio * width, max_shift_ratio * width);
    std::uniform_real_distribution<> shift_dist_y(-max_shift_ratio * height, max_shift_ratio * height);

    double shift_x = shift_dist_x(gen);
    double shift_y = shift_dist_y(gen);
    Eigen::Vector2d new_center = center + Eigen::Vector2d(shift_x, shift_y);

    // 生成缩放比例，x和y方向各自独立
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);
    bool expand_x = prob_dist(gen) < scale_expand_prob;
    bool expand_y = prob_dist(gen) < scale_expand_prob;

    std::uniform_real_distribution<> scale_expand_dist(1.0, scale_max);
    std::uniform_real_distribution<> scale_shrink_dist(scale_min, 1.0);

    double scale_x = expand_x ? scale_expand_dist(gen) : scale_shrink_dist(gen);
    double scale_y = expand_y ? scale_expand_dist(gen) : scale_shrink_dist(gen);

    double new_width = width * scale_x;
    double new_height = height * scale_y;

    Eigen::Vector4d new_bbox;
    new_bbox(0) = new_center.x() - new_width / 2.0;
    new_bbox(1) = new_center.y() - new_height / 2.0;
    new_bbox(2) = new_center.x() + new_width / 2.0;
    new_bbox(3) = new_center.y() + new_height / 2.0;

    return new_bbox;
}

void expandBBoxAndMask(
    Eigen::Vector4d& bbox,
    vector<Eigen::Vector2d>& mask,
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

    //for (auto& xy : mask) {
    //    Eigen::Vector2d from_center = xy - center;
    //    xy = center + from_center * expand_scale;
    //}

    return;
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

        // Ignore class label
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

void rasterizeWholeMesh(
    const Eigen::MatrixXd& UV,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& attr,
    const Eigen::VectorXi& labels,
    size_t width, size_t height,
    vector<uint8_t>& rgb_out,
    vector<uint8_t>& mask_out) {

    const int channels = attr.cols();
    rgb_out.resize(width * height * channels, 0);
    mask_out.resize(width * height, 0);  // (w*h, 255)

    for (int fi = 0; fi < F.rows(); ++fi) {
        Eigen::Vector2d uv0 = UV.row(F(fi, 0));
        Eigen::Vector2d uv1 = UV.row(F(fi, 1));
        Eigen::Vector2d uv2 = UV.row(F(fi, 2));

        uv0 = uv0.cwiseProduct(Eigen::Vector2d(width, height));
        uv1 = uv1.cwiseProduct(Eigen::Vector2d(width, height));
        uv2 = uv2.cwiseProduct(Eigen::Vector2d(width, height));

        Eigen::AlignedBox2d bbox;
        bbox.extend(uv0);
        bbox.extend(uv1);
        bbox.extend(uv2);

        int x0 = max(0, int(floor(bbox.min().x())));
        int x1 = min(int(width) - 1, int(ceil(bbox.max().x())));
        int y0 = max(0, int(floor(bbox.min().y())));
        int y1 = min(int(height) - 1, int(ceil(bbox.max().y())));

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                Eigen::Vector2d p(x + 0.5, y + 0.5);
                Eigen::Vector3d bary;
                if (insideTriangle(p, uv0, uv1, uv2, bary)) {
                    Eigen::Vector3i vids = F.row(fi);
                    Eigen::RowVectorXd color =
                        bary(0) * attr.row(vids[0]) +
                        bary(1) * attr.row(vids[1]) +
                        bary(2) * attr.row(vids[2]);

                    int idx = (height - 1 - y) * width + x; // flip y

                    // Max barycentric coord vertex -> decides the label of pixel
                    size_t max_vid;
                    bary.maxCoeff(&max_vid);
                    if (mask_out[idx] == 0) {
                        // 只添加，不去除label
                        mask_out[idx] = static_cast<uint8_t>(labels[vids[max_vid]]);

                        if (channels == 3) {
                            // RGB image
                            for (int c = 0; c < channels; ++c) {
                                rgb_out[idx * channels + c] = static_cast<uint8_t>(std::clamp(color[c] * 255.0, 0.0, 255.0));
                            }
                        }
                        else {
                            for (int c = 0; c < channels; ++c)
                                rgb_out[idx * channels + c] = color[c];
                        }
                    }

                    
                }
            }
        }
    }
}

map<int, Eigen::Vector4d> rasterizePerInstance(
    const Eigen::MatrixXd& UV,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& attr,
    const Eigen::VectorXi& labels,
    const Eigen::VectorXi& instances,
    size_t width, size_t height,
    map<int, vector<uint8_t>>& rgbs_out,
    map<int, vector<uint8_t>>& masks_out,
    float remove_ratio,
    bool random_jitter
) {

    // All unique instance number ('0' excluded)
    set<int> unique_instances(instances.data(), instances.data() + instances.size());
    unique_instances.erase(0);

    // Random remove some instances with labels 1~7 (due to large amounts)
    if (remove_ratio > 0) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> prob_dist(0., 1.);
        for (auto it = unique_instances.begin(); it != unique_instances.end(); ) {
            int vi = 0;
            while (instances[vi] != *it) { vi++; }
            int current_label = labels[vi];

            if (prob_dist(gen) < remove_ratio && current_label % 10 != 8) {
                it = unique_instances.erase(it);
            }
            else {
                it++;
            }
        }
    }
    

    // === Find each instance's uv bbox
    map<int, Eigen::Vector4d> uv_bboxes;
    map<int, vector<Eigen::Vector2d>> uv_masks;

    // a. If ground-truth instances
    if (pred_result_fp == "") {
        for (int instance : unique_instances) {
            uv_bboxes[instance] = Eigen::Vector4d::Constant(-1);
            uv_masks[instance] = vector<Eigen::Vector2d>();
        }
        for (int i = 0; i < UV.rows(); i++) {
            extendBbox(uv_bboxes[instances[i]], UV.row(i));
        }
    }
    // b. If predicted instances
    else {
        readPredInstances(pred_result_fp, uv_bboxes, uv_masks);
        // update unique_instances;
        //  otherwise, only the instance_ids in GT will be processed
        unique_instances.clear();
        for (const auto& [key, _] : uv_bboxes) {
            unique_instances.insert(key);
        }
    }

    // For each instance
    for (int instance : unique_instances) {
        auto t0 = clock();

        // Resize bounding box (and masks) here
        if (pred_result_fp != "") {
            expandBBoxAndMask(uv_bboxes[instance], uv_masks[instance], expand_pred_scale);
        }
        else if (random_jitter) {
            uv_bboxes[instance] = randomExpandOrShrinkBBox(uv_bboxes[instance]);
        }

        // Get all faces (with labels) inside bbox
        Eigen::Vector4d uv_bbox = uv_bboxes[instance];
        vector<Eigen::Vector2d> uv_mask = uv_masks[instance];
        vector<int> valid_faces;
        Eigen::VectorXi labels_new(labels);
        for (int fi = 0; fi < F.rows(); fi++) {
            Eigen::Vector3i vids = F.row(fi);
            if (insideBbox(uv_bbox, UV.row(vids[0])) ||
                insideBbox(uv_bbox, UV.row(vids[1])) ||
                insideBbox(uv_bbox, UV.row(vids[2]))) {

                valid_faces.push_back(fi);
                for (size_t i = 0; i < 3; i++) {
                    int vi = vids[i];
                    if (pred_result_fp != "") {
                        if (pointInPolygon(UV.row(vi), uv_mask)) labels_new[vi] = 255;
                        else labels_new[vi] = 0;
                    }
                    else {
                        if (instances[vi] == instance) labels_new[vi] = 255;
                        else labels_new[vi] = 0;
                    }
                }
            }
        }

        // == Construct sub-mesh
        // 1. faces
        Eigen::MatrixXi F_sub(valid_faces.size(), 3);
        for (size_t i = 0; i < valid_faces.size(); i++)
            F_sub.row(i) = F.row(valid_faces[i]);
        // 2. uvs
        Eigen::MatrixXd UV_new(UV);
        UV_new.col(0).array() -= uv_bbox.x();
        UV_new.col(0).array() /= (uv_bbox.z() - uv_bbox.x());
        UV_new.col(1).array() -= uv_bbox.y();
        UV_new.col(1).array() /= (uv_bbox.w() - uv_bbox.y());

        // 
        rasterizeWholeMesh(UV_new, F_sub, attr, labels_new, width, height, rgbs_out[instance], masks_out[instance]);

        LOG(LogLevel::debug, format("{}ms for instance {}.", clock() - t0, instance));
    }

    return uv_bboxes;
}


int main(int argc, char* argv[]) {
    auto t0 = clock();

    cxxopts::Options options("Image Extractor", "Read a npz file and generate image.");
    options.add_options()
        ("i,input", "Input .npz file", cxxopts::value<string>())
        ("p,pred-result", "The file path containing predicted instances in YOLO segment format", cxxopts::value<string>())
        ("o,output-dir", "Output file directory", cxxopts::value<string>())
        ("a,attributes", "Attributes used to generate image (N|K1|K2|K3|NK|NKM)", cxxopts::value<string>()->default_value("N"))
        ("w,width", "Image width", cxxopts::value<size_t>()->default_value("1280"))
        ("h,height", "Image height", cxxopts::value<size_t>()->default_value("1280"))
        ("lower", "Lower percentile to clamp", cxxopts::value<float>()->default_value("0.005"))
        ("upper", "Upper percentile to clamp", cxxopts::value<float>()->default_value("0.995"))
        ("instance", "Enable per-instance image generation")
        ("expand-scale", "Fixed the expanding scale of per instance uv bbox", cxxopts::value<float>()->default_value("1.1"))
        ("r,remove-ratio", "Randomly remove some instances to generate", cxxopts::value<float>()->default_value("0.8"))
        ("j,random-jitter", "Randomly jitter uvbox for per-instance generation", cxxopts::value<bool>()->default_value("false"))
        ("l,log-level", "Log level (error|warning|info|debug|trace)", cxxopts::value<string>()->default_value("debug"))
        ("help", "Print help");

    
    // == Parse args
    auto result = options.parse(argc, argv);
    if (result.count("help") || argc < 2) {
        cout << options.help() << endl;
        return 0;
    }
    // retrieve args
    input_fp = result["input"].as<string>();
    attribute = result["attributes"].as<string>();
    img_width = result["width"].as<size_t>();
    img_height = result["height"].as<size_t>();
    clamp_lower = result["lower"].as<float>();
    clamp_upper = result["upper"].as<float>();
    per_instance = result["instance"].as<bool>();
    expand_pred_scale = result["expand-scale"].as<float>();
    remove_ratio = result["remove-ratio"].as<float>();
    random_jitter = result["random-jitter"].as<bool>();
    if (result.count("pred-result")) {
        pred_result_fp = result["pred-result"].as<string>();
    }
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


    Eigen::MatrixXd attr;
    if (attribute == "N") {
        attr = N;
        normalizeUV(attr);
    }
    else if (attribute == "K1") {
        clampToPercentile(K1, clamp_lower, clamp_upper);
        normalizeAttribute(K1);
        attr = K1.replicate(1, 3);
    }
    else if (attribute == "K2") {
        clampToPercentile(K2, clamp_lower, clamp_upper);
        normalizeAttribute(K2);
        attr = K2.replicate(1, 3);
    }
    else if (attribute == "K3") {
        clampToPercentile(K1, clamp_lower, clamp_upper);
        clampToPercentile(K2, clamp_lower, clamp_upper);
        normalizeAttribute(K1);
        normalizeAttribute(K2);
        Eigen::VectorXd K3 = sqrt((K1.array() * K1.array() + K2.array() * K2.array()) / 2);
        attr = Eigen::MatrixXd(K1.rows(), 3);
        attr.col(0) = K1;
        attr.col(1) = K2;
        attr.col(2) = K3;
        //attr = K3.replicate(1, 3);
    } 
    else if (attribute == "NK" || attribute == "NKM") {
        normalizeUV(N);
        clampToPercentile(K1, clamp_lower, clamp_upper);
        clampToPercentile(K2, clamp_lower, clamp_upper);
        normalizeAttribute(K1);
        normalizeAttribute(K2);
        Eigen::VectorXd K3 = sqrt((K1.array() * K1.array() + K2.array() * K2.array()) / 2);
        //attr = Eigen::MatrixXd(K1.rows(), (attribute == "NK") ? 6 : 7);
        attr = Eigen::MatrixXd(K1.rows(), 6);
        attr.leftCols(3) = N;
        attr.col(3) = K1;
        attr.col(4) = K2;
        attr.col(5) = K3;
    }
    else {
        LOG(LogLevel::error, "Unsupported attribute. Use N, K1, K2, K3, NK or NKM.");
        return 1;
    }


    if (per_instance) {
        map<int, vector<uint8_t>> mask_imgs;
        map<int, vector<uint8_t>> rgb_imgs;
        map<int, Eigen::Vector4d> uv_bboxes = rasterizePerInstance(UV, F, attr, labels, instances, img_width, img_height, rgb_imgs, mask_imgs, remove_ratio, random_jitter);

        for (const auto& [instance, rgb_img] : rgb_imgs) {
            string filename = filesystem::path(input_fp).stem().string() + "_" + to_string(instance);
            string mask_path = output_dir + "/" + filename + ".mask.png";
            string rgb_path = output_dir + "/" + filename + ".png";
            string npy_path = output_dir + "/" + filename + ".npy";
            string txt_path = output_dir + "/" + filename + ".uv.txt";
            vector<uint8_t> mask_img = mask_imgs[instance];

            // Write mask (.png)
            if (attribute != "NKM")
                stbi_write_png(mask_path.c_str(), img_width, img_height, 1, mask_img.data(), img_width); 

            // Write image (.png / .npy)
            size_t channels = attr.cols();
            if (channels == 3) {
                vector<uint8_t> rgb_img_uint(rgb_img.size());
                for (size_t i = 0; i < rgb_img.size(); ++i) {
                    rgb_img_uint[i] = static_cast<uint8_t>(rgb_img[i]);
                }
                stbi_write_png(rgb_path.c_str(), img_width, img_height, 3, rgb_img_uint.data(), img_width * 3);
            }
            else {
                // insert mask into rgb_img
                vector<float> rgb_img_float;
                if (attribute == "NKM") {
                    channels += 1;
                    rgb_img_float.resize(rgb_img.size() + mask_img.size());
                    for (size_t index = 0; index < rgb_img_float.size() / channels; index++) {
                        for (size_t c = 0; c < channels - 1; c++) {
                            rgb_img_float[index * channels + c] = static_cast<float>(rgb_img[index * (channels - 1) + c]);
                        }
                        rgb_img_float[index * channels + (channels - 1)] = static_cast<float>(mask_img[index] / 255.0);
                    }
                }
                else {
                    rgb_img_float.resize(rgb_img.size());
                    for (size_t i = 0; i < rgb_img.size(); ++i) {
                        rgb_img_float[i] = static_cast<float>(rgb_img[i]);
                    }
                }
                vector<size_t> shape = { img_height, img_width, channels };
                cnpy::npy_save(npy_path, &rgb_img_float[0], shape, "w");
            }

            // Write uv bbox data (.txt)
            ofstream file(txt_path);
            file << uv_bboxes[instance].transpose() << endl;
            file.close();
        }
        LOG(LogLevel::info, format("[{} ms] All images saved.", clock() - t0));
        return 0;
    }
    else {
        vector<uint8_t> mask_img;
        vector<uint8_t> rgb_img;
        rasterizeWholeMesh(UV, F, attr, labels, img_width, img_height, rgb_img, mask_img);

        string rgb_path = output_dir + "/" + filesystem::path(input_fp).stem().string() + ".png";
        string mask_path = output_dir + "/" + filesystem::path(input_fp).stem().string() + ".mask.png";

        stbi_write_png(rgb_path.c_str(), img_width, img_height, 3, rgb_img.data(), img_width * 3);
        stbi_write_png(mask_path.c_str(), img_width, img_height, 1, mask_img.data(), img_width);

        LOG(LogLevel::info, format("[{} ms] Images saved as {} and {}", clock() - t0, rgb_path, mask_path));
        return 0;
    }
}
