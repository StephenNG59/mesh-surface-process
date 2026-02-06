#include <iostream>
#include <filesystem>
#include <cnpy.h>
#include <Eigen/Core>

using namespace std;
using namespace std::filesystem;


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



int main(int argc, char* argv[]) {
    Eigen::MatrixXd V, N, UV;
    Eigen::MatrixXi F;
    Eigen::VectorXd K1, K2;
    Eigen::VectorXi labels, instances;
    readNpzFile(string(argv[1]), V, F, N, UV, K1, K2, labels, instances);


}
