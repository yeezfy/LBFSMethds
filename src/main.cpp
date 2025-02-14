#include "ReadData/gReadData.h"
#include "Model/gModelLBFS.h"
#include "../lib/eigen-3.4/Dense"
#include "../lib/eigen-3.4/Sparse"

int main(int argc, char* argv[]) {
  Eigen:: MatrixXd cell_center = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/cell_center.csv");
  int cell_num{(int)cell_center.rows()};
  Eigen::VectorXd fai_;
  fai_.resize(cell_num);
  fai_ = GReadData::ExtractDataFromCSVVector("F:/LBFSMethods/Project/src/Data/initial_info.csv");
  Eigen::SparseMatrix<double> A;
  A.resize(cell_num, cell_num);
  A.setZero();
  Eigen::VectorXd b;
  b.resize(cell_num);
  b.setZero();

  GModelLBFS model;
  model.setScalarPtr(&fai_);

  double time_step{1};
  double time_delta{1};
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> sparse_solver;
  for (int i=0; i!=time_step; ++i) {
    // In this code, we just give an example for calculating the first time step.
    // If multi-time step is needed, the gradient should be upgraded by other package.
    b.setZero();
    model.assembleMatrix(A, b, time_delta, i);
    sparse_solver.compute(A);
    fai_ = sparse_solver.solve(b);
  }
}
