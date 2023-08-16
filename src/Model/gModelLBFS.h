#ifndef PROJECT_Model_GModelLBFS_H
#define PROJECT_Model_GModelLBFS_H

#include "../../../../lib/eigen-3.4/Sparse"
#include "../FVMDiscreter/gFVMDiscreter.h"

class GModelLBFS {
 public:
  GModelLBFS();
  ~GModelLBFS();

  double cs2{1/3.0};
  Eigen::Vector3d omega {2.0/3.0, 1.0/6.0, 1.0/6.0};
  double calcFeqCoe(int n, const Eigen::Vector3d& e, const Eigen::Vector3d& velocity) {
    return omega(n)*(1 + e.dot(velocity)/cs2);
  }
  const Eigen::VectorXd* const getScalarPtr() const {
    return scalar_ptr_;
  }
  void setScalarPtr(Eigen::VectorXd* const scalar_ptr) {
    scalar_ptr_ = scalar_ptr;
  }

  void assembleMatrix(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &b, double delta_t, int step=0);

 private:
  Eigen::VectorXd* scalar_ptr_{nullptr};
};

#endif //PROJECT_MODEL_GModelLBFS_H
