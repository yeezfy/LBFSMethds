#ifndef PROJECT_FVMDISCRETER_GFVMDISCRETER_H
#define PROJECT_FVMDISCRETER_GFVMDISCRETER_H

#include <iostream>
#include "../../../lib/eigen-3.4/Dense"

namespace GFVMDiscreter {
  inline void diffusionTermDirichlet(Eigen::VectorXd cell_center,
                                     Eigen::VectorXd face_center,
                                     Eigen::VectorXd face_vector,
                                     Eigen::VectorXd gradient,
                                     const Eigen::Matrix3d& diffusion,
                                     double& coe_orthogonal,
                                     double& coe_non_orthogonal) {
    Eigen::Vector3d link_vec{face_center - cell_center};
    Eigen::Vector3d link_vec_unit{link_vec.normalized()};
    Eigen::Vector3d orthogonal_vec{face_vector.dot(face_vector) / link_vec_unit.dot(face_vector) * link_vec_unit};
    Eigen::Vector3d non_orthogonal_vec{face_vector - orthogonal_vec};

    coe_orthogonal = (-1 * diffusion * ((1 / link_vec.norm()) * link_vec_unit)).dot(orthogonal_vec);
    coe_non_orthogonal = (-1 * diffusion * gradient).dot(non_orthogonal_vec);
  }

  inline void sourceTerm(double source_value, double& coe_source) {
    coe_source = source_value;
  }

  inline void transientTerm(double transient_coe, double cell_volume, double delta_t, double& coe_transient) {
    coe_transient = transient_coe * cell_volume / delta_t;
  }
}

#endif // PROJECT_FVMDISCRETER_GFVMDISCRETER_H


