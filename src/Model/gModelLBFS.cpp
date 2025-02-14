#include "gModelLBFS.h"
#include "../ReadData/gReadData.h"

GModelLBFS::GModelLBFS() {}
GModelLBFS::~GModelLBFS() {}

void GModelLBFS::assembleMatrix(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b, double delta_t, int step) {
  Eigen:: MatrixXd cell_center = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/cell_center.csv");
  Eigen:: MatrixXd cell_info = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/cell_info.csv");
  Eigen:: MatrixXd face_cell = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/face_cell.csv");
  Eigen:: MatrixXd face_center = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/face_center.csv");
  Eigen:: MatrixXd face_info = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/face_info.csv");
  Eigen:: MatrixXd face_vector = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/face_vector.csv");
  Eigen:: MatrixXd gradient = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/gradient.csv");
  Eigen:: MatrixXd velocity = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/velocity.csv");
  Eigen:: MatrixXd velocity_interface = GReadData::ExtractDataFromCSV("F:/LBFSMethods/Project/src/Data/velocity_interface.csv");
  Eigen:: VectorXd cell_volume = GReadData::ExtractDataFromCSVVector("F:/LBFSMethods/Project/src/Data/cell_volume.csv");
  Eigen:: VectorXd convection_coe = GReadData::ExtractDataFromCSVVector("F:/LBFSMethods/Project/src/Data/convection_coe.csv");
  Eigen:: VectorXd transient_coe = GReadData::ExtractDataFromCSVVector("F:/LBFSMethods/Project/src/Data/transient_coe.csv");

  int cell_num{(int)cell_center.rows()};
  int face_num{(int)face_center.rows()};
  double cs2_{1/3.0};

  std::vector<Eigen::Matrix3d> diffusion_coe;
  diffusion_coe.reserve(face_num);
  diffusion_coe.resize(face_num);
  for (int i=0; i!=face_num; ++i) {
    auto info = face_cell.col(0);
    diffusion_coe[i](0,0)= cell_info.col(2)((int)(info)(i));
    diffusion_coe[i](0,1)=cell_info.col(3)((int)(info)(i));
    diffusion_coe[i](0,2)=cell_info.col(4)((int)(info)(i));
    diffusion_coe[i](1,0)=cell_info.col(5)((int)(info)(i));
    diffusion_coe[i](1,1)=cell_info.col(6)((int)(info)(i));
    diffusion_coe[i](1,0)=cell_info.col(7)((int)(info)(i));
    diffusion_coe[i](2,0)=cell_info.col(8)((int)(info)(i));
    diffusion_coe[i](2,1)=cell_info.col(9)((int)(info)(i));
    diffusion_coe[i](2,0)=cell_info.col(10)((int)(info)(i));
  }

  std::vector<Eigen::Triplet<double>> sparse_triplet{};
  for (int i=0; i!=face_num; ++i) {
    if (face_info.row(i)(0)==0) {
      int cell_0{(int) face_cell.row(i)(0)};
      int cell_1{(int) face_cell.row(i)(1)};

      Eigen::Vector3d flux_face{0.0, 0.0, 0.0};
      double stream_distance{(cell_center.row(cell_1) - cell_center.row(cell_0)).norm() / 2};
      Eigen::Vector3d e0{0, 0, 0};
      Eigen::Vector3d e1{(cell_center.row(cell_1) - cell_center.row(cell_0)).normalized()};
      Eigen::Vector3d e2{-1 * e1};
      Eigen::Vector3d velocity_fc{velocity.row(i)};
      Eigen::Vector3d scalar_i_0{0.5, 0.5, 0.0};
      Eigen::Vector3d scalar_i_1{1.0, 0.0, 0.0};
      Eigen::Vector3d scalar_i_2{0.0, 1.0, 0.0};
      Eigen::Vector3d feq_i_0{scalar_i_0 * this->calcFeqCoe(0, e0, velocity_fc)};
      Eigen::Vector3d feq_i_1{scalar_i_1 * this->calcFeqCoe(1, e1, velocity_fc)};
      Eigen::Vector3d feq_i_2{scalar_i_2 * this->calcFeqCoe(2, e2, velocity_fc)};
      Eigen::Vector3d scalar_lm{feq_i_0 + feq_i_1 + feq_i_2};
      Eigen::Vector3d feq_lm_0{scalar_lm * this->calcFeqCoe(0, e0, velocity_fc)};
      Eigen::Vector3d feq_lm_1{scalar_lm * this->calcFeqCoe(1, e1, velocity_fc)};
      Eigen::Vector3d feq_lm_2{scalar_lm * this->calcFeqCoe(2, e2, velocity_fc)};
      Eigen::Matrix3d unit_matrix{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      Eigen::Matrix3d relaxation_matrix{};
      relaxation_matrix = diffusion_coe[i] / (cs2_ * stream_distance) + 0.5 * unit_matrix;
      relaxation_matrix = (unit_matrix - 0.5 * relaxation_matrix.inverse()) * relaxation_matrix;
      Eigen::Vector3d flux_orthogonal{0.0, 0.0, 0.0};
      Eigen::Vector3d link_vec_unit{e1};
      Eigen::Vector3d vector_fc{face_vector.row(i)};
      Eigen::Vector3d orthogonal_vec{vector_fc.dot(vector_fc) / link_vec_unit.dot(vector_fc) * link_vec_unit};
      flux_orthogonal += -1 * (feq_lm_0 - feq_i_0) * (relaxation_matrix * e0).dot(orthogonal_vec);
      flux_orthogonal += -1 * (feq_lm_1 - feq_i_1) * (relaxation_matrix * e1).dot(orthogonal_vec);
      flux_orthogonal += -1 * (feq_lm_2 - feq_i_2) * (relaxation_matrix * e2).dot(orthogonal_vec);
      flux_face += flux_orthogonal;
      Eigen::Vector3d non_orthogonal_vec{vector_fc - orthogonal_vec};
      Eigen::Vector3d gradient_lm{(gradient.row(cell_0) + gradient.row(cell_1)) / 2};
      double flux_non_orthogonal{(-1 * diffusion_coe[i] * gradient_lm).dot(non_orthogonal_vec)};
      flux_face(2) += flux_non_orthogonal;

      Eigen::Vector3d location_lm{(cell_center.row(cell_1) + cell_center.row(cell_0)) / 2};
      Eigen::Vector3d scalar_fc{scalar_lm};
      Eigen::Vector3d face_center_1{face_center.row(i)};
      scalar_fc(2) += (face_center_1 - location_lm).dot(gradient_lm);
      Eigen::Vector3d flux_convection{convection_coe(cell_0) * scalar_fc * velocity_fc.dot(face_vector.row(i))};
      flux_face += flux_convection;

      sparse_triplet.emplace_back(cell_0, cell_0, flux_face(0));
      sparse_triplet.emplace_back(cell_0, cell_1, flux_face(1));
      b(cell_0) += -1 * flux_face(2);
      sparse_triplet.emplace_back(cell_1, cell_0, -1 * flux_face(0));
      sparse_triplet.emplace_back(cell_1, cell_1, -1 * flux_face(1));
      b(cell_1) += flux_face(2);
      exit(0);
    } else if (face_info.row(i)(0)==1) {
      int cell_0{(int) face_cell.row(i)(0)};

      double coe_orthogonal{0.0};
      double coe_non_orthogonal{0.0};
      GFVMDiscreter::diffusionTermDirichlet(cell_center.row(cell_0), face_center.row(i), face_vector.row(i),
                                            gradient.row(cell_0), diffusion_coe[i],
                                            coe_orthogonal, coe_non_orthogonal);

      sparse_triplet.emplace_back(cell_0, cell_0, -1 * coe_orthogonal);
      b(cell_0) += -1 * coe_orthogonal * face_info.row(i)(1);
      b(cell_0) += -1 * coe_non_orthogonal;

      Eigen::Vector3d flux_face{0.0, 0.0, 0.0};
      Eigen::Vector3d flux_convection{0.0, 0.0, 0.0};
      double coe{convection_coe(cell_0) * velocity.row(i).dot(face_vector.row(i))};

      if (coe>0) {
        flux_convection(0) = coe;
      } else {
        flux_convection(2) = coe;
      }
      flux_face += flux_convection;

      sparse_triplet.emplace_back(cell_0, cell_0, 1 * flux_face(0));
      b(cell_0) += -1 * flux_face(2) * face_info.row(i)(1);
    } else if (face_info.row(i)(0)==2) {
    } else if (face_info.row(i)(0)==5) {
      // get
      int cell_0{(int) face_cell.row(i)(0)};
      int cell_1{(int) face_cell.row(i)(1)};
      int cell_segment{(int) face_info.row(i)(1)};

      double coe_orthogonal{0.0};
      double coe_non_orthogonal{0.0};
      GFVMDiscreter::diffusionTermDirichlet(cell_center.row(cell_0), face_center.row(i), face_vector.row(i),
                                            gradient.row(cell_0), diffusion_coe[i],
                                            coe_orthogonal, coe_non_orthogonal);

      sparse_triplet.emplace_back(cell_0, cell_0, -1 * coe_orthogonal);
      sparse_triplet.emplace_back(cell_0, cell_segment, 1 * coe_orthogonal);
      b(cell_0) += -1 * coe_non_orthogonal;

      sparse_triplet.emplace_back(cell_segment, cell_segment, -1 * coe_orthogonal);
      sparse_triplet.emplace_back(cell_segment, cell_0, 1 * coe_orthogonal);
      b(cell_segment) += 1 * coe_non_orthogonal;

      coe_orthogonal = 0.0;
      coe_non_orthogonal = 0.0;
      GFVMDiscreter::diffusionTermDirichlet(cell_center.row(cell_1), face_center.row(i), face_vector.row(i),
                                            gradient.row(cell_1), diffusion_coe[i],
                                            coe_orthogonal, coe_non_orthogonal);

      sparse_triplet.emplace_back(cell_1, cell_1, 1 * coe_orthogonal);
      sparse_triplet.emplace_back(cell_1, cell_segment, -1 * coe_orthogonal);
      b(cell_1) += 1 * coe_non_orthogonal;

      sparse_triplet.emplace_back(cell_segment, cell_segment, 1 * coe_orthogonal);
      sparse_triplet.emplace_back(cell_segment, cell_1, -1 * coe_orthogonal);
      b(cell_segment) += -1 * coe_non_orthogonal;

      Eigen::Vector3d flux_face_0{0.0, 0.0, 0.0};
      Eigen::Vector3d flux_convection_0{0.0, 0.0, 0.0};
      auto velocity_0 = velocity_interface.row(cell_0);
      double coe_0{convection_coe(cell_0) * velocity_0.dot(face_vector.row(i))};

      if (coe_0>0) {
        flux_convection_0(0) = coe_0;
      } else {
        flux_convection_0(1) = coe_0;
      }
      flux_face_0 += flux_convection_0;

      sparse_triplet.emplace_back(cell_0, cell_0, 1 * flux_face_0(0));
      sparse_triplet.emplace_back(cell_0, cell_segment, 1 * flux_face_0(1));
      sparse_triplet.emplace_back(cell_segment, cell_0, -1 * flux_face_0(0));
      sparse_triplet.emplace_back(cell_segment, cell_segment, -1 * flux_face_0(1));

      Eigen::Vector3d flux_face_1{0.0, 0.0, 0.0};
      Eigen::Vector3d flux_convection_1{0.0, 0.0, 0.0};
      auto velocity_1 = velocity_interface.row(cell_1);
      double coe_1{-1 * convection_coe(cell_1) * velocity_1.dot(face_vector.row(i))};
      if (coe_1>0) {
        flux_convection_1(0) = coe_1;
      } else {
        flux_convection_1(1) = coe_1;
      }
      flux_face_1 += flux_convection_1;

      sparse_triplet.emplace_back(cell_1, cell_1, 1 * flux_face_1(0));
      sparse_triplet.emplace_back(cell_1, cell_segment, 1 * flux_face_1(1));
      sparse_triplet.emplace_back(cell_segment, cell_1, -1 * flux_face_1(0));
      sparse_triplet.emplace_back(cell_segment, cell_segment, -1 * flux_face_1(1));
    } else if (face_info.row(i)(0)==6) {
      int cell_0{(int) face_cell.row(i)(0)};
      Eigen::Vector3d flux_face{0.0, 0.0, 0.0};
      Eigen::Vector3d flux_convection{0.0, 0.0, 0.0};
      double coe{convection_coe(cell_0) * velocity.row(i).dot(face_vector.row(i))};
      if (coe>0) {
        flux_convection(0) = coe;
      } else {
        flux_convection(2) = coe;
      }
      flux_face += flux_convection;
      sparse_triplet.emplace_back(cell_0, cell_0, 1 * flux_face(0));
      b(cell_0) += -1 * flux_face(2) * face_info.row(i)(1);
    }
  }

  for (int i=0; i!=cell_num; ++i) {
    double coe_source{0.0};
    GFVMDiscreter::sourceTerm(cell_info.row(i)(0), coe_source);
    b(i) += coe_source;

    double coe_transient{0.0};
    GFVMDiscreter::transientTerm(transient_coe(i), cell_volume(i), delta_t, coe_transient);
    sparse_triplet.emplace_back(i, i, coe_transient);
    b(i) += coe_transient*(*this->getScalarPtr())(i);
  }

  A.setFromTriplets(sparse_triplet.begin(), sparse_triplet.end());
  A.makeCompressed();
}
