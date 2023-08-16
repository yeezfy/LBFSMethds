
#ifndef PROJECT_GREADDATA_H
#define PROJECT_GREADDATA_H

#include <fstream>
#include "../../lib/eigen-3.4/Dense"

namespace GReadData {
  inline Eigen:: MatrixXd ExtractDataFromCSV(const std::string& filename) {
    std::ifstream file(filename); // Replace "data.csv" with your CSV file's name
    if (!file.is_open()) {
      std::cerr << "Failed to open the file." << std::endl;
    }
    std::vector<std::vector<double>> data;
    std::string line;

    while (getline(file, line)) {
      std::vector<double> row;
      std::istringstream lineStream(line);
      std::string cell;
      while (getline(lineStream, cell, ',')) {
        row.push_back(std::stod(cell));
      }
      data.push_back(row);
    }

    file.close();

    int rows = data.size();
    int cols = data.empty() ? 0 : data[0].size();
    Eigen::MatrixXd matrixData(rows, cols);
    for (int i=0; i!=rows; ++i) {
      for (int j=0; j!=cols; ++j) {
        matrixData(i, j) = data[i][j];
      }
    }
    return matrixData;
  }

  inline Eigen:: VectorXd ExtractDataFromCSVVector(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cout<<"Failed to open the file."<<std::endl;
    }
    std::vector<double> data;
    std::string line;

    while (getline(file, line)) {
      std::istringstream lineStream(line);
      std::string cell;
      while (getline(lineStream, cell, ',')) {
        data.push_back(std::stod(cell));
      }
    }
    file.close();

    Eigen::VectorXd vectorData(data.size());
    for (int i=0; i!=data.size(); ++i) {
      vectorData(i) = data[i];
    }
    return vectorData;
  }
}

#endif //PROJECT_GREADDATA_H
