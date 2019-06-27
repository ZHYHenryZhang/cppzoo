#include <iostream>
#include <cassert>
#include <vector>
#include <ctime>
#include <eigen3/Eigen/Dense>

void prepareData(unsigned int ROW, unsigned int COL, std::vector< std::vector<double> > &dataMatrix, Eigen::MatrixXd &points_xy)
{
  float A = 1.0;
  float B = 1.0;
  for (size_t i = 0; i < ROW; i++)
  {
    // std::cout << i << std::endl;
    std::vector<double> point;
    point.push_back( i);
    point.push_back( A * i + B);
    dataMatrix.push_back(point);
    points_xy(i,0) = i;
    points_xy(i,1) = A * i + B;
  }
}

void leastSquareLoop(std::vector< std::vector<double> > &dataMatrix)
{
  double sum_x, sum_y, sum_xy, sum_xx;
  sum_x = 0;
  sum_y = 0;
  sum_xy = 0;
  sum_xx = 0;
  size_t point_size = dataMatrix.size();
  for (auto &i : dataMatrix)
  {
    // std::cout << i[0] << " " << i[1] << std::endl;
    sum_x += i[0];
    sum_y += i[1];
    sum_xy += i[0] * i[1];
    sum_xx += i[0] * i[0];
  }
  double a = (point_size * sum_xy - sum_x * sum_y) / (point_size * sum_xx - sum_x * sum_x);
  double b = (sum_y - a * sum_x) / point_size; 
  std::cout << "a: " << a << " b:" << b << " size:" << point_size << std::endl;
}

void leastSquareEigenSVD(Eigen::MatrixXd &points_xy, Eigen::VectorXd &points_homo)
{
  Eigen::VectorXd line_param = points_xy.bdcSvd(Eigen::ComputeThinU| Eigen::ComputeThinV).solve(points_homo);
  std::cout << line_param << std::endl;
}

void leastSquareEigenQR(Eigen::MatrixXd &points_xy, Eigen::VectorXd &points_homo)
{
  std::cout << points_xy.colPivHouseholderQr().solve(points_homo) << std::endl;
}

void leastSquareEigenNE(Eigen::MatrixXd &points_xy, Eigen::VectorXd &points_homo)
{
  std::cout << (points_xy.transpose() * points_xy).ldlt().solve(points_xy.transpose() * points_homo) << std::endl;
}

int main(int argc, char const *argv[])
{   
  unsigned int ROW = 1000;
  unsigned int COL = 2;
  std::vector< std::vector<double> > dataMatrix;
  Eigen::MatrixXd points_xy(ROW,COL);
  Eigen::VectorXd points_homo = Eigen::VectorXd::Ones(ROW);
  prepareData(ROW, COL, dataMatrix, points_xy);
  std::cout << "hello" << std::endl;
  assert(dataMatrix[3][1] == 4);
  assert(dataMatrix[3][0] == 3);
  assert(points_xy(3,0) == 3);
  assert(points_xy(3,1) == 4);

  clock_t time_ls_loop;
  time_ls_loop = clock();
  leastSquareLoop(dataMatrix);
  time_ls_loop = clock() - time_ls_loop;

  clock_t time_ls_eigen_SVD;
  time_ls_eigen_SVD = clock();
  leastSquareEigenSVD(points_xy, points_homo);
  time_ls_eigen_SVD = clock() - time_ls_eigen_SVD;

  clock_t time_ls_eigen_QR;
  time_ls_eigen_QR = clock();
  leastSquareEigenQR(points_xy, points_homo);
  time_ls_eigen_QR = clock() - time_ls_eigen_QR;

  clock_t time_ls_eigen_NE;
  time_ls_eigen_NE = clock();
  leastSquareEigenNE(points_xy, points_homo);
  time_ls_eigen_NE = clock() - time_ls_eigen_NE;

  std::cout << "ls loop:" << (float)time_ls_loop/CLOCKS_PER_SEC << " seconds" << std::endl;
  std::cout << "ls SVD:" << (float)time_ls_eigen_SVD/CLOCKS_PER_SEC << " seconds" << std::endl;
  std::cout << "ls QR:" << (float)time_ls_eigen_QR/CLOCKS_PER_SEC << " seconds" << std::endl;
  std::cout << "ls NE:" << (float)time_ls_eigen_NE/CLOCKS_PER_SEC << " seconds" << std::endl;

  return 0;
}
