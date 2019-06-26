#include <iostream>
#include <cassert>
#include <vector>
#include <ctime>

void prepareData(unsigned int ROW, unsigned int COL, std::vector< std::vector<float>> &dataMatrix)
{
    float A = 1.0;
    float B = 1.0;
    for (size_t i = 0; i < ROW; i++)
    {
        // std::cout << i << std::endl;
        std::vector<float> point;
        point.push_back( i);
        point.push_back( A * i + B);
        dataMatrix.push_back(point);
    }
}

void leastSquareLoop(std::vector< std::vector<float>> dataMatrix)
{

}

void leastSquareEigen(std::vector< std::vector<float>> dataMatrix)
{

}

int main(int argc, char const *argv[])
{   
    unsigned int ROW = 1000;
    unsigned int COL = 2;
    std::vector< std::vector<float>> dataMatrix;
    prepareData(ROW, COL, dataMatrix);
    std::cout << "hello" << std::endl;
    assert(dataMatrix[3][1] == 4);
    assert(dataMatrix[3][0] == 3);

    clock_t time_ls_loop;
    time_ls_loop = clock();
    leastSquareLoop(dataMatrix);
    time_ls_loop = clock() - time_ls_loop;

    clock_t time_ls_eigen;
    time_ls_eigen = clock();
    leastSquareEigen(dataMatrix);
    time_ls_eigen = clock() - time_ls_eigen;

    std::cout << "ls loop:" << (float)time_ls_loop/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "ls loop:" << (float)time_ls_eigen/CLOCKS_PER_SEC << " seconds" << std::endl;

    
    return 0;
}
