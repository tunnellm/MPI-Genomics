#ifndef            __MPI_FUNCTIONS_HEADER_CIS_677__
#define            __MPI_FUNCTIONS_HEADER_CIS_677__
#include <vector>
#include <string>

auto doStuff(const double * dataVec, const int & vectorSize, const int & repeats) -> std::vector<double> &;
auto loadDataVector(std::string & file) -> std::vector<double> &;
auto studentT(const double * dataVec, const int & vectorSize) -> std::vector<double> &;
auto calculateTheD(const double * dataVec, const double * studentT, const int & vectorSize, const int & repeats) -> std::vector<double> &;

#endif