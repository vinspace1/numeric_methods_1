#pragma once
#include <vector>
#include <string>

namespace it {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct IterResult {
    Vector x;
    int iterations = 0;
    bool converged = false;
    std::string error;
    double residualInf = 0.0;
};

bool enforceDiagonalDominance(Matrix& A, Vector& b,
                              std::string* reason = nullptr);

struct IterationScheme {
    Matrix B;
    Vector g;
    bool ok = false;
    std::string error;
};
IterationScheme buildScheme(const Matrix& A, const Vector& b);

IterResult simpleIteration(const Matrix& A, const Vector& b,
                           double eps, int maxIter = 100000);
IterResult seidel(const Matrix& A, const Vector& b,
                  double eps, int maxIter = 100000);

double normInf(const Matrix& M);
double normInf(const Vector& v);
double residual(const Matrix& A, const Vector& b, const Vector& x);

struct ComparisonReport {
    double eps = 0.0;
    IterResult simple, seidel;
    Vector reference;
    bool hasReference = false;
};
void printComparison(const ComparisonReport& rep);

} 