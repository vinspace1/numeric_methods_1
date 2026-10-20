#pragma once
#include <vector>
#include <string>

namespace jacobi {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct JacobiResult {
    Vector eigenvalues;
    Matrix eigenvectors;
    Matrix diagonalized;
    int sweeps = 0;
    bool converged = false;
    std::string error;
};

bool isSymmetric(const Matrix& A, double tol = 1e-10);
Matrix symmetrize(const Matrix& A);
JacobiResult solve(const Matrix& A, double eps, int maxIter = 1000000);

double normInf(const Matrix& M);
double offDiagFrobenius(const Matrix& A);

Matrix multiply(const Matrix& A, const Matrix& B);
Matrix buildLambda(const Vector& lambda);

void printReport(const Matrix& A, const JacobiResult& res, double eps);

} 