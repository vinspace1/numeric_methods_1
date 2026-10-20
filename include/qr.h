#pragma once
#include <vector>
#include <string>
#include <complex>

namespace qr {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

struct QRResult {
    Matrix Q, R;
    bool ok = false;
    std::string error;
};
QRResult decompose(const Matrix& A);

struct EigenResult {
    ComplexVector eigenvalues;
    int iterations = 0;
    bool converged = false;
    std::string error;
};
EigenResult eigenvalues(const Matrix& A, double eps = 1e-12, int maxIter = 100000);

Matrix multiply(const Matrix& A, const Matrix& B);
double normInf(const Matrix& M);
double orthogonalityError(const Matrix& Q);
void printMatrix(const Matrix& M, const std::string& name);
void printComplexVector(const ComplexVector& v, const std::string& name);
void printReport(const Matrix& A, const QRResult& qr,
                 const EigenResult& eig, double eps);

} 