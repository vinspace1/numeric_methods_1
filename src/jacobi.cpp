#include "jacobi.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace jacobi {

double normInf(const Matrix& M) {
    double m = 0.0;
    for (const auto& row : M) {
        double s = 0.0;
        for (double x : row) s += std::fabs(x);
        m = std::max(m, s);
    }
    return m;
}

double offDiagFrobenius(const Matrix& A) {
    const int n = static_cast<int>(A.size());
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) s += A[i][j] * A[i][j];
    return std::sqrt(s);
}

bool isSymmetric(const Matrix& A, double tol) {
    const int n = static_cast<int>(A.size());
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (std::fabs(A[i][j] - A[j][i]) > tol) return false;
    return true;
}

Matrix symmetrize(const Matrix& A) {
    const int n = static_cast<int>(A.size());
    Matrix B(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B[i][j] = 0.5 * (A[i][j] + A[j][i]);
    return B;
}

Matrix multiply(const Matrix& A, const Matrix& B) {
    const int n = static_cast<int>(A.size());
    const int m = static_cast<int>(B[0].size());
    const int k = static_cast<int>(B.size());
    Matrix C(n, Vector(m, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int t = 0; t < k; ++t) s += A[i][t] * B[t][j];
            C[i][j] = s;
        }
    return C;
}

Matrix buildLambda(const Vector& lambda) {
    const int n = static_cast<int>(lambda.size());
    Matrix L(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) L[i][i] = lambda[i];
    return L;
}

JacobiResult solve(const Matrix& A0, double eps, int maxIter) {
    JacobiResult res;
    const int n = static_cast<int>(A0.size());
    if (n == 0) { res.error = "Empty matrix"; return res; }

    Matrix A = isSymmetric(A0, 1e-10) ? A0 : symmetrize(A0);

    Matrix V(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1.0;

    const double target = eps * eps;
    int iter = 0;

    while (offDiagFrobenius(A) * offDiagFrobenius(A) > target) {
        if (iter++ >= maxIter) { res.error = "Iteration limit reached"; break; }

        int p = 0, q = 1;
        double maxVal = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (std::fabs(A[i][j]) > maxVal) {
                    maxVal = std::fabs(A[i][j]); p = i; q = j;
                }
        if (maxVal < 1e-300) break;

        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        double theta = (aqq - app) / (2.0 * apq);
        double t;
        if (std::fabs(theta) > 1e150) t = 0.5 / theta;
        else {
            double sign = (theta >= 0.0) ? 1.0 : -1.0;
            t = sign / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        }
        double c = 1.0 / std::sqrt(t * t + 1.0);
        double s = t * c;

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) continue;
            double akp = A[k][p], akq = A[k][q];
            A[k][p] = A[p][k] = c * akp - s * akq;
            A[k][q] = A[q][k] = s * akp + c * akq;
        }
        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = A[q][p] = 0.0;

        for (int k = 0; k < n; ++k) {
            double vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
        }
    }

    res.sweeps = iter;
    res.converged = res.error.empty();
    res.eigenvalues.resize(n);
    for (int i = 0; i < n; ++i) res.eigenvalues[i] = A[i][i];
    res.eigenvectors = V;
    res.diagonalized = A;
    return res;
}

void printReport(const Matrix& A0, const JacobiResult& res, double eps) {
    std::cout << "=== Jacobi rotation method ===\n";
    std::cout << "1) eps = " << std::scientific << std::setprecision(3)
              << eps << std::defaultfloat << '\n';

    const int n = static_cast<int>(res.eigenvalues.size());
    std::cout << "2) eigenvalues (rotations: " << res.sweeps << "):\n";
    for (int i = 0; i < n; ++i)
        std::cout << "   lambda[" << i << "] = " << std::setw(16)
                  << std::setprecision(8) << std::fixed << res.eigenvalues[i] << '\n';

    std::cout << "3) eigenvector matrix V (columns):\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "   ";
        for (int j = 0; j < n; ++j)
            std::cout << std::setw(14) << std::setprecision(6) << std::fixed
                      << res.eigenvectors[i][j] << ' ';
        std::cout << '\n';
    }

    Matrix AV = multiply(A0, res.eigenvectors);
    Matrix Lambda = buildLambda(res.eigenvalues);
    Matrix VL = multiply(res.eigenvectors, Lambda);
    double maxDiff = 0.0;
    for (size_t i = 0; i < AV.size(); ++i)
        for (size_t j = 0; j < AV[i].size(); ++j)
            maxDiff = std::max(maxDiff, std::fabs(AV[i][j] - VL[i][j]));

    std::cout << "4) ||A*V - V*Lambda||_max = " << std::scientific
              << std::setprecision(3) << maxDiff << std::defaultfloat
              << (maxDiff < std::max(1e-9, eps * 100.0) ? "  OK" : "  FAIL")
              << "\n\n";
}

}