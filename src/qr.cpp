#include "qr.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace qr {

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

double normInf(const Matrix& M) {
    double m = 0.0;
    for (const auto& row : M) {
        double s = 0.0;
        for (double x : row) s += std::fabs(x);
        m = std::max(m, s);
    }
    return m;
}

double orthogonalityError(const Matrix& Q) {
    const int n = static_cast<int>(Q.size());
    double maxErr = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int t = 0; t < n; ++t) s += Q[t][i] * Q[t][j];
            double expected = (i == j) ? 1.0 : 0.0;
            maxErr = std::max(maxErr, std::fabs(s - expected));
        }
    return maxErr;
}

QRResult decompose(const Matrix& A) {
    QRResult res;
    const int n = static_cast<int>(A.size());
    const int m = static_cast<int>(A[0].size());
    if (n == 0 || m == 0) { res.error = "Empty matrix"; return res; }

    Matrix R = A;
    Matrix Q(n, Vector(n, 0.0));
    for (int i = 0; i < n; ++i) Q[i][i] = 1.0;

    const int k_max = std::min(n, m);
    for (int k = 0; k < k_max; ++k) {
        double normX = 0.0;
        for (int i = k; i < n; ++i) normX += R[i][k] * R[i][k];
        normX = std::sqrt(normX);
        if (normX < 1e-300) continue;

        double x0 = R[k][k];
        double alpha = (x0 >= 0.0) ? normX : -normX;
        Vector v(n - k, 0.0);
        for (int i = 0; i < n - k; ++i) v[i] = R[k + i][k];
        v[0] += alpha;

        double vtv = 0.0;
        for (double vi : v) vtv += vi * vi;
        if (vtv < 1e-300) continue;

        for (int j = k; j < m; ++j) {
            double dot = 0.0;
            for (int i = 0; i < n - k; ++i) dot += v[i] * R[k + i][j];
            double coef = 2.0 * dot / vtv;
            for (int i = 0; i < n - k; ++i) R[k + i][j] -= coef * v[i];
        }
        for (int i = 0; i < n; ++i) {
            double dot = 0.0;
            for (int j = 0; j < n - k; ++j) dot += Q[i][k + j] * v[j];
            double coef = 2.0 * dot / vtv;
            for (int j = 0; j < n - k; ++j) Q[i][k + j] -= coef * v[j];
        }
    }

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < std::min(i, m); ++j) R[i][j] = 0.0;

    res.Q = std::move(Q);
    res.R = std::move(R);
    res.ok = true;
    return res;
}

static void qrStep(Matrix& A) {
    const int n = static_cast<int>(A.size());
    if (n < 2) return;

    double a = A[n - 2][n - 2], b = A[n - 2][n - 1];
    double c = A[n - 1][n - 2], d = A[n - 1][n - 1];
    double tr = a + d, det = a * d - b * c;
    double disc = tr * tr - 4.0 * det;
    double mu;
    if (disc >= 0.0) {
        double s1 = 0.5 * (tr + std::sqrt(disc));
        double s2 = 0.5 * (tr - std::sqrt(disc));
        mu = (std::fabs(s1 - d) < std::fabs(s2 - d)) ? s1 : s2;
    } else {
        mu = d;
    }

    Matrix S = A;
    for (int i = 0; i < n; ++i) S[i][i] -= mu;

    QRResult qrRes = decompose(S);
    Matrix RQ = multiply(qrRes.R, qrRes.Q);
    for (int i = 0; i < n; ++i) RQ[i][i] += mu;
    A = std::move(RQ);
}

EigenResult eigenvalues(const Matrix& A0, double eps, int maxIter) {
    EigenResult res;
    const int n = static_cast<int>(A0.size());
    if (n == 0) { res.error = "Empty matrix"; return res; }

    Matrix A = A0;
    int iter = 0;
    int m = n;
    ComplexVector lambdas;

    while (m > 0) {
        if (m == 1) {
            lambdas.push_back(A[0][0]);
            break;
        }

        double off = std::fabs(A[m - 1][m - 2]);
        double scale = std::fabs(A[m - 2][m - 2]) + std::fabs(A[m - 1][m - 1]) + 1e-300;
        if (off < eps * scale) {
            lambdas.push_back(A[m - 1][m - 1]);
            --m;
            continue;
        }

        if (m >= 2) {
            double off2 = (m >= 3) ? std::fabs(A[m - 2][m - 3]) : 0.0;
            double scale2 = (m >= 3)
                ? std::fabs(A[m - 3][m - 3]) + std::fabs(A[m - 2][m - 2]) + 1e-300
                : 1.0;
            if (m == 2 || off2 < eps * scale2) {
                double a = A[m - 2][m - 2], b = A[m - 2][m - 1];
                double c = A[m - 1][m - 2], d = A[m - 1][m - 1];
                double tr = a + d, det = a * d - b * c;
                double disc = tr * tr - 4.0 * det;
                if (disc >= 0.0) {
                    lambdas.push_back(0.5 * (tr + std::sqrt(disc)));
                    lambdas.push_back(0.5 * (tr - std::sqrt(disc)));
                } else {
                    double re = 0.5 * tr, im = 0.5 * std::sqrt(-disc);
                    lambdas.push_back(Complex(re,  im));
                    lambdas.push_back(Complex(re, -im));
                }
                m -= 2;
                continue;
            }
        }

        if (iter++ >= maxIter) { res.error = "QR iteration limit reached"; break; }

        Matrix block(m, Vector(m, 0.0));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) block[i][j] = A[i][j];
        qrStep(block);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) A[i][j] = block[i][j];
    }

    res.eigenvalues = lambdas;
    res.iterations = iter;
    res.converged = res.error.empty();
    return res;
}

void printMatrix(const Matrix& M, const std::string& name) {
    std::cout << name << ":\n";
    for (const auto& row : M) {
        std::cout << "   ";
        for (double v : row)
            std::cout << std::setw(14) << std::setprecision(6)
                      << std::fixed << v << ' ';
        std::cout << '\n';
    }
    std::cout << '\n';
}

void printComplexVector(const ComplexVector& v, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << "   lambda[" << i << "] = ";
        if (std::fabs(v[i].imag()) < 1e-10)
            std::cout << std::setw(16) << std::setprecision(8) << std::fixed
                      << v[i].real() << '\n';
        else
            std::cout << std::setw(16) << std::setprecision(8) << std::fixed
                      << v[i].real()
                      << (v[i].imag() >= 0 ? " + " : " - ")
                      << std::setw(10) << std::fabs(v[i].imag()) << "i\n";
    }
    std::cout << '\n';
}

void printReport(const Matrix& A, const QRResult& qr,
                 const EigenResult& eig, double eps) {
    std::cout << "=== QR decomposition (Householder) ===\n";
    std::cout << "1)";
    printMatrix(A, "   A");
    if (!qr.ok) { std::cout << "Error: " << qr.error << '\n'; return; }

    std::cout << "2) Q and R:\n";
    printMatrix(qr.Q, "   Q");
    printMatrix(qr.R, "   R");

    Matrix QR = multiply(qr.Q, qr.R);
    double maxDiff = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[i].size(); ++j)
            maxDiff = std::max(maxDiff, std::fabs(QR[i][j] - A[i][j]));

    std::cout << "3)";
    printMatrix(QR, "   Q*R");
    std::cout << "   ||Q*R - A|| = " << std::scientific << std::setprecision(3)
              << maxDiff << "   ||Q^T Q - E|| = " << orthogonalityError(qr.Q)
              << std::defaultfloat << '\n';

    std::cout << "4) eigenvalues (steps: " << eig.iterations << "):\n";
    printComplexVector(eig.eigenvalues, "   lambda");
    std::cout << '\n';
}

} 