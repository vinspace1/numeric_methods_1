#include "lu.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <algorithm>

namespace lu {

LUResult decompose(const Matrix& A, PivotStrategy strategy) {
    LUResult res;
    res.strategy = strategy;
    const int n = static_cast<int>(A.size());
    if (n == 0) { res.error = "Empty matrix"; return res; }
    for (const auto& row : A)
        if (static_cast<int>(row.size()) != n) {
            res.error = "Matrix is not square";
            return res;
        }

    Matrix U = A;
    Matrix L(n, Vector(n, 0.0));
    Matrix P(n, Vector(n, 0.0));
    Matrix Q(n, Vector(n, 0.0));
    res.rowPerm.resize(n);
    res.colPerm.resize(n);
    for (int i = 0; i < n; ++i) {
        L[i][i] = 1.0;
        P[i][i] = 1.0;
        Q[i][i] = 1.0;
        res.rowPerm[i] = i;
        res.colPerm[i] = i;
    }

    for (int k = 0; k < n; ++k) {
        int pivotRow = k, pivotCol = k;
        double maxVal = 0.0;
        switch (strategy) {
            case PivotStrategy::None:
                maxVal = std::fabs(U[k][k]);
                break;
            case PivotStrategy::Partial:
                maxVal = std::fabs(U[k][k]);
                for (int i = k + 1; i < n; ++i)
                    if (std::fabs(U[i][k]) > maxVal) {
                        maxVal = std::fabs(U[i][k]);
                        pivotRow = i;
                    }
                break;
            case PivotStrategy::Full:
                for (int i = k; i < n; ++i)
                    for (int j = k; j < n; ++j) {
                        double v = std::fabs(U[i][j]);
                        if (v > maxVal) {
                            maxVal = v;
                            pivotRow = i;
                            pivotCol = j;
                        }
                    }
                break;
        }
        if (maxVal < 1e-14) {
            res.error = "Matrix is singular";
            return res;
        }

        if (pivotRow != k) {
            std::swap(U[k], U[pivotRow]);
            std::swap(P[k], P[pivotRow]);
            std::swap(res.rowPerm[k], res.rowPerm[pivotRow]);
            for (int j = 0; j < k; ++j)
                std::swap(L[k][j], L[pivotRow][j]);
        }

        if (pivotCol != k) {
            for (int i = 0; i < n; ++i)
                std::swap(U[i][k], U[i][pivotCol]);
            for (int i = 0; i < n; ++i)
                std::swap(Q[i][k], Q[i][pivotCol]);
            std::swap(res.colPerm[k], res.colPerm[pivotCol]);
        }

        for (int i = k + 1; i < n; ++i) {
            double factor = U[i][k] / U[k][k];
            L[i][k] = factor;
            for (int j = k; j < n; ++j)
                U[i][j] -= factor * U[k][j];
        }
    }

    res.L = std::move(L);
    res.U = std::move(U);
    res.P = std::move(P);
    res.Q = std::move(Q);
    res.ok = true;
    return res;
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

Vector multiply(const Matrix& A, const Vector& b) {
    const int n = static_cast<int>(A.size());
    Vector r(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += A[i][j] * b[j];
        r[i] = s;
    }
    return r;
}

Vector solve(const LUResult& lu, const Vector& b) {
    if (!lu.ok) throw std::runtime_error("LU not decomposed");
    const int n = static_cast<int>(b.size());

    Vector Pb(n);
    for (int i = 0; i < n; ++i) Pb[i] = b[lu.rowPerm[i]];

    Vector y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = Pb[i];
        for (int j = 0; j < i; ++j) s -= lu.L[i][j] * y[j];
        y[i] = s;
    }

    Vector y2(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < n; ++j) s -= lu.U[i][j] * y2[j];
        y2[i] = s / lu.U[i][i];
    }

    Vector x(n, 0.0);
    for (int j = 0; j < n; ++j) x[lu.colPerm[j]] = y2[j];
    return x;
}

Matrix inverse(const LUResult& lu) {
    const int n = static_cast<int>(lu.L.size());
    Matrix inv(n, Vector(n, 0.0));
    for (int j = 0; j < n; ++j) {
        Vector e(n, 0.0);
        e[j] = 1.0;
        Vector col = solve(lu, e);
        for (int i = 0; i < n; ++i) inv[i][j] = col[i];
    }
    return inv;
}

bool isIdentity(const Matrix& M, double eps) {
    const int n = static_cast<int>(M.size());
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double e = (i == j) ? 1.0 : 0.0;
            if (std::fabs(M[i][j] - e) > eps) return false;
        }
    return true;
}

double maxAbsDiff(const Matrix& A, const Matrix& B) {
    double d = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[i].size(); ++j)
            d = std::max(d, std::fabs(A[i][j] - B[i][j]));
    return d;
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

void printVector(const Vector& v, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < v.size(); ++i)
        std::cout << "   [" << i << "] = " << std::setw(16)
                  << std::setprecision(8) << std::fixed << v[i] << '\n';
    std::cout << '\n';
}

Task loadTask(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);
    
    unsigned char b0 = 0, b1 = 0, b2 = 0;
    in.get(reinterpret_cast<char&>(b0));
    in.get(reinterpret_cast<char&>(b1));
    in.get(reinterpret_cast<char&>(b2));
    if (!(b0 == 0xEF && b1 == 0xBB && b2 == 0xBF)) {
        in.clear();
        in.seekg(0);
    }

    Task t;
    if (!(in >> t.n) || t.n <= 0)
        throw std::runtime_error("Invalid n");

    t.A.assign(t.n, Vector(t.n, 0.0));
    for (int i = 0; i < t.n; ++i)
        for (int j = 0; j < t.n; ++j)
            if (!(in >> t.A[i][j]))
                throw std::runtime_error("Read error in A");

    t.b.assign(t.n, 0.0);
    for (int i = 0; i < t.n; ++i)
        if (!(in >> t.b[i]))
            throw std::runtime_error("Read error in b");

    return t;
}

} 