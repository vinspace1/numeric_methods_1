#include "iterative.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <string>

namespace it {

double normInf(const Vector& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
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

double residual(const Matrix& A, const Vector& b, const Vector& x) {
    const int n = static_cast<int>(A.size());
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += A[i][j] * x[j];
        m = std::max(m, std::fabs(s - b[i]));
    }
    return m;
}

bool enforceDiagonalDominance(Matrix& A, Vector& b, std::string* reason) {
    const int n = static_cast<int>(A.size());
    std::vector<bool> used(n, false);
    Matrix A2(n, Vector(n, 0.0));
    Vector b2(n, 0.0);

    for (int i = 0; i < n; ++i) {
        int pick = -1;
        for (int r = 0; r < n; ++r) {
            if (used[r]) continue;
            double diag = std::fabs(A[r][i]);
            double off = 0.0;
            for (int j = 0; j < n; ++j)
                if (j != i) off += std::fabs(A[r][j]);
            if (diag > off) { pick = r; break; }
        }
        if (pick < 0)
            for (int r = 0; r < n; ++r)
                if (!used[r]) { pick = r; break; }
        used[pick] = true;
        A2[i] = A[pick];
        b2[i] = b[pick];
    }
    A = std::move(A2);
    b = std::move(b2);

    for (int i = 0; i < n; ++i) {
        double diag = std::fabs(A[i][i]);
        double off = 0.0;
        for (int j = 0; j < n; ++j)
            if (j != i) off += std::fabs(A[i][j]);
        if (diag <= off) {
            if (reason) *reason = "Cannot achieve diagonal dominance";
            return false;
        }
    }
    return true;
}

IterationScheme buildScheme(const Matrix& A, const Vector& b) {
    IterationScheme s;
    const int n = static_cast<int>(A.size());
    for (int i = 0; i < n; ++i)
        if (std::fabs(A[i][i]) < 1e-15) {
            s.error = "Zero diagonal element in row " + std::to_string(i);
            return s;
        }
    s.B.assign(n, Vector(n, 0.0));
    s.g.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (i != j) s.B[i][j] = -A[i][j] / A[i][i];
        s.g[i] = b[i] / A[i][i];
    }
    s.ok = true;
    return s;
}

IterResult simpleIteration(const Matrix& A, const Vector& b,
                           double eps, int maxIter) {
    IterResult r;
    const int n = static_cast<int>(A.size());
    IterationScheme s = buildScheme(A, b);
    if (!s.ok) { r.error = s.error; return r; }

    Vector x(n, 0.0), xNew(n, 0.0);
    for (int k = 1; k <= maxIter; ++k) {
        for (int i = 0; i < n; ++i) {
            double acc = s.g[i];
            for (int j = 0; j < n; ++j)
                if (j != i) acc += s.B[i][j] * x[j];
            xNew[i] = acc;
        }
        double diff = 0.0;
        for (int i = 0; i < n; ++i)
            diff = std::max(diff, std::fabs(xNew[i] - x[i]));
        x = xNew;
        r.iterations = k;
        if (diff < eps) {
            r.converged = true; r.x = x;
            r.residualInf = residual(A, b, x);
            return r;
        }
    }
    r.x = x; r.residualInf = residual(A, b, x);
    r.error = "Iteration limit reached";
    return r;
}

IterResult seidel(const Matrix& A, const Vector& b,
                  double eps, int maxIter) {
    IterResult r;
    const int n = static_cast<int>(A.size());
    IterationScheme s = buildScheme(A, b);
    if (!s.ok) { r.error = s.error; return r; }

    Vector x(n, 0.0), xPrev(n, 0.0);
    for (int k = 1; k <= maxIter; ++k) {
        xPrev = x;
        for (int i = 0; i < n; ++i) {
            double acc = s.g[i];
            for (int j = 0; j < n; ++j)
                if (j != i) acc += s.B[i][j] * x[j];
            x[i] = acc;
        }
        double diff = 0.0;
        for (int i = 0; i < n; ++i)
            diff = std::max(diff, std::fabs(x[i] - xPrev[i]));
        r.iterations = k;
        if (diff < eps) {
            r.converged = true; r.x = x;
            r.residualInf = residual(A, b, x);
            return r;
        }
    }
    r.x = x; r.residualInf = residual(A, b, x);
    r.error = "Iteration limit reached";
    return r;
}

void printComparison(const ComparisonReport& rep) {
    std::cout << "=== Iterative methods ===\n";
    std::cout << "1) eps = " << std::scientific << std::setprecision(3)
              << rep.eps << std::defaultfloat << '\n';

    std::cout << "2) iterations: simple = " << rep.simple.iterations
              << (rep.simple.converged ? "" : " (not converged)")
              << ", Seidel = " << rep.seidel.iterations
              << (rep.seidel.converged ? "" : " (not converged)") << '\n';

    std::cout << "3) reference from LU:\n";
    if (rep.hasReference) {
        for (size_t i = 0; i < rep.reference.size(); ++i)
            std::cout << "   x[" << i << "] = " << std::setw(14)
                      << std::setprecision(8) << std::fixed
                      << rep.reference[i] << '\n';

        std::cout << "   solution (simple iteration):\n";
        for (size_t i = 0; i < rep.simple.x.size(); ++i)
            std::cout << "   x[" << i << "] = " << std::setw(14)
                      << std::setprecision(8) << std::fixed
                      << rep.simple.x[i] << '\n';

        std::cout << "   solution (Seidel):\n";
        for (size_t i = 0; i < rep.seidel.x.size(); ++i)
            std::cout << "   x[" << i << "] = " << std::setw(14)
                      << std::setprecision(8) << std::fixed
                      << rep.seidel.x[i] << '\n';

        auto diff = [&](const Vector& v) {
            double d = 0.0;
            for (size_t i = 0; i < v.size(); ++i)
                d = std::max(d, std::fabs(v[i] - rep.reference[i]));
            return d;
        };
        std::cout << "   deviation simple = " << std::scientific
                  << std::setprecision(3) << diff(rep.simple.x) << '\n';
        std::cout << "   deviation Seidel = " << std::scientific
                  << std::setprecision(3) << diff(rep.seidel.x)
                  << std::defaultfloat << '\n';
    } else {
        std::cout << "   (no reference)\n";
        std::cout << "   solution (simple iteration):\n";
        for (size_t i = 0; i < rep.simple.x.size(); ++i)
            std::cout << "   x[" << i << "] = " << std::setw(14)
                      << std::setprecision(8) << std::fixed
                      << rep.simple.x[i] << '\n';
        std::cout << "   solution (Seidel):\n";
        for (size_t i = 0; i < rep.seidel.x.size(); ++i)
            std::cout << "   x[" << i << "] = " << std::setw(14)
                      << std::setprecision(8) << std::fixed
                      << rep.seidel.x[i] << '\n';
    }

    std::cout << "4) speed: ";
    if (rep.simple.converged && rep.seidel.converged &&
        rep.simple.iterations > 0 && rep.seidel.iterations > 0) {
        double ratio = double(rep.simple.iterations) / rep.seidel.iterations;
        if (ratio > 1.0)
            std::cout << "Seidel is faster by " << std::setprecision(3) << ratio << "x\n";
        else if (ratio < 1.0)
            std::cout << "simple is faster by " << std::setprecision(3) << (1.0 / ratio) << "x\n";
        else
            std::cout << "same\n";
    } else {
        std::cout << "cannot compare\n";
    }
    std::cout << '\n';
}

}