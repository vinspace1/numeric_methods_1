#include "tridiagonal.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace tri {

static double readNum(std::istream& in, const char* what) {
    double v;
    if (!(in >> v))
        throw std::runtime_error(std::string("Read error: ") + what);
    return v;
}

SweepSolver::Result SweepSolver::solveFromStream(std::istream& in) {
    Result r;
    int n = 0;
    if (!(in >> n) || n <= 0) throw std::runtime_error("Invalid n");
    r.n = n;

    r.alpha.assign(n, 0.0);
    r.beta.assign(n, 0.0);

    double a, b, c, d;
    if (!(in >> a >> b >> c >> d))
        throw std::runtime_error("Read error at row 0");
    if (std::fabs(b) < 1e-15) { r.error = "b[0]=0"; return r; }
    r.beta[0] = d / b;
    if (n > 1) r.alpha[1] = -c / b;

    for (int i = 1; i < n; ++i) {
        if (!(in >> a >> b >> c >> d))
            throw std::runtime_error("Read error at row " + std::to_string(i));
        double denom = b + a * r.alpha[i];
        if (std::fabs(denom) < 1e-15) {
            r.error = "Zero denominator at step " + std::to_string(i);
            return r;
        }
        r.beta[i] = (d - a * r.beta[i - 1]) / denom;
        if (i < n - 1) r.alpha[i + 1] = -c / denom;
    }
    r.maxDoublesInMemory = static_cast<size_t>(3 * n);

    r.x.assign(n, 0.0);
    r.x[n - 1] = r.beta[n - 1];
    for (int i = n - 2; i >= 0; --i)
        r.x[i] = r.alpha[i + 1] * r.x[i + 1] + r.beta[i];

    r.ok = true;
    return r;
}

SweepSolver::Result SweepSolver::solveFromFile(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);
    return solveFromStream(in);
}

void SweepSolver::printReport(const Result& r, std::ostream& os) {
    if (!r.ok) { os << "Error: " << r.error << '\n'; return; }

    os << "a) Sweep coefficients:\n";
    os << std::setw(6) << "i"
       << std::setw(18) << "alpha[i]"
       << std::setw(18) << "beta[i]" << '\n';
    for (int i = 0; i < r.n; ++i) {
        os << std::setw(6) << i;
        if (i == 0)
            os << std::setw(18) << "-";
        else
            os << std::setw(18) << std::setprecision(8) << std::fixed << r.alpha[i];
        os << std::setw(18) << std::setprecision(8) << std::fixed << r.beta[i] << '\n';
    }
    os << '\n';

    os << "b) Solution:\n";
    os << std::setw(6) << "i" << std::setw(18) << "x[i]" << '\n';
    for (int i = 0; i < r.n; ++i)
        os << std::setw(6) << i
           << std::setw(18) << std::setprecision(8) << std::fixed << r.x[i] << '\n';
    os << '\n';

    os << "Memory: " << r.maxDoublesInMemory << " double (O(n)), "
       << "dense would be " << (r.n * r.n) << " (O(n^2))\n\n";
}

}