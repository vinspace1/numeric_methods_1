#ifdef _WIN32
#  define NOMINMAX
#  include <windows.h>
#endif

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "lu.h"
#include "tridiagonal.h"
#include "iterative.h"
#include "jacobi.h"
#include "qr.h"

using namespace std;

static void clr() {
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

static string askFile(const string& prompt, const string& def) {
    cout << prompt << " [" << def << "]: ";
    string s;
    getline(cin, s);
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
    if (s.empty()) return def;
    return s;
}

static double askD(const string& p) {
    double v;
    while (true) {
        cout << p;
        if (cin >> v) { clr(); return v; }
        cout << "Try again: ";
        clr();
    }
}

static int askI(const string& p) {
    int v;
    while (true) {
        cout << p;
        if (cin >> v) { clr(); return v; }
        cout << "Try again: ";
        clr();
    }
}

static qr::Matrix loadMat(const string& file) {
    ifstream in(file);
    if (!in) throw runtime_error("Cannot open file: " + file);
    int n = 0;
    if (!(in >> n) || n <= 0) throw runtime_error("Invalid n");
    qr::Matrix A(n, qr::Vector(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!(in >> A[i][j])) throw runtime_error("Read error in A");
    return A;
}

static lu::PivotStrategy askPivot() {
    cout << "Pivot: 1-partial, 2-full, 3-none\n";
    int s = askI("Choice: ");
    if (s == 2) return lu::PivotStrategy::Full;
    if (s == 3) return lu::PivotStrategy::None;
    return lu::PivotStrategy::Partial;
}

static void doLU(const string& file, lu::PivotStrategy st) {
    lu::Task task;
    try { task = lu::loadTask(file); }
    catch (const exception& e) { cerr << "Load error: " << e.what() << '\n'; return; }

    lu::LUResult lu = lu::decompose(task.A, st);
    if (!lu.ok) { cerr << "LU error: " << lu.error << '\n'; return; }

    lu::printMatrix(lu.L, "a) L");
    lu::printMatrix(lu.U, "a) U");
    lu::printMatrix(lu.P, "   P");
    if (st == lu::PivotStrategy::Full) lu::printMatrix(lu.Q, "   Q");

    lu::Matrix LU = lu::multiply(lu.L, lu.U);
    lu::printMatrix(LU, "b) L*U");

    try {
        lu::Vector x = lu::solve(lu, task.b);
        lu::printVector(x, "c) x = A^(-1)*b");
        lu::Vector r = lu::multiply(task.A, x);
        cout << "   residual A*x - b:\n";
        for (int i = 0; i < task.n; ++i)
            cout << "     " << scientific << setprecision(3)
                 << (r[i] - task.b[i]) << '\n';
        cout << defaultfloat << '\n';
    } catch (const exception& e) {
        cerr << "Solve error: " << e.what() << '\n';
    }

    lu::Matrix inv;
    try {
        inv = lu::inverse(lu);
        lu::printMatrix(inv, "d) A^(-1)");
    } catch (const exception& e) {
        cerr << "Inverse error: " << e.what() << '\n';
        return;
    }

    lu::Matrix E = lu::multiply(task.A, inv);
    lu::printMatrix(E, "e) A*A^(-1)");
    lu::Matrix I(task.n, lu::Vector(task.n, 0.0));
    for (int i = 0; i < task.n; ++i) I[i][i] = 1.0;
    double dev = lu::maxAbsDiff(E, I);
    cout << "   A*A^(-1) == E ? " << (lu::isIdentity(E, 1e-6) ? "YES" : "NO")
         << "   (||E - A*A^(-1)||_max = " << scientific << setprecision(3)
         << dev << defaultfloat << ")\n\n";

    lu::Matrix PA = lu::multiply(lu.P, task.A);
    if (st == lu::PivotStrategy::Full) {
        lu::Matrix PAQ = lu::multiply(PA, lu.Q);
        cout << "f) L*U = P*A*Q check: ||L*U - P*A*Q||_max = "
             << scientific << setprecision(3) << lu::maxAbsDiff(LU, PAQ)
             << defaultfloat << '\n';
    } else {
        cout << "f) L*U = P*A check: ||L*U - P*A||_max = "
             << scientific << setprecision(3) << lu::maxAbsDiff(LU, PA)
             << defaultfloat << '\n';
    }
    cout << '\n';
}

static void doSweep(const string& file) {
    try {
        auto r = tri::SweepSolver::solveFromFile(file);
        tri::SweepSolver::printReport(r, cout);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
    }
}

static void doIter(const string& file) {
    lu::Task task;
    try { task = lu::loadTask(file); }
    catch (const exception& e) { cerr << "Load error: " << e.what() << '\n'; return; }

    double eps = askD("eps: ");

    lu::Matrix A = task.A;
    lu::Vector b = task.b;
    string why;
    if (!it::enforceDiagonalDominance(A, b, &why))
        cout << "Warning: " << why << '\n';

    it::ComparisonReport rep;
    rep.eps = eps;
    rep.simple = it::simpleIteration(A, b, eps);
    rep.seidel = it::seidel(A, b, eps);

    lu::LUResult lu = lu::decompose(task.A, lu::PivotStrategy::Partial);
    if (lu.ok) {
        try {
            rep.reference = lu::solve(lu, task.b);
            rep.hasReference = true;
        } catch (...) {}
    }

    it::printComparison(rep);
}

static void doJacobi(const string& file) {
    qr::Matrix A;
    try { A = loadMat(file); }
    catch (const exception& e) { cerr << "Load error: " << e.what() << '\n'; return; }
    if (!jacobi::isSymmetric(A, 1e-10))
        cout << "Warning: matrix is not symmetric, symmetrizing\n";
    double eps = askD("eps: ");
    auto r = jacobi::solve(A, eps);
    jacobi::printReport(A, r, eps);
}

static void doQR(const string& file) {
    qr::Matrix A;
    try { A = loadMat(file); }
    catch (const exception& e) { cerr << "Load error: " << e.what() << '\n'; return; }
    double eps = askD("eps: ");
    auto q = qr::decompose(A);
    auto e = qr::eigenvalues(A, eps);
    qr::printReport(A, q, e, eps);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    ios::sync_with_stdio(false);

    // Значения по умолчанию (можно переопределить аргументами командной строки)
    string luF     = (argc > 1) ? argv[1] : "data/lu/lu1.txt";
    string sweepF  = (argc > 2) ? argv[2] : "data/tri/tri1.txt";
    string iterF   = (argc > 3) ? argv[3] : "data/it/it1.txt";
    string jacobiF = (argc > 4) ? argv[4] : "data/jacobi/jacobi1.txt";
    string qrF     = (argc > 5) ? argv[5] : "data/qr/qr1.txt";

    while (true) {
        cout << "\n1-LU  2-Sweep  3-Iter  4-Jacobi  5-QR  0-exit\n> ";
        int c;
        if (!(cin >> c)) break;
        clr();

        if (c == 0) return 0;

        if (c == 1) {
            lu::PivotStrategy st = askPivot();
            luF = askFile("LU file", luF);
            doLU(luF, st);
            continue;
        }
        if (c == 2) {
            sweepF = askFile("Sweep file", sweepF);
            doSweep(sweepF);
            continue;
        }
        if (c == 3) {
            iterF = askFile("Iter file", iterF);
            doIter(iterF);
            continue;
        }
        if (c == 4) {
            jacobiF = askFile("Jacobi file", jacobiF);
            doJacobi(jacobiF);
            continue;
        }
        if (c == 5) {
            qrF = askFile("QR file", qrF);
            doQR(qrF);
            continue;
        }
        cout << "?\n";
    }
    return 0;
}