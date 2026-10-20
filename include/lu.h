#pragma once
#include <vector>
#include <string>
#include <iostream>

namespace lu {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

enum class PivotStrategy { None, Partial, Full };

struct LUResult {
    Matrix L, U, P, Q;
    std::vector<int> rowPerm, colPerm;
    PivotStrategy strategy = PivotStrategy::Partial;
    bool ok = false;
    std::string error;
};

LUResult decompose(const Matrix& A,
                   PivotStrategy strategy = PivotStrategy::Partial);

Matrix multiply(const Matrix& A, const Matrix& B);
Vector multiply(const Matrix& A, const Vector& b);

Vector solve(const LUResult& lu, const Vector& b);
Matrix inverse(const LUResult& lu);

bool isIdentity(const Matrix& M, double eps = 1e-9);
double maxAbsDiff(const Matrix& A, const Matrix& B);

void printMatrix(const Matrix& M, const std::string& name);
void printVector(const Vector& v, const std::string& name);

struct Task {
    int n = 0;
    Matrix A;
    Vector b;
};
Task loadTask(const std::string& filename);

} 