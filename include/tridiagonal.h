#pragma once
#include <vector>
#include <string>
#include <istream>

namespace tri {

using Vector = std::vector<double>;

class SweepSolver {
public:
    struct Result {
        Vector alpha, beta, x;
        int n = 0;
        bool ok = false;
        std::string error;
        size_t maxDoublesInMemory = 0;
    };

    static Result solveFromStream(std::istream& in);
    static Result solveFromFile(const std::string& filename);
    static void printReport(const Result& r, std::ostream& os);
};

} 