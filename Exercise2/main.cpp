#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

Vector2d solvePALU(Matrix2d A, Vector2d b) {
    Vector2d x = A.fullPivLu().solve(b);

    return x;
}

Vector2d solveQR(Matrix2d A, Vector2d b) {
    Vector2d x = A.fullPivHouseholderQr().solve(b);

    return x;
}

double calcRelativeErr(Vector2d sol, Vector2d x) {
    double relErr = (x-sol).norm() / sol.norm();
    //std::cout << "Relative error is: " << relErr << "\n";
    return relErr;
}

void solveBoth(Matrix2d A, Vector2d b, Vector2d sol, std::string name) {
    Vector2d xPALU = solvePALU(A, b);
    double errPALU = calcRelativeErr(xPALU, sol);

    Vector2d xQR = solveQR(A, b);
    double errQR = calcRelativeErr(xQR, sol);

    std::cout << "Solution for " << name << " with PALU decomposition is:\n" << xPALU << "\n";
    std::cout << "Relative error for " << name << " with PALU decomposition is: " << errPALU << "\n\n";
    std::cout << "Solution for " << name << " with QR decomposition is:\n" << xQR << "\n";
    std::cout << "Relative error for " << name << " with QR decomposition is: " << errQR << "\n\n";

}

int main()
{
    Vector2d sol;
    sol  << -1, -1;

    Matrix2d A1;
    Vector2d b1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,   8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01,   1.672384680188350e-01;

    Matrix2d A2;
    Vector2d b2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    Matrix2d A3;
    Vector2d b3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    solveBoth(A1, b1, sol, "System 1");
    solveBoth(A2, b2, sol, "System 2");
    solveBoth(A3, b3, sol, "System 3");

    return 0;
}
