/* Example: bounded least squares (uses the OSQP-style front-end).
 *
 *   minimize    || M x - d ||_2^2
 *   subject to  -1 <= x <= 1
 *
 * cvxpy:
 *     x = cp.Variable(3)
 *     prob = cp.Problem(cp.Minimize(cp.sum_squares(M @ x - d)),
 *                       [x >= -1, x <= 1])
 *     prob.solve()
 *
 * ||Mx-d||^2 = x'(M'M)x - 2 d'M x + d'd, so as 1/2 x'P x + q'x we use
 *   P = 2 M'M,   q = -2 M'd.
 * The box -1<=x<=1 is passed straight to vs_build_osqp as l<=A x<=u with A=I.
 */
#include "vsolver.h"
#include <stdio.h>
#include <math.h>

#define M_ROWS 4
#define N      3

int main(void) {
    const double M[M_ROWS][N] = {
        { 1.0,  0.0, -1.0},
        { 0.5,  1.0,  0.0},
        { 0.0,  2.0,  1.0},
        {-1.0,  0.0,  1.0},
    };
    const double d[M_ROWS] = {1.0, -2.0, 0.5, 3.0};

    /* P = 2 M'M (upper triangle), q = -2 M'd */
    double MtM[N][N] = {{0}}, Mtd[N] = {0};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            for (int r = 0; r < M_ROWS; ++r) MtM[i][j] += M[r][i] * M[r][j];
        for (int r = 0; r < M_ROWS; ++r) Mtd[i] += M[r][i] * d[r];
    }
    int    Pi[N*(N+1)/2], Pj[N*(N+1)/2]; double Px[N*(N+1)/2]; int pn = 0;
    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) {
            Pi[pn] = i; Pj[pn] = j; Px[pn] = 2.0 * MtM[i][j]; ++pn;
        }
    vs_csc *P = vs_csc_from_triplets(N, N, pn, Pi, Pj, Px);
    double q[N]; for (int i = 0; i < N; ++i) q[i] = -2.0 * Mtd[i];

    /* A = I, bounds -1 <= x <= 1 */
    int    Ai[N], Aj[N]; double Ax[N];
    double lo[N], hi[N];
    for (int i = 0; i < N; ++i) { Ai[i]=i; Aj[i]=i; Ax[i]=1.0; lo[i]=-1.0; hi[i]=1.0; }
    vs_csc *A = vs_csc_from_triplets(N, N, N, Ai, Aj, Ax);

    vs_problem *prob = vs_build_osqp(P, q, A, lo, hi);
    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(prob, &set, &sol);

    /* residual ||Mx-d|| */
    double res = 0.0;
    for (int r = 0; r < M_ROWS; ++r) {
        double e = -d[r];
        for (int j = 0; j < N; ++j) e += M[r][j] * sol.x[j];
        res += e * e;
    }

    printf("Bounded least squares\n");
    printf("  status   : %s  (%d iters)\n", vs_status_str(sol.status), sol.iters);
    printf("  x        : %.4f  %.4f  %.4f\n", sol.x[0], sol.x[1], sol.x[2]);
    printf("  ||Mx-d|| : %.4f\n", sqrt(res));

    vs_solution_free(&sol);
    vs_problem_free(prob);
    vs_csc_free(P); vs_csc_free(A);
    return 0;
}
