/* Example: a second-order cone program.
 *
 * Minimize a linear objective over a Euclidean ball:
 *
 *   minimize    c' x
 *   subject to  || x - x0 ||_2 <= r
 *
 * cvxpy:
 *     x = cp.Variable(3)
 *     prob = cp.Problem(cp.Minimize(c @ x),
 *                       [cp.norm(x - x0, 2) <= r])
 *     prob.solve()
 *
 * The constraint is the second-order cone  s = (r, x - x0) in SOC,
 * i.e.  s0 = r >= || s_1: || = || x - x0 ||.  In standard form A x + s = b:
 *   s0     = r            -> A row 0 = 0,  b0 = r
 *   s_{1:} = x - x0       -> A = -I,       b  = -x0
 * The analytic optimum is  x* = x0 - r c/||c||,  value c'x0 - r||c||.
 */
#include "vsolver.h"
#include <stdio.h>
#include <math.h>

#define N 3

int main(void) {
    const double c[N]  = {1.0, 1.0, 1.0};
    const double x0[N] = {1.0, 1.0, 1.0};
    const double r     = 2.0;

    /* A (4x3): row 0 zero; rows 1..3 = -I.  b = (r, -x0) */
    int    Ai[N], Aj[N]; double Ax[N];
    for (int i = 0; i < N; ++i) { Ai[i] = 1 + i; Aj[i] = i; Ax[i] = -1.0; }
    vs_csc *A = vs_csc_from_triplets(N + 1, N, N, Ai, Aj, Ax);
    double b[N + 1]; b[0] = r;
    for (int i = 0; i < N; ++i) b[1 + i] = -x0[i];

    vs_cone cone = { VS_CONE_SOC, N + 1 };
    vs_problem prob = { N, N + 1, NULL, (double *)c, A, b, &cone, 1 };

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(&prob, &set, &sol);

    double cn = sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
    double cx0 = c[0]*x0[0] + c[1]*x0[1] + c[2]*x0[2];

    printf("SOCP (minimize linear over a ball)\n");
    printf("  status   : %s  (%d iters)\n", vs_status_str(sol.status), sol.iters);
    printf("  x        : %.4f  %.4f  %.4f\n", sol.x[0], sol.x[1], sol.x[2]);
    printf("  c'x      : %.4f   (analytic %.4f)\n", sol.pobj, cx0 - r * cn);

    vs_solution_free(&sol);
    vs_csc_free(A);
    return 0;
}
