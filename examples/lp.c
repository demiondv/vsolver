/* Example: a small linear program (the classic "diet" problem).
 *
 *   minimize    c' x                 (total cost)
 *   subject to  N x >= r             (meet nutrient minimums)
 *               x  >= 0
 *
 * cvxpy:
 *     x = cp.Variable(3)
 *     prob = cp.Problem(cp.Minimize(c @ x), [N @ x >= r, x >= 0])
 *     prob.solve()
 *
 * vsolver standard form  (A x + s = b, s >= 0):
 *   N x >= r   ->   -N x <= -r
 *   x   >= 0   ->     -x <= 0
 * stacked into one nonnegative cone.
 */
#include "vsolver.h"
#include <stdio.h>

int main(void) {
    /* 3 foods, 2 nutrients */
    const double c[3] = {2.0, 3.0, 1.0};            /* cost per unit  */
    /* nutrient matrix N (2x3) and minimum requirements r */
    /* N = [[1,2,1],[3,1,2]],  r = [10, 12] */

    /* standard-form A (5x3): rows = (-N ; -I),  b = (-r ; 0) */
    int    Ai[] = {0,0,0, 1,1,1, 2, 3, 4};
    int    Aj[] = {0,1,2, 0,1,2, 0, 1, 2};
    double Ax[] = {-1,-2,-1, -3,-1,-2, -1, -1, -1};
    double b[]  = {-10, -12, 0, 0, 0};
    vs_csc *A = vs_csc_from_triplets(5, 3, 9, Ai, Aj, Ax);

    vs_cone cone = { VS_CONE_NONNEG, 5 };
    vs_problem prob = { 3, 5, NULL, (double *)c, A, b, &cone, 1 };

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(&prob, &set, &sol);

    printf("LP (diet problem)\n");
    printf("  status : %s  (%d iters)\n", vs_status_str(sol.status), sol.iters);
    printf("  x      : %.4f  %.4f  %.4f\n", sol.x[0], sol.x[1], sol.x[2]);
    printf("  cost   : %.4f\n", sol.pobj);

    vs_solution_free(&sol);
    vs_csc_free(A);
    return 0;
}
