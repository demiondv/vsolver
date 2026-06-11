/* Small QP sanity checks with known closed-form optima. */
#include "vsolver.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int check(const char *name, double got, double want, double tol) {
    double e = fabs(got - want);
    printf("  %-22s got %+.6f  want %+.6f  err %.2e  %s\n",
           name, got, want, e, e <= tol ? "OK" : "FAIL");
    return e <= tol;
}

/* minimize 1/2 (x0^2 + x1^2) + q'x  s.t.  x <= u  (nonneg slacks)
 * Unconstrained optimum is x = -q; clip by the active upper bounds. */
static int run(double q0, double q1, double u0, double u1,
               double w0, double w1) {
    /* vars n=2; constraints x0<=u0, x1<=u1  -> A=I, b=u, nonneg cone */
    int Ti[2] = {0, 1}, Tj[2] = {0, 1};
    double Tx[2] = {1.0, 1.0};
    vs_csc *A = vs_csc_from_triplets(2, 2, 2, Ti, Tj, Tx);

    int Pi[2] = {0, 1}, Pj[2] = {0, 1};
    double Px[2] = {1.0, 1.0};
    vs_csc *P = vs_csc_from_triplets(2, 2, 2, Pi, Pj, Px);

    double q[2] = {q0, q1}, b[2] = {u0, u1};
    vs_cone cone = { VS_CONE_NONNEG, 2 };
    vs_problem prob = { 2, 2, P, q, A, b, &cone, 1 };

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(&prob, &set, &sol);

    int ok = 1;
    ok &= check("x0", sol.x[0], w0, 1e-6);
    ok &= check("x1", sol.x[1], w1, 1e-6);
    printf("  status=%s iters=%d pobj=%.6f\n",
           vs_status_str(sol.status), sol.iters, sol.pobj);

    vs_solution_free(&sol);
    vs_csc_free(A); vs_csc_free(P);
    return ok;
}

int main(void) {
    int ok = 1;
    printf("[unconstrained-ish] q=(-1,-2), u=(5,5) -> x=(1,2)\n");
    ok &= run(-1, -2, 5, 5, 1, 2);
    printf("[upper-bound active] q=(-3,-3), u=(0.5,2) -> x=(0.5,3->2)\n");
    ok &= run(-3, -3, 0.5, 2.0, 0.5, 2.0);
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
