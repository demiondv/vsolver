/* OSQP-style front-end: box-constrained QP
 *
 *     minimize  1/2 ||x - p||^2     subject to  lo <= x <= hi
 *
 * has the analytic solution x_i = clamp(p_i, lo_i, hi_i).  Built via
 * vs_build_osqp with A = I and two-sided bounds. */
#include "vsolver.h"
#include <stdio.h>
#include <math.h>

#define N 4

static double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int main(void) {
    double p[N]  = { 2.0, -1.5,  0.3,  5.0};
    double lo[N] = {-1.0, -1.0, -1.0,  1.0};
    double hi[N] = { 1.0,  1.0,  1.0,  2.0};

    int Pi[N], Pj[N]; double Px[N];
    int Ai[N], Aj[N]; double Ax[N];
    double q[N];
    for (int k = 0; k < N; ++k) {
        Pi[k]=k; Pj[k]=k; Px[k]=1.0;     /* P = I */
        Ai[k]=k; Aj[k]=k; Ax[k]=1.0;     /* A = I */
        q[k] = -p[k];                    /* q = -p */
    }
    vs_csc *P = vs_csc_from_triplets(N, N, N, Pi, Pj, Px);
    vs_csc *A = vs_csc_from_triplets(N, N, N, Ai, Aj, Ax);

    vs_problem *prob = vs_build_osqp(P, q, A, lo, hi);
    printf("expanded to m=%d nonneg rows from %d two-sided bounds\n", prob->m, N);

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(prob, &set, &sol);

    int ok = (sol.status == VS_SOLVED);
    printf("status=%s iters=%d\n", vs_status_str(sol.status), sol.iters);
    for (int k = 0; k < N; ++k) {
        double want = clampd(p[k], lo[k], hi[k]);
        double e = fabs(sol.x[k] - want);
        ok &= (e <= 1e-6);
        printf("  x[%d]=%+.6f  clamp=%+.6f  err=%.2e %s\n",
               k, sol.x[k], want, e, e <= 1e-6 ? "OK" : "FAIL");
    }
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");

    vs_solution_free(&sol);
    vs_problem_free(prob);
    vs_csc_free(P); vs_csc_free(A);
    return ok ? 0 : 1;
}
