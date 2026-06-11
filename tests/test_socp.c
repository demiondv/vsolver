/* SOCP validation: Euclidean projection onto the second-order cone has a
 * closed form, so we solve
 *
 *     minimize  1/2 ||x - p||^2     subject to   x in SOC_3
 *
 * as  min 1/2 x'x - p'x  s.t.  A x + s = 0, s = x in SOC  (A = -I, b = 0)
 * and compare x to the analytic projection of p.
 */
#include "vsolver.h"
#include <stdio.h>
#include <math.h>

#define D 3

/* projection of p=(p0,p1..) onto { x0 >= ||x1:|| } */
static void proj_soc(const double *p, double *out) {
    double nrm = 0.0;
    for (int k = 1; k < D; ++k) nrm += p[k] * p[k];
    nrm = sqrt(nrm);
    if (p[0] >= nrm) {                       /* inside */
        for (int k = 0; k < D; ++k) out[k] = p[k];
    } else if (p[0] <= -nrm) {               /* polar -> 0 */
        for (int k = 0; k < D; ++k) out[k] = 0.0;
    } else {                                 /* blend onto boundary */
        double a = (nrm + p[0]) / 2.0;
        out[0] = a;
        for (int k = 1; k < D; ++k) out[k] = a * p[k] / nrm;
    }
}

static int run(const double *p, const char *name) {
    int Pi[D], Pj[D]; double Px[D];
    int Ai[D], Aj[D]; double Ax[D];
    for (int k = 0; k < D; ++k) {
        Pi[k] = k; Pj[k] = k; Px[k] = 1.0;     /* P = I */
        Ai[k] = k; Aj[k] = k; Ax[k] = -1.0;    /* A = -I */
    }
    vs_csc *P = vs_csc_from_triplets(D, D, D, Pi, Pj, Px);
    vs_csc *A = vs_csc_from_triplets(D, D, D, Ai, Aj, Ax);
    double q[D]; for (int k = 0; k < D; ++k) q[k] = -p[k];   /* q = -p */
    double b[D] = {0, 0, 0};
    vs_cone cone = { VS_CONE_SOC, D };
    vs_problem prob = { D, D, P, q, A, b, &cone, 1 };

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(&prob, &set, &sol);

    double want[D]; proj_soc(p, want);
    double err = 0.0;
    for (int k = 0; k < D; ++k) err = fmax(err, fabs(sol.x[k] - want[k]));

    printf("%-10s p=(%.1f,%.1f,%.1f)  x=(%+.5f,%+.5f,%+.5f)  proj=(%+.5f,%+.5f,%+.5f)  err=%.2e  %s\n",
           name, p[0], p[1], p[2], sol.x[0], sol.x[1], sol.x[2],
           want[0], want[1], want[2], err,
           (sol.status == VS_SOLVED && err <= 1e-6) ? "OK" : "FAIL");

    int ok = (sol.status == VS_SOLVED && err <= 1e-6);
    vs_solution_free(&sol);
    vs_csc_free(P); vs_csc_free(A);
    return ok;
}

int main(void) {
    int ok = 1;
    double a[D] = { 3.0, 1.0, 0.0};   /* interior  -> p          */
    double b[D] = { 1.0, 2.0, 0.0};   /* blend     -> (1.5,1.5,0) */
    double c[D] = {-3.0, 1.0, 0.0};   /* polar     -> 0          */
    double e[D] = { 0.5,-1.2, 0.9};   /* blend, 3D               */
    ok &= run(a, "interior");
    ok &= run(b, "blend");
    ok &= run(c, "zero");
    ok &= run(e, "blend3d");
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
