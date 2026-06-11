/* Cross-validation against cvxpy (CLARABEL) reference solutions.
 *
 * The problem data and reference primal/objective are generated into
 * cvxpy_refs.h by tests/gen_cvxpy_refs.py. Each case is solved with vsolver
 * and compared to the cvxpy reference (primal x and optimal objective). */
#include "vsolver.h"
#include "cvxpy_refs.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define TOL_X    1e-4
#define TOL_OBJ  1e-5

static int run_case(const cvxpy_case *c) {
    vs_csc *P = (c->Pnnz > 0)
        ? vs_csc_from_triplets(c->n, c->n, c->Pnnz, c->Pi, c->Pj, c->Px) : NULL;
    vs_csc *A = vs_csc_from_triplets(c->m, c->n, c->Annz, c->Ai, c->Aj, c->Ax);

    vs_cone *cones = malloc(sizeof(vs_cone) * c->ncones);
    for (int k = 0; k < c->ncones; ++k) {
        cones[k].type = (vs_cone_type)c->cone_type[k];
        cones[k].dim  = c->cone_dim[k];
    }
    vs_problem prob = { c->n, c->m, P, (double *)c->q, A, (double *)c->b,
                        cones, c->ncones };

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(&prob, &set, &sol);

    double xerr = 0.0;
    for (int k = 0; k < c->n; ++k)
        xerr = fmax(xerr, fabs(sol.x[k] - c->refx[k]));
    double oerr = fabs(sol.pobj - c->refobj) / (1.0 + fabs(c->refobj));

    int ok = (sol.status == VS_SOLVED) && xerr <= TOL_X && oerr <= TOL_OBJ;
    printf("  %-18s iters=%-2d  |dx|=%.2e  obj=% .6f (ref % .6f) derr=%.2e  %s\n",
           c->name, sol.iters, xerr, sol.pobj, c->refobj, oerr,
           ok ? "OK" : "FAIL");

    vs_solution_free(&sol);
    vs_csc_free(P); vs_csc_free(A); free(cones);
    return ok;
}

int main(void) {
    printf("vsolver vs cvxpy (CLARABEL), %d cases:\n", cvxpy_ncases);
    int ok = 1;
    for (int i = 0; i < cvxpy_ncases; ++i) ok &= run_case(&cvxpy_cases[i]);
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
    return ok ? 0 : 1;
}
