/* Scaling benchmark: a large sparse box-constrained QP
 *
 *     minimize  1/2 x'P x + q'x      subject to  lo <= x <= hi
 *
 * with P tridiagonal (2 on the diagonal, -1 off) -- the kind of sparse system
 * where a dense (n+m)^3 KKT factorization is hopeless but sparse LDL is cheap.
 * Reports solve time and the per-iteration KKT fill (nnz of L). */
#include "vsolver.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { return (double)clock() / CLOCKS_PER_SEC; }

static void run(int n) {
    /* P = tridiag(-1,2,-1), upper triangle */
    int cap = 2 * n;
    int *Pi = malloc(cap * sizeof(int)), *Pj = malloc(cap * sizeof(int));
    double *Px = malloc(cap * sizeof(double));
    int pn = 0;
    for (int j = 0; j < n; ++j) {
        if (j > 0) { Pi[pn] = j - 1; Pj[pn] = j; Px[pn] = -1.0; ++pn; }
        Pi[pn] = j; Pj[pn] = j; Px[pn] = 2.0; ++pn;
    }
    vs_csc *P = vs_csc_from_triplets(n, n, pn, Pi, Pj, Px);

    /* A = I, box bounds */
    int *Ai = malloc(n * sizeof(int)), *Aj = malloc(n * sizeof(int));
    double *Ax = malloc(n * sizeof(double));
    double *q = malloc(n * sizeof(double));
    double *lo = malloc(n * sizeof(double)), *hi = malloc(n * sizeof(double));
    for (int k = 0; k < n; ++k) {
        Ai[k] = k; Aj[k] = k; Ax[k] = 1.0;
        q[k] = ((k % 7) - 3) * 0.5;     /* deterministic, varied */
        lo[k] = -1.0; hi[k] = 1.0;
    }
    vs_csc *A = vs_csc_from_triplets(n, n, n, Ai, Aj, Ax);
    vs_problem *prob = vs_build_osqp(P, q, A, lo, hi);

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    double t0 = now();
    vs_solve(prob, &set, &sol);
    double dt = now() - t0;

    printf("n=%-6d  m=%-6d  status=%-7s  iters=%-2d  solve=%7.1f ms  pobj=%.4e\n",
           n, prob->m, vs_status_str(sol.status), sol.iters, dt * 1e3, sol.pobj);
    fflush(stdout);

    vs_solution_free(&sol);
    vs_problem_free(prob);
    vs_csc_free(P); vs_csc_free(A);
    free(Pi); free(Pj); free(Px); free(Ai); free(Aj); free(Ax);
    free(q); free(lo); free(hi);
}

int main(void) {
    printf("sparse QP scaling (tridiagonal P, box constraints):\n");
    int sizes[] = {1000, 10000, 50000, 200000, 500000};
    for (int i = 0; i < 5; ++i) run(sizes[i]);
    return 0;
}
