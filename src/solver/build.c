/* vsolver - OSQP-style problem builder: l <= Ax <= u  ->  Ax + s = b, s >= 0 */
#include "internal.h"

static int is_finite_bound(const double *v, int i) {
    return v && fabs(v[i]) < VS_INF;
}

vs_problem *vs_build_osqp(const vs_csc *P, const double *q, const vs_csc *A,
                          const double *l, const double *u) {
    if (!A) return NULL;
    int mo = A->m, n = A->n;

    /* map each original row to its new (negated) rows; -1 if absent */
    int *row_u = vs_malloc_i(mo), *row_l = vs_malloc_i(mo);
    int mnew = 0;
    for (int i = 0; i < mo; ++i) {
        row_u[i] = is_finite_bound(u, i) ? mnew++ : -1;
        row_l[i] = is_finite_bound(l, i) ? mnew++ : -1;
    }

    /* triplets for the expanded constraint matrix */
    int nnz = A->p[n];
    int cap = 2 * nnz + 1;
    int *Ti = vs_malloc_i(cap), *Tj = vs_malloc_i(cap);
    double *Tx = vs_malloc_d(cap);
    int t = 0;
    for (int c = 0; c < n; ++c)
        for (int k = A->p[c]; k < A->p[c + 1]; ++k) {
            int i = A->i[k]; double v = A->x[k];
            if (row_u[i] >= 0) { Ti[t]=row_u[i]; Tj[t]=c; Tx[t]= v; ++t; }
            if (row_l[i] >= 0) { Ti[t]=row_l[i]; Tj[t]=c; Tx[t]=-v; ++t; }
        }
    vs_csc *Anew = vs_csc_from_triplets(mnew, n, t, Ti, Tj, Tx);
    free(Ti); free(Tj); free(Tx);

    double *b = vs_malloc_d(mnew);
    for (int i = 0; i < mo; ++i) {
        if (row_u[i] >= 0) b[row_u[i]] =  u[i];   /*  Ax <= u      */
        if (row_l[i] >= 0) b[row_l[i]] = -l[i];   /* -Ax <= -l     */
    }
    free(row_u); free(row_l);

    vs_cone *cone = (vs_cone *)malloc(sizeof(vs_cone));
    cone->type = VS_CONE_NONNEG; cone->dim = mnew;

    vs_problem *prob = (vs_problem *)malloc(sizeof(vs_problem));
    prob->n = n; prob->m = mnew;
    prob->P = (vs_csc *)P; prob->q = (double *)q;
    prob->A = Anew; prob->b = b;
    prob->cones = cone; prob->ncones = 1;
    return prob;
}

/* Frees builder-owned parts (A, b, cones, struct); leaves caller's P, q. */
void vs_problem_free(vs_problem *prob) {
    if (!prob) return;
    vs_csc_free(prob->A);
    free(prob->b);
    free(prob->cones);
    free(prob);
}
