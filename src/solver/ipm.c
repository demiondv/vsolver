/* vsolver - primal-dual interior-point method (Mehrotra predictor-corrector)
 * over a product of symmetric cones (zero, nonnegative, second-order).
 *
 * Each iteration factors the symmetric quasidefinite KKT matrix
 *
 *     [ P + rI      A'      ] [dx]   [ -rx              ]
 *     [ A       -(W^2 + rI) ] [dz] = [ -rz - W (lam\rc) ]
 *
 * where W is the Nesterov-Todd scaling (W^2 = diag(s/z) on nonnegative rows,
 * a dense block per second-order cone), lam = Wz is the scaled point, and rc
 * is the complementarity target.  dx,dz recover ds = -rz - A dx uniformly.
 * Factored once per iteration with LAPACK dsytrf, reused for both solves.
 */
#include "internal.h"
#include <stdio.h>

/* When the iteration cannot make further progress (ill-conditioned scaling
 * near the boundary, tiny step), accept the current iterate if it already
 * meets a relaxed tolerance; otherwise report a numerical failure. */
static vs_status graceful(double pres, double dres, double mu, int deg) {
    const double t = 1e-7;
    if (pres <= t && dres <= t && (deg == 0 || mu <= t)) return VS_SOLVED;
    return VS_NUMERICAL;
}

vs_status vs_solve(const vs_problem *prob, const vs_settings *set,
                   vs_solution *sol) {
    const int n = prob->n, m = prob->m;
    const vs_csc *P = prob->P, *A = prob->A;
    const double *q = prob->q, *b = prob->b;

    vs_coneset cs;
    if (vs_coneset_init(&cs, prob->cones, prob->ncones) != 0) {
        sol->status = VS_ERROR; return VS_ERROR;
    }
    vs_kkt kk;
    if (vs_kkt_setup(&kk, P, A, &cs, set->reg_static) != 0) {
        vs_coneset_free(&cs); sol->status = VS_ERROR; return VS_ERROR;
    }

    const int N = n + m, deg = cs.degree;
    double *x  = vs_calloc_d(n);
    double *s  = vs_calloc_d(m), *z = vs_calloc_d(m);
    double *rx = vs_malloc_d(n),  *rz = vs_malloc_d(m);
    double *rhs = vs_malloc_d(N);
    double *evec = vs_calloc_d(m);
    double *lamlam = vs_malloc_d(m), *tgt = vs_malloc_d(m);
    double *dlam = vs_malloc_d(m),   *Wd = vs_malloc_d(m);
    double *dx = vs_malloc_d(n), *dz = vs_malloc_d(m), *ds = vs_malloc_d(m);
    double *dx_a = vs_malloc_d(n), *dz_a = vs_malloc_d(m), *ds_a = vs_malloc_d(m);
    double *t1 = vs_malloc_d(m), *t2 = vs_malloc_d(m), *cross = vs_malloc_d(m);

    vs_cone_identity(&cs, s, z);           /* start at s = z = e */
    vs_cone_identity(&cs, evec, dz);       /* evec = e (reuse dz as scratch) */

    vs_status st = VS_MAX_ITER;
    int it = 0;
    double mu = 0.0, pres = 0.0, dres = 0.0;

    for (it = 0; it < set->max_iter; ++it) {
        vs_copy(q, rx, n);                 /* rx = Px + q + A'z */
        vs_csc_symv(P, 1.0, x, rx);
        vs_csc_gemtv(A, 1.0, z, rx);
        vs_copy(s, rz, m);                 /* rz = Ax + s - b */
        vs_axpy(-1.0, b, rz, m);
        vs_csc_gemv(A, 1.0, x, rz);

        double sz = vs_dot(s, z, m);
        mu = (deg > 0) ? sz / deg : 0.0;
        pres = vs_nrminf(rz, m);
        dres = vs_nrminf(rx, n);

        if (set->verbose)
            fprintf(stderr, "it %2d  pres %.3e  dres %.3e  mu %.3e\n",
                    it, pres, dres, mu);

        if (pres <= set->eps_abs && dres <= set->eps_abs &&
            (deg == 0 || mu <= set->eps_abs)) { st = VS_SOLVED; break; }

        if (vs_cone_nt(&cs, s, z) != 0)  { st = graceful(pres, dres, mu, deg); break; }
        if (vs_kkt_factor(&kk, &cs) != 0) { st = graceful(pres, dres, mu, deg); break; }

        vs_cone_prod(&cs, cs.lam, cs.lam, lamlam);

        /* ---- affine: target = -lam o lam ---- */
        for (int r = 0; r < m; ++r) tgt[r] = -lamlam[r];
        vs_cone_div(&cs, cs.lam, tgt, dlam);
        vs_cone_W(&cs, dlam, Wd);
        for (int i = 0; i < n; ++i) rhs[i] = -rx[i];
        for (int r = 0; r < m; ++r) rhs[n + r] = -rz[r] - Wd[r];
        vs_kkt_solve(&kk, rhs);
        vs_copy(rhs, dx_a, n);
        for (int r = 0; r < m; ++r) dz_a[r] = rhs[n + r];
        for (int r = 0; r < m; ++r) ds_a[r] = -rz[r];
        vs_csc_gemv(A, -1.0, dx_a, ds_a);

        double a_p = vs_cone_max_step(&cs, s, ds_a);
        double a_d = vs_cone_max_step(&cs, z, dz_a);
        double a_aff = VS_MIN(a_p, a_d); if (a_aff > 1.0) a_aff = 1.0;

        double sz_aff = 0.0;
        for (int r = 0; r < m; ++r)
            sz_aff += (s[r] + a_aff * ds_a[r]) * (z[r] + a_aff * dz_a[r]);
        double mu_aff = (deg > 0) ? sz_aff / deg : 0.0;
        double sigma = (mu > 0.0) ? pow(mu_aff / mu, 3.0) : 0.0;

        /* ---- corrector: target = sigma*mu*e - lam o lam - (W^-1 ds_a)o(W dz_a) */
        vs_cone_Winv(&cs, ds_a, t1);
        vs_cone_W(&cs, dz_a, t2);
        vs_cone_prod(&cs, t1, t2, cross);
        for (int r = 0; r < m; ++r)
            tgt[r] = sigma * mu * evec[r] - lamlam[r] - cross[r];
        vs_cone_div(&cs, cs.lam, tgt, dlam);
        vs_cone_W(&cs, dlam, Wd);
        for (int i = 0; i < n; ++i) rhs[i] = -rx[i];
        for (int r = 0; r < m; ++r) rhs[n + r] = -rz[r] - Wd[r];
        vs_kkt_solve(&kk, rhs);
        vs_copy(rhs, dx, n);
        for (int r = 0; r < m; ++r) dz[r] = rhs[n + r];
        for (int r = 0; r < m; ++r) ds[r] = -rz[r];
        vs_csc_gemv(A, -1.0, dx, ds);

        double sp = vs_cone_max_step(&cs, s, ds);
        double sd = vs_cone_max_step(&cs, z, dz);
        double alpha = 0.99 * VS_MIN(sp, sd);
        if (alpha > 1.0) alpha = 1.0;
        if (alpha <= 0.0) { st = graceful(pres, dres, mu, deg); break; }

        vs_axpy(alpha, dx, x, n);
        vs_axpy(alpha, ds, s, m);
        vs_axpy(alpha, dz, z, m);
    }

    double pobj = vs_dot(q, x, n);
    if (P) {
        double *Px = vs_calloc_d(n);
        vs_csc_symv(P, 1.0, x, Px);
        pobj += 0.5 * vs_dot(x, Px, n);
        free(Px);
    }

    sol->status = st;
    sol->x = x; sol->s = s; sol->z = z;
    sol->pobj = pobj; sol->dobj = pobj - mu * deg;
    sol->gap = mu * deg; sol->pres = pres; sol->dres = dres; sol->iters = it;

    free(rx); free(rz); free(rhs); free(evec);
    free(lamlam); free(tgt); free(dlam); free(Wd);
    free(dx); free(dz); free(ds); free(dx_a); free(dz_a); free(ds_a);
    free(t1); free(t2); free(cross);
    vs_kkt_free(&kk);
    vs_coneset_free(&cs);
    return st;
}
