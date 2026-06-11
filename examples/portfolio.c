/* Example: Markowitz mean-variance portfolio optimization.
 *
 *   maximize    mu' w  -  gamma * w' Sigma w     (return minus risk)
 *   subject to  sum(w) = 1,   w >= 0             (fully invested, long only)
 *
 * cvxpy:
 *     w = cp.Variable(n)
 *     ret  = mu @ w
 *     risk = cp.quad_form(w, Sigma)
 *     prob = cp.Problem(cp.Maximize(ret - gamma*risk),
 *                       [cp.sum(w) == 1, w >= 0])
 *     prob.solve()
 *
 * vsolver minimizes 1/2 w'P w + q'w, so with P = 2*gamma*Sigma and q = -mu
 * the objective is  gamma*w'Sigma w - mu'w  (= negated maximand).
 * Constraints: sum(w)=1 -> zero cone;  w>=0 -> nonnegative cone.
 */
#include "vsolver.h"
#include <stdio.h>

#define N 4

int main(void) {
    const double mu[N]    = {0.12, 0.10, 0.07, 0.03};   /* expected returns */
    const double Sigma[N][N] = {                        /* covariance (SPD) */
        {0.10, 0.02, 0.01, 0.00},
        {0.02, 0.08, 0.01, 0.00},
        {0.01, 0.01, 0.05, 0.00},
        {0.00, 0.00, 0.00, 0.02},
    };
    const double gamma = 5.0;                           /* risk aversion */

    /* P = 2*gamma*Sigma, upper triangle */
    int    Pi[N*(N+1)/2], Pj[N*(N+1)/2]; double Px[N*(N+1)/2]; int pn = 0;
    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j)
            if (Sigma[i][j] != 0.0) {
                Pi[pn] = i; Pj[pn] = j; Px[pn] = 2.0 * gamma * Sigma[i][j]; ++pn;
            }
    vs_csc *P = vs_csc_from_triplets(N, N, pn, Pi, Pj, Px);

    double q[N]; for (int i = 0; i < N; ++i) q[i] = -mu[i];

    /* A: row 0 = ones (budget, zero cone); rows 1..N = -I (no short) */
    int    Ai[2*N], Aj[2*N]; double Ax[2*N]; int an = 0;
    for (int j = 0; j < N; ++j) { Ai[an]=0;   Aj[an]=j; Ax[an]= 1.0; ++an; }
    for (int j = 0; j < N; ++j) { Ai[an]=1+j; Aj[an]=j; Ax[an]=-1.0; ++an; }
    vs_csc *A = vs_csc_from_triplets(N + 1, N, an, Ai, Aj, Ax);
    double b[N + 1] = {1.0};   /* rest zero */

    vs_cone cones[2] = { {VS_CONE_ZERO, 1}, {VS_CONE_NONNEG, N} };
    vs_problem prob = { N, N + 1, P, q, A, b, cones, 2 };

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(&prob, &set, &sol);

    double ret = 0.0, risk = 0.0;
    for (int i = 0; i < N; ++i) {
        ret += mu[i] * sol.x[i];
        for (int j = 0; j < N; ++j) risk += sol.x[i] * Sigma[i][j] * sol.x[j];
    }

    printf("Markowitz portfolio (gamma=%.1f)\n", gamma);
    printf("  status  : %s  (%d iters)\n", vs_status_str(sol.status), sol.iters);
    printf("  weights : %.4f  %.4f  %.4f  %.4f  (sum %.4f)\n",
           sol.x[0], sol.x[1], sol.x[2], sol.x[3],
           sol.x[0]+sol.x[1]+sol.x[2]+sol.x[3]);
    printf("  return  : %.4f    risk(var): %.5f\n", ret, risk);

    vs_solution_free(&sol);
    vs_csc_free(P); vs_csc_free(A);
    return 0;
}
