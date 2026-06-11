/* vsolver - sparse LDL^T factorization for quasidefinite systems.
 *
 * Davis-style up-looking LDL' (no pivoting). The KKT matrix is symmetric
 * quasidefinite (PD/ND diagonal blocks) and, with static regularization,
 * admits an LDL' factorization for any fill-reducing ordering without
 * numerical pivoting -- so the symbolic pattern is computed once and only the
 * numeric values change between interior-point iterations.
 *
 * Input matrix is the UPPER triangle in CSC (column j holds rows i <= j),
 * with every diagonal entry present.
 */
#include "internal.h"

/* Symbolic analysis: elimination tree + column counts of L.
 * Fills Lp[0..n], Parent[0..n-1], Lnz[0..n-1]; returns nnz(L). work is n. */
int vs_ldl_symbolic(int n, const int *Ap, const int *Ai,
                    int *Lp, int *Parent, int *Lnz, int *flag) {
    for (int k = 0; k < n; ++k) {
        Parent[k] = -1; flag[k] = k; Lnz[k] = 0;
        for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
            int i = Ai[p];
            if (i < k) {
                for (; flag[i] != k; i = Parent[i]) {
                    if (Parent[i] == -1) Parent[i] = k;
                    Lnz[i]++;
                    flag[i] = k;
                }
            }
        }
    }
    Lp[0] = 0;
    for (int k = 0; k < n; ++k) Lp[k + 1] = Lp[k] + Lnz[k];
    return Lp[n];
}

/* Numeric factorization A = L D L'. Returns n on success, else the index of
 * the first zero pivot. Lnz/flag/Y/Pattern are work arrays of length n. */
int vs_ldl_numeric(int n, const int *Ap, const int *Ai, const double *Ax,
                   const int *Lp, const int *Parent, int *Lnz,
                   int *Li, double *Lx, double *D, double *Dinv,
                   double *Y, int *Pattern, int *flag) {
    for (int k = 0; k < n; ++k) {
        Y[k] = 0.0; int top = n; flag[k] = k; Lnz[k] = 0;
        for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
            int i = Ai[p];
            if (i <= k) {
                Y[i] += Ax[p];
                int len = 0;
                for (; flag[i] != k; i = Parent[i]) { Pattern[len++] = i; flag[i] = k; }
                while (len > 0) Pattern[--top] = Pattern[--len];
            }
        }
        D[k] = Y[k]; Y[k] = 0.0;
        for (; top < n; ++top) {
            int i = Pattern[top];
            double yi = Y[i]; Y[i] = 0.0;
            int pend = Lp[i] + Lnz[i];
            for (int p = Lp[i]; p < pend; ++p) Y[Li[p]] -= Lx[p] * yi;
            double l_ki = yi * Dinv[i];
            D[k] -= l_ki * yi;
            Li[pend] = k; Lx[pend] = l_ki; Lnz[i]++;
        }
        if (D[k] == 0.0) return k;
        Dinv[k] = 1.0 / D[k];
    }
    return n;
}

/* Solve A x = b in place (b -> x): forward, diagonal, backward. */
void vs_ldl_solve(int n, const int *Lp, const int *Li, const double *Lx,
                  const double *Dinv, double *x) {
    for (int k = 0; k < n; ++k)
        for (int p = Lp[k]; p < Lp[k + 1]; ++p) x[Li[p]] -= Lx[p] * x[k];
    for (int k = 0; k < n; ++k) x[k] *= Dinv[k];
    for (int k = n - 1; k >= 0; --k)
        for (int p = Lp[k]; p < Lp[k + 1]; ++p) x[k] -= Lx[p] * x[Li[p]];
}
