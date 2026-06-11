/* vsolver internal shared declarations */
#ifndef VS_INTERNAL_H
#define VS_INTERNAL_H

#include "vsolver.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ----- small allocation helpers (zeroing) ----- */
static inline double *vs_calloc_d(size_t n) {
    return (double *)calloc(n ? n : 1, sizeof(double));
}
static inline int *vs_calloc_i(size_t n) {
    return (int *)calloc(n ? n : 1, sizeof(int));
}
static inline double *vs_malloc_d(size_t n) {
    return (double *)malloc((n ? n : 1) * sizeof(double));
}
static inline int *vs_malloc_i(size_t n) {
    return (int *)malloc((n ? n : 1) * sizeof(int));
}

#define VS_MAX(a, b) ((a) > (b) ? (a) : (b))
#define VS_MIN(a, b) ((a) < (b) ? (a) : (b))

/* ===================== symmetric-cone algebra ===================== */
/* A coneset wraps the product cone K and holds the current Nesterov-Todd
 * scaling. The interior-point loop talks to cones only through these calls,
 * so adding a cone type is localized to cone.c. */
typedef struct {
    const vs_cone *cones;
    int   ncones;
    int   m;          /* total rows                         */
    int  *off;        /* row offset of each cone, length ncones */
    int   degree;     /* sum of cone ranks (for mu)         */

    /* scaling, refreshed by vs_cone_nt() */
    double  *w2diag;  /* length m: nonneg rows hold s/z (else 0)  */
    double   *eta;    /* length ncones: SOC scale (else 0)        */
    double  **wbar;   /* length ncones: SOC w-bar vector (else 0) */
    double  **W2;     /* length ncones: SOC dense d*d -W^2 source */
    double   *lam;    /* length m: scaled point lambda = W z      */
} vs_coneset;

int  vs_coneset_init(vs_coneset *cs, const vs_cone *cones, int ncones);
void vs_coneset_free(vs_coneset *cs);

/* s = z = e (the cone identity); x untouched */
void vs_cone_identity(const vs_coneset *cs, double *s, double *z);
/* out = a o b  (Jordan product) */
void vs_cone_prod(const vs_coneset *cs, const double *a, const double *b,
                  double *out);
/* solve lam o out = y  (cone division) */
void vs_cone_div(const vs_coneset *cs, const double *lam, const double *y,
                 double *out);
/* refresh NT scaling from (s,z); also stores lam = W z in cs->lam */
int  vs_cone_nt(vs_coneset *cs, const double *s, const double *z);
/* out = W x  /  out = W^{-1} x   (block scaling) */
void vs_cone_W(const vs_coneset *cs, const double *x, double *out);
void vs_cone_Winv(const vs_coneset *cs, const double *x, double *out);
/* largest alpha >= 0 with v + alpha*dv in K (INFINITY if unbounded) */
double vs_cone_max_step(const vs_coneset *cs, const double *v, const double *dv);

/* ===================== sparse KKT (kkt.c) ===================== */
/* Assembles and factors the symmetric quasidefinite KKT matrix
 *     [ P+rI    A'      ;   A   -(W^2+rI) ]
 * in upper-triangular CSC. The P+reg and A' blocks are constant, set once at
 * setup; only the -(W^2+rI) block is rewritten each iteration. */
typedef struct {
    int N, n, m;
    int *Kp, *Ki; double *Kx; int Knnz;     /* upper-tri KKT, original order */
    double reg;
    int   *slotW;    /* length m: Kx slot of nonneg/zero diagonal (-1 if SOC) */
    int  **slotSOC;  /* length ncones: slots of SOC upper block, col-major */
    /* fill-reducing permutation + permuted matrix */
    int *perm, *iperm;          /* perm[newpos]=old, iperm[old]=newpos */
    double *ptmp;               /* length N solve scratch */
    int *Kp2, *Ki2; double *Kx2; int *kmap;  /* permuted KKT + orig->perm map */
    /* LDL factors + work (on the permuted matrix) */
    int *Lp, *Li; double *Lx; int Lnnz;
    int *Parent, *Lnz, *flag, *Pattern; double *D, *Dinv, *Y;
} vs_kkt;

int  vs_kkt_setup(vs_kkt *kk, const vs_csc *P, const vs_csc *A,
                  const vs_coneset *cs, double reg);
void vs_kkt_free(vs_kkt *kk);
/* refresh the W^2 block from the current scaling and factor; 0 on success */
int  vs_kkt_factor(vs_kkt *kk, const vs_coneset *cs);
void vs_kkt_solve(vs_kkt *kk, double *rhs);   /* in place */

/* ----- dense vector BLAS-1 (length n) ----- */
double vs_dot(const double *a, const double *b, int n);
double vs_nrm2(const double *a, int n);
double vs_nrminf(const double *a, int n);
void   vs_axpy(double alpha, const double *x, double *y, int n); /* y += a*x   */
void   vs_scal(double alpha, double *x, int n);                  /* x *= a     */
void   vs_copy(const double *x, double *y, int n);               /* y  = x     */

/* ----- sparse matrix-vector (CSC) ----- */
/* y += alpha * A * x   (A is m x n) */
void vs_csc_gemv(const vs_csc *A, double alpha, const double *x, double *y);
/* y += alpha * A' * x  (A is m x n, so A' is n x m) */
void vs_csc_gemtv(const vs_csc *A, double alpha, const double *x, double *y);
/* y += alpha * S * x  where S is symmetric, stored as upper triangle (i<=j) */
void vs_csc_symv(const vs_csc *S, double alpha, const double *x, double *y);
/* return A transposed (new CSC), or NULL on alloc failure */
vs_csc *vs_csc_transpose(const vs_csc *A);

/* ===================== sparse LDL^T (ldl.c) ===================== */
int  vs_ldl_symbolic(int n, const int *Ap, const int *Ai,
                     int *Lp, int *Parent, int *Lnz, int *flag);
int  vs_ldl_numeric(int n, const int *Ap, const int *Ai, const double *Ax,
                    const int *Lp, const int *Parent, int *Lnz,
                    int *Li, double *Lx, double *D, double *Dinv,
                    double *Y, int *Pattern, int *flag);
void vs_ldl_solve(int n, const int *Lp, const int *Li, const double *Lx,
                  const double *Dinv, double *x);

#endif /* VS_INTERNAL_H */
