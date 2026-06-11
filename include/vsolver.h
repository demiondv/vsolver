/* vsolver - a small conic interior-point optimizer (MOSEK-like, MIT-style)
 *
 * Solves the convex conic quadratic program
 *
 *     minimize     (1/2) x' P x + q' x
 *     subject to   A x + s = b
 *                  s in K
 *
 * where K is a Cartesian product of cones:
 *     - zero cone        { 0 }              (encodes equality A x = b)
 *     - nonnegative      { s >= 0 }         (encodes inequalities A x <= b)
 *     - second-order     { (t,u) : t >= ||u|| }
 *     - exponential      (3-dim, planned)
 *     - PSD              (vectorized symmetric matrix, planned)
 *
 * The solver is a primal-dual interior-point method with Nesterov-Todd
 * scaling for the symmetric cones and a Mehrotra predictor-corrector step.
 */
#ifndef VSOLVER_H
#define VSOLVER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----- sparse matrix, compressed sparse column (CSC) ----- */
typedef struct {
    int     m;     /* number of rows                       */
    int     n;     /* number of columns                    */
    int    *p;     /* column pointers, length n+1          */
    int    *i;     /* row indices,     length nnz = p[n]   */
    double *x;     /* numerical values, length nnz         */
} vs_csc;

/* ----- cone descriptor ----- */
typedef enum {
    VS_CONE_ZERO   = 0,  /* equality:  s == 0                       */
    VS_CONE_NONNEG = 1,  /* s >= 0 (componentwise)                  */
    VS_CONE_SOC    = 2,  /* second-order: s0 >= ||s1:||             */
    VS_CONE_EXP    = 3,  /* exponential cone (3-dim)  [planned]     */
    VS_CONE_PSD    = 4   /* semidefinite (svec of n x n) [planned]  */
} vs_cone_type;

typedef struct {
    vs_cone_type type;
    int          dim;   /* SOC: length of the cone block (>=1)
                         * ZERO/NONNEG: number of scalar entries
                         * EXP: must be 3
                         * PSD: side length n  (block occupies n*(n+1)/2) */
} vs_cone;

/* ----- problem ----- */
typedef struct {
    int      n;        /* number of primal variables x                 */
    int      m;        /* number of conic rows (== sum of cone dims)   */
    vs_csc  *P;        /* n x n, upper triangle, symmetric PSD; or NULL */
    double  *q;        /* length n  (linear objective)                 */
    vs_csc  *A;        /* m x n constraint matrix                      */
    double  *b;        /* length m                                     */
    vs_cone *cones;    /* array of cone blocks, in row order           */
    int      ncones;   /* number of cone blocks                        */
} vs_problem;

/* ----- settings ----- */
typedef struct {
    int    max_iter;      /* iteration cap                       */
    double eps_abs;       /* absolute tolerance on residuals/gap */
    double eps_rel;       /* relative tolerance                  */
    double reg_static;    /* static KKT regularization           */
    int    verbose;       /* 0 = silent, 1 = per-iteration log   */
} vs_settings;

/* ----- status codes ----- */
typedef enum {
    VS_SOLVED       =  0,
    VS_PRIMAL_INFEAS =  1,
    VS_DUAL_INFEAS   =  2,
    VS_MAX_ITER     =  3,
    VS_NUMERICAL    = -1,
    VS_ERROR        = -2
} vs_status;

/* ----- solution ----- */
typedef struct {
    vs_status status;
    double   *x;       /* length n   primal variables          */
    double   *s;       /* length m   primal slacks (s in K)    */
    double   *z;       /* length m   dual variables (z in K*)  */
    double    pobj;    /* primal objective                     */
    double    dobj;    /* dual objective                       */
    double    gap;     /* duality gap                          */
    double    pres;    /* primal residual norm                 */
    double    dres;    /* dual residual norm                   */
    int       iters;   /* iterations taken                     */
} vs_solution;

/* Fill `set` with library defaults. */
void vs_default_settings(vs_settings *set);

/* Solve `prob`. The solution arrays are allocated by the solver and must be
 * released with vs_solution_free. Returns the status (also in sol->status). */
vs_status vs_solve(const vs_problem *prob, const vs_settings *set,
                   vs_solution *sol);

void vs_solution_free(vs_solution *sol);

/* Human-readable status string. */
const char *vs_status_str(vs_status s);

/* ----- OSQP-style front-end ----- */
/* Bounds with magnitude >= VS_INF are treated as infinite (one-sided). */
#define VS_INF 1e30

/* Build a problem from the OSQP form
 *
 *     minimize     (1/2) x' P x + q' x
 *     subject to   l <= A x <= u
 *
 * Each finite bound becomes one nonnegative-cone row (Ax<=u, or -Ax<=-l).
 * The returned problem shares P and q with the caller (do not free them
 * before the problem) and owns its A, b and cones. Release it with
 * vs_problem_free. `l` or `u` may be NULL to mean -inf / +inf respectively. */
vs_problem *vs_build_osqp(const vs_csc *P, const double *q, const vs_csc *A,
                          const double *l, const double *u);
void vs_problem_free(vs_problem *prob);

/* ----- CSC helpers ----- */
vs_csc *vs_csc_alloc(int m, int n, int nnz);
void    vs_csc_free(vs_csc *M);
/* Build a CSC from triplet (i,j,x) entries; duplicates are summed. */
vs_csc *vs_csc_from_triplets(int m, int n, int nnz,
                             const int *Ti, const int *Tj, const double *Tx);

#ifdef __cplusplus
}
#endif
#endif /* VSOLVER_H */
