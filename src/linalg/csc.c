/* vsolver - sparse CSC matrices and dense vector primitives */
#include "internal.h"

/* ===================== dense BLAS-1 ===================== */
double vs_dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s += a[k] * b[k];
    return s;
}
double vs_nrm2(const double *a, int n) { return sqrt(vs_dot(a, a, n)); }

double vs_nrminf(const double *a, int n) {
    double m = 0.0;
    for (int k = 0; k < n; ++k) {
        double v = fabs(a[k]);
        if (v > m) m = v;
    }
    return m;
}
void vs_axpy(double alpha, const double *x, double *y, int n) {
    for (int k = 0; k < n; ++k) y[k] += alpha * x[k];
}
void vs_scal(double alpha, double *x, int n) {
    for (int k = 0; k < n; ++k) x[k] *= alpha;
}
void vs_copy(const double *x, double *y, int n) {
    memcpy(y, x, (size_t)n * sizeof(double));
}

/* ===================== CSC matrix ===================== */
vs_csc *vs_csc_alloc(int m, int n, int nnz) {
    vs_csc *M = (vs_csc *)malloc(sizeof(vs_csc));
    if (!M) return NULL;
    M->m = m;
    M->n = n;
    M->p = vs_calloc_i((size_t)n + 1);
    M->i = vs_malloc_i((size_t)nnz);
    M->x = vs_malloc_d((size_t)nnz);
    if (!M->p || !M->i || !M->x) {
        vs_csc_free(M);
        return NULL;
    }
    return M;
}

void vs_csc_free(vs_csc *M) {
    if (!M) return;
    free(M->p);
    free(M->i);
    free(M->x);
    free(M);
}

/* Triplet -> CSC, summing duplicates. Classic two-pass counting sort. */
vs_csc *vs_csc_from_triplets(int m, int n, int nnz,
                             const int *Ti, const int *Tj, const double *Tx) {
    vs_csc *M = vs_csc_alloc(m, n, nnz);
    if (!M) return NULL;

    int *colcnt = vs_calloc_i((size_t)n + 1);
    if (!colcnt) { vs_csc_free(M); return NULL; }

    for (int k = 0; k < nnz; ++k) colcnt[Tj[k]]++;
    for (int j = 0, cum = 0; j <= n; ++j) {
        int c = (j < n) ? colcnt[j] : 0;
        M->p[j] = cum;
        cum += c;
    }
    /* scatter using a working copy of the column heads */
    int *head = vs_malloc_i((size_t)n);
    if (!head) { free(colcnt); vs_csc_free(M); return NULL; }
    for (int j = 0; j < n; ++j) head[j] = M->p[j];

    for (int k = 0; k < nnz; ++k) {
        int j = Tj[k];
        int dst = head[j]++;
        M->i[dst] = Ti[k];
        M->x[dst] = Tx[k];
    }
    free(head);
    free(colcnt);

    /* merge duplicates within each column (stable, in place) */
    int *seen = vs_malloc_i((size_t)m);
    if (!seen) { vs_csc_free(M); return NULL; }
    for (int r = 0; r < m; ++r) seen[r] = -1;

    int w = 0;
    int *newp = vs_calloc_i((size_t)n + 1);
    if (!newp) { free(seen); vs_csc_free(M); return NULL; }
    for (int j = 0; j < n; ++j) {
        int start = w;
        for (int k = M->p[j]; k < M->p[j + 1]; ++k) {
            int r = M->i[k];
            if (seen[r] >= start) {        /* duplicate within this column */
                M->x[seen[r]] += M->x[k];
            } else {
                seen[r] = w;
                M->i[w] = r;
                M->x[w] = M->x[k];
                ++w;
            }
        }
        newp[j + 1] = w;
    }
    memcpy(M->p, newp, (size_t)(n + 1) * sizeof(int));
    free(newp);
    free(seen);
    return M;
}

/* A (m x n) -> A' (n x m), counting sort by row of A. */
vs_csc *vs_csc_transpose(const vs_csc *A) {
    int m = A->m, n = A->n, nnz = A->p[n];
    vs_csc *T = vs_csc_alloc(n, m, nnz);
    if (!T) return NULL;
    int *cnt = vs_calloc_i((size_t)m + 1);
    if (!cnt) { vs_csc_free(T); return NULL; }
    for (int k = 0; k < nnz; ++k) cnt[A->i[k]]++;
    for (int r = 0, cum = 0; r <= m; ++r) { int c = (r < m) ? cnt[r] : 0; T->p[r] = cum; cum += c; }
    int *head = vs_malloc_i((size_t)m);
    if (!head) { free(cnt); vs_csc_free(T); return NULL; }
    for (int r = 0; r < m; ++r) head[r] = T->p[r];
    for (int j = 0; j < n; ++j)
        for (int k = A->p[j]; k < A->p[j + 1]; ++k) {
            int r = A->i[k], dst = head[r]++;
            T->i[dst] = j;            /* column of A becomes row of T */
            T->x[dst] = A->x[k];
        }
    free(head); free(cnt);
    return T;
}

/* ===================== sparse mat-vec ===================== */
void vs_csc_gemv(const vs_csc *A, double alpha, const double *x, double *y) {
    if (!A) return;
    for (int j = 0; j < A->n; ++j) {
        double xj = alpha * x[j];
        if (xj == 0.0) continue;
        for (int k = A->p[j]; k < A->p[j + 1]; ++k)
            y[A->i[k]] += A->x[k] * xj;
    }
}

void vs_csc_gemtv(const vs_csc *A, double alpha, const double *x, double *y) {
    if (!A) return;
    for (int j = 0; j < A->n; ++j) {
        double s = 0.0;
        for (int k = A->p[j]; k < A->p[j + 1]; ++k)
            s += A->x[k] * x[A->i[k]];
        y[j] += alpha * s;
    }
}

/* y += alpha * S * x, S symmetric with only the upper triangle (i<=j) stored. */
void vs_csc_symv(const vs_csc *S, double alpha, const double *x, double *y) {
    if (!S) return;
    for (int j = 0; j < S->n; ++j) {
        for (int k = S->p[j]; k < S->p[j + 1]; ++k) {
            int i = S->i[k];
            double v = alpha * S->x[k];
            y[i] += v * x[j];           /* upper entry (i,j) */
            if (i != j) y[j] += v * x[i]; /* mirror (j,i)    */
        }
    }
}
