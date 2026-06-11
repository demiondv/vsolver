/* vsolver - sparse KKT assembly, fill-reducing reordering, factor and solve.
 *
 * Upper-triangular CSC layout (columns 0..n-1 are variables, n..n+m-1 conic):
 *   var col j      : P column j (rows i<=j) with reg on the diagonal  [constant]
 *   conic col n+r  : A'(:,r) entries (rows < n)                       [constant]
 *                    + the -(W^2+reg) block (diagonal or SOC upper)   [updated]
 *
 * Only the W-block slots change between iterations. A reverse Cuthill-McKee
 * permutation is computed once on the KKT graph to limit factorization fill
 * (the natural [vars; constraints] order fills the bordered system densely).
 * Values are assembled in original order, then scattered to the permuted
 * matrix through a precomputed index map before each numeric factorization.
 */
#include "internal.h"

void vs_kkt_free(vs_kkt *kk) {
    if (!kk) return;
    free(kk->Kp); free(kk->Ki); free(kk->Kx);
    free(kk->slotW); free(kk->slotSOC);
    free(kk->Kp2); free(kk->Ki2); free(kk->Kx2); free(kk->kmap);
    free(kk->perm); free(kk->iperm); free(kk->ptmp);
    free(kk->Lp); free(kk->Li); free(kk->Lx);
    free(kk->Parent); free(kk->Lnz); free(kk->flag); free(kk->Pattern);
    free(kk->D); free(kk->Dinv); free(kk->Y);
}

/* Reverse Cuthill-McKee ordering of the symmetric KKT graph given by the
 * upper-triangular pattern (Kp,Ki). Fills perm/iperm. */
static void rcm_order(int N, const int *Kp, const int *Ki, int *perm, int *iperm) {
    /* build symmetric adjacency (excluding diagonal) */
    int *deg = vs_calloc_i(N);
    for (int j = 0; j < N; ++j)
        for (int p = Kp[j]; p < Kp[j + 1]; ++p) {
            int i = Ki[p];
            if (i != j) { deg[i]++; deg[j]++; }
        }
    int *adjp = vs_malloc_i(N + 1);
    adjp[0] = 0;
    for (int j = 0; j < N; ++j) adjp[j + 1] = adjp[j] + deg[j];
    int *adj = vs_malloc_i(adjp[N] > 0 ? adjp[N] : 1);
    int *head = vs_malloc_i(N);
    for (int j = 0; j < N; ++j) head[j] = adjp[j];
    for (int j = 0; j < N; ++j)
        for (int p = Kp[j]; p < Kp[j + 1]; ++p) {
            int i = Ki[p];
            if (i != j) { adj[head[i]++] = j; adj[head[j]++] = i; }
        }

    char *seen = (char *)calloc(N ? N : 1, 1);
    int *queue = vs_malloc_i(N);
    int cnt = 0;
    for (int start = 0; start < N; ++start) {
        if (seen[start]) continue;
        /* pick the min-degree unvisited node in this component as root */
        int root = start;
        for (int j = start; j < N; ++j)
            if (!seen[j] && deg[j] < deg[root]) root = j;
        int qh = 0, qt = 0;
        seen[root] = 1; queue[qt++] = root;
        while (qh < qt) {
            int v = queue[qh++];
            perm[cnt++] = v;
            int s = qt;                          /* sort new neighbors by degree */
            for (int p = adjp[v]; p < adjp[v + 1]; ++p) {
                int w = adj[p];
                if (!seen[w]) { seen[w] = 1; queue[qt++] = w; }
            }
            for (int a = s + 1; a < qt; ++a) {   /* insertion sort [s,qt) by deg */
                int key = queue[a], d = deg[key], b = a - 1;
                while (b >= s && deg[queue[b]] > d) { queue[b + 1] = queue[b]; --b; }
                queue[b + 1] = key;
            }
        }
    }
    /* perm currently holds Cuthill-McKee order; reverse it for RCM */
    for (int k = 0; k < N; ++k) iperm[perm[k]] = N - 1 - k;  /* old -> newpos */
    for (int old = 0; old < N; ++old) perm[iperm[old]] = old; /* newpos -> old */

    free(deg); free(adjp); free(adj); free(head);
    free(seen); free(queue);
}

int vs_kkt_setup(vs_kkt *kk, const vs_csc *P, const vs_csc *A,
                 const vs_coneset *cs, double reg) {
    int n = A->n, m = cs->m, N = n + m;
    kk->N = N; kk->n = n; kk->m = m; kk->reg = reg;

    vs_csc *At = vs_csc_transpose(A);
    if (!At) return -1;

    int *rcone = vs_malloc_i(m);
    for (int c = 0; c < cs->ncones; ++c)
        for (int k = 0; k < cs->cones[c].dim; ++k) rcone[cs->off[c] + k] = c;

    /* ---- precount nnz of the upper-triangular KKT (original order) ---- */
    int knnz = 0;
    for (int j = 0; j < n; ++j) {
        int hasdiag = 0;
        for (int p = P ? P->p[j] : 0; P && p < P->p[j + 1]; ++p) {
            ++knnz; if (P->i[p] == j) hasdiag = 1;
        }
        if (!hasdiag) ++knnz;
    }
    for (int r = 0; r < m; ++r) {
        knnz += At->p[r + 1] - At->p[r];
        int c = rcone[r], o = cs->off[c];
        if (cs->cones[c].type == VS_CONE_SOC) knnz += (r - o) + 1;
        else ++knnz;
    }

    kk->Knnz = knnz;
    kk->Kp = vs_malloc_i(N + 1);
    kk->Ki = vs_malloc_i(knnz);
    kk->Kx = vs_calloc_d(knnz);
    kk->slotW = vs_malloc_i(m);
    kk->slotSOC = (int **)calloc(cs->ncones ? cs->ncones : 1, sizeof(int *));
    for (int r = 0; r < m; ++r) kk->slotW[r] = -1;
    for (int c = 0; c < cs->ncones; ++c)
        kk->slotSOC[c] = (cs->cones[c].type == VS_CONE_SOC)
            ? vs_malloc_i((size_t)cs->cones[c].dim * (cs->cones[c].dim + 1) / 2)
            : NULL;

    /* ---- assemble pattern + constant values (original order) ---- */
    int cur = 0;
    for (int j = 0; j < n; ++j) {
        kk->Kp[j] = cur;
        int diagpos = -1;
        for (int p = P ? P->p[j] : 0; P && p < P->p[j + 1]; ++p) {
            kk->Ki[cur] = P->i[p]; kk->Kx[cur] = P->x[p];
            if (P->i[p] == j) diagpos = cur;
            ++cur;
        }
        if (diagpos < 0) { kk->Ki[cur] = j; kk->Kx[cur] = 0.0; diagpos = cur; ++cur; }
        kk->Kx[diagpos] += reg;
    }
    int *socidx = vs_calloc_i(cs->ncones ? cs->ncones : 1);
    for (int r = 0; r < m; ++r) {
        kk->Kp[n + r] = cur;
        for (int p = At->p[r]; p < At->p[r + 1]; ++p) {
            kk->Ki[cur] = At->i[p]; kk->Kx[cur] = At->x[p]; ++cur;
        }
        int c = rcone[r], o = cs->off[c];
        if (cs->cones[c].type == VS_CONE_SOC) {
            int l = r - o;
            for (int i = 0; i <= l; ++i) {
                kk->Ki[cur] = n + o + i; kk->Kx[cur] = 0.0;
                kk->slotSOC[c][socidx[c]++] = cur; ++cur;
            }
        } else {
            kk->Ki[cur] = n + r; kk->Kx[cur] = 0.0; kk->slotW[r] = cur; ++cur;
        }
    }
    kk->Kp[N] = cur;
    free(socidx); free(rcone);
    vs_csc_free(At);

    /* ---- fill-reducing permutation ---- */
    kk->perm  = vs_malloc_i(N);
    kk->iperm = vs_malloc_i(N);
    kk->ptmp  = vs_malloc_d(N);
    rcm_order(N, kk->Kp, kk->Ki, kk->perm, kk->iperm);

    /* ---- build permuted upper-tri pattern + index map ---- */
    kk->Kp2 = vs_malloc_i(N + 1);
    kk->Ki2 = vs_malloc_i(knnz);
    kk->Kx2 = vs_malloc_d(knnz);
    kk->kmap = vs_malloc_i(knnz);
    int *colcnt = vs_calloc_i(N + 1);
    for (int j = 0; j < N; ++j)
        for (int p = kk->Kp[j]; p < kk->Kp[j + 1]; ++p) {
            int a = kk->iperm[kk->Ki[p]], b = kk->iperm[j];
            int nc = (a > b) ? a : b;            /* new column = max */
            colcnt[nc]++;
        }
    kk->Kp2[0] = 0;
    for (int j = 0; j < N; ++j) kk->Kp2[j + 1] = kk->Kp2[j] + colcnt[j];
    int *hd = vs_malloc_i(N);
    for (int j = 0; j < N; ++j) hd[j] = kk->Kp2[j];
    for (int j = 0; j < N; ++j)
        for (int p = kk->Kp[j]; p < kk->Kp[j + 1]; ++p) {
            int a = kk->iperm[kk->Ki[p]], b = kk->iperm[j];
            int nr = (a < b) ? a : b, nc = (a > b) ? a : b;
            int dst = hd[nc]++;
            kk->Ki2[dst] = nr;
            kk->kmap[p] = dst;                   /* original slot p -> permuted */
        }
    free(colcnt); free(hd);

    /* ---- symbolic factorization on the permuted pattern ---- */
    kk->Parent  = vs_malloc_i(N);
    kk->Lnz     = vs_malloc_i(N);
    kk->flag    = vs_malloc_i(N);
    kk->Pattern = vs_malloc_i(N);
    kk->Lp      = vs_malloc_i(N + 1);
    kk->D    = vs_malloc_d(N);
    kk->Dinv = vs_malloc_d(N);
    kk->Y    = vs_calloc_d(N);
    kk->Lnnz = vs_ldl_symbolic(N, kk->Kp2, kk->Ki2, kk->Lp, kk->Parent, kk->Lnz, kk->flag);
    kk->Li = vs_malloc_i(kk->Lnnz);
    kk->Lx = vs_malloc_d(kk->Lnnz);
    return 0;
}

int vs_kkt_factor(vs_kkt *kk, const vs_coneset *cs) {
    double reg = kk->reg;
    for (int c = 0; c < cs->ncones; ++c) {       /* refresh -(W^2+reg) block */
        int d = cs->cones[c].dim;
        if (cs->cones[c].type == VS_CONE_SOC) {
            const double *W2 = cs->W2[c];
            int *slot = kk->slotSOC[c], idx = 0;
            for (int l = 0; l < d; ++l)
                for (int i = 0; i <= l; ++i) {
                    double v = -W2[(size_t)i * d + l];
                    if (i == l) v -= reg;
                    kk->Kx[slot[idx++]] = v;
                }
        } else {
            int o = cs->off[c];
            for (int k = 0; k < d; ++k) {
                double w2 = (cs->cones[c].type == VS_CONE_NONNEG)
                                ? cs->w2diag[o + k] : 0.0;
                kk->Kx[kk->slotW[o + k]] = -(w2 + reg);
            }
        }
    }
    for (int p = 0; p < kk->Knnz; ++p) kk->Kx2[kk->kmap[p]] = kk->Kx[p];

    int info = vs_ldl_numeric(kk->N, kk->Kp2, kk->Ki2, kk->Kx2, kk->Lp, kk->Parent,
                              kk->Lnz, kk->Li, kk->Lx, kk->D, kk->Dinv,
                              kk->Y, kk->Pattern, kk->flag);
    return (info == kk->N) ? 0 : -1;
}

void vs_kkt_solve(vs_kkt *kk, double *rhs) {
    int N = kk->N;
    for (int i = 0; i < N; ++i) kk->ptmp[i] = rhs[kk->perm[i]];   /* permute */
    vs_ldl_solve(N, kk->Lp, kk->Li, kk->Lx, kk->Dinv, kk->ptmp);
    for (int i = 0; i < N; ++i) rhs[kk->perm[i]] = kk->ptmp[i];   /* unpermute */
}
