# vsolver

A small, from-scratch **conic interior-point optimizer** in C — in the spirit of
MOSEK / OSQP / ECOS / Clarabel. It solves the convex conic quadratic program

```
minimize     ½ xᵀP x + qᵀx
subject to   A x + s = b
             s ∈ K
```

where `K` is a Cartesian product of cones. An OSQP-style two-sided form
`l ≤ A x ≤ u` is supported directly through a front-end that introduces
nonnegative slacks.

No external solver dependencies — the linear algebra (sparse matrices, LDLᵀ
factorization, fill-reducing ordering) is all in-house. The only runtime
dependency is the C math library.

## Status

| Cone                       | Encodes                       | Status     |
|----------------------------|-------------------------------|------------|
| Zero                       | equalities `A x = b`          | ✅ working |
| Nonnegative                | inequalities / LP / QP        | ✅ working |
| Second-order (Lorentz)     | SOCP, norm / QCQP constraints | ✅ working |
| Exponential                | log / entropy / geometric     | ⬜ planned |
| Semidefinite (PSD)         | LMIs, SDP                     | ⬜ planned |

## Algorithm

A primal-dual interior-point method with a **Mehrotra predictor-corrector**
step and **Nesterov-Todd scaling** for the symmetric cones. Each iteration
forms the symmetric quasidefinite KKT system

```
[ P + rI      Aᵀ      ] [Δx]   [ rhs_x ]
[ A       -(W² + rI)  ] [Δz] = [ rhs_z ]
```

and factors it with an in-house quasidefinite **LDLᵀ** (Davis-style, no
pivoting — valid because the regularized KKT is quasidefinite). A **reverse
Cuthill-McKee** ordering is computed once to limit fill; the symbolic pattern
is fixed across iterations, so only the `W²` block's numeric values are
refreshed each iteration. On an ill-conditioned step near the boundary the
solver gracefully returns the current near-optimal iterate.

## Performance

`tests/bench.c` solves a sparse box-constrained QP (tridiagonal `P`). Scaling
is linear in problem size (7 interior-point iterations throughout):

| n        | KKT size (N) | solve   |
|----------|--------------|---------|
| 10,000   | 30,000       | 12 ms   |
| 50,000   | 150,000      | 72 ms   |
| 200,000  | 600,000      | 318 ms  |
| 500,000  | 1,500,000    | 845 ms  |

The natural ordering, by contrast, fills the bordered KKT densely and is
~O(N³); the fill-reducing ordering is what makes it scale.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Requires a C11 compiler, CMake ≥ 3.16, and a C math library. No BLAS/LAPACK.

## Quick start

```c
#include "vsolver.h"
#include <stdio.h>

/* minimize ½(x0² + x1²) - x0 - 2 x1   s.t.  x0 ≤ 5, x1 ≤ 5  (x ≥ -∞)
 * unconstrained optimum (1, 2) is feasible. */
int main(void) {
    int    Pi[] = {0, 1}, Pj[] = {0, 1};   double Px[] = {1.0, 1.0};   /* P = I  */
    int    Ai[] = {0, 1}, Aj[] = {0, 1};   double Ax[] = {1.0, 1.0};   /* A = I  */
    vs_csc *P = vs_csc_from_triplets(2, 2, 2, Pi, Pj, Px);
    vs_csc *A = vs_csc_from_triplets(2, 2, 2, Ai, Aj, Ax);
    double q[] = {-1.0, -2.0}, b[] = {5.0, 5.0};

    vs_cone    cone = { VS_CONE_NONNEG, 2 };
    vs_problem prob = { 2, 2, P, q, A, b, &cone, 1 };

    vs_settings set; vs_default_settings(&set);
    vs_solution sol;
    vs_solve(&prob, &set, &sol);
    printf("%s  x = (%.3f, %.3f)\n", vs_status_str(sol.status), sol.x[0], sol.x[1]);

    vs_solution_free(&sol);
    vs_csc_free(P); vs_csc_free(A);
}
```

### OSQP-style front-end

```c
/* minimize ½ xᵀP x + qᵀx   subject to   l ≤ A x ≤ u
 * (NULL l/u, or |bound| ≥ VS_INF, means that side is absent) */
vs_problem *prob = vs_build_osqp(P, q, A, l, u);
vs_solve(prob, &set, &sol);
vs_problem_free(prob);   /* frees the builder-owned A/b/cones, not your P/q */
```

## Public API

| Function | Purpose |
|----------|---------|
| `vs_csc_from_triplets` | build a sparse matrix from `(i, j, value)` entries |
| `vs_default_settings`  | fill a `vs_settings` with defaults |
| `vs_solve`             | solve a `vs_problem` |
| `vs_build_osqp`        | construct a problem from `l ≤ Ax ≤ u` |
| `vs_solution_free` / `vs_problem_free` | release results / built problems |
| `vs_status_str`        | human-readable status |

The objective `P` is the **upper triangle** of a symmetric PSD matrix (or
`NULL` for an LP). Cones are listed in row order and must partition the rows of
`A`.

## Validation

Each cone is checked against a problem with a known answer:

- `tests/test_qp.c`   — quadratic programs with active / inactive bounds.
- `tests/test_socp.c` — projection onto the second-order cone (closed form:
  interior / boundary-blend / polar regimes).
- `tests/test_osqp.c` — a box-constrained QP through `vs_build_osqp`, versus the
  analytic clamp.

These solve to ~1e-11.

`tests/test_cvxpy.c` additionally cross-checks vsolver against
[cvxpy](https://www.cvxpy.org/) (CLARABEL) on 13 LP / QP / SOCP problems —
inequality and equality constraints, second-order cones, product cones, a
random QP, a robust LP, and more. References are embedded in
`tests/cvxpy_refs.h`, generated by `tests/gen_cvxpy_refs.py`; cvxpy is only
needed to *regenerate* them:

```sh
python3 -m venv .venv && .venv/bin/pip install cvxpy
.venv/bin/python tests/gen_cvxpy_refs.py
```

vsolver matches the cvxpy objective and primal to ~1e-9.

## Examples

`examples/` has small, self-contained, annotated programs (each with the
equivalent cvxpy formulation in a comment):

| file               | class | API shown                                   |
|--------------------|-------|---------------------------------------------|
| `lp.c`             | LP    | direct cone construction                    |
| `portfolio.c`      | QP    | Markowitz; product cone (zero + nonneg)     |
| `least_squares.c`  | QP    | OSQP-style `vs_build_osqp` (box bounds)     |
| `socp.c`           | SOCP  | second-order cone construction              |

They build with the project: `./build/example_lp`, `example_portfolio`, etc.
See `examples/README.md`.

## Layout

```
include/vsolver.h          public API (problem, cones, settings, solution)
src/linalg/csc.c          CSC sparse matrices, BLAS-1, mat-vec, transpose
src/linalg/ldl.c          sparse quasidefinite LDLᵀ (symbolic/numeric/solve)
src/cones/cone.c          symmetric-cone Jordan algebra + NT scaling
src/kkt/kkt.c             sparse KKT assembly, RCM ordering, factor/solve
src/solver/api.c          settings, status strings, solution lifecycle
src/solver/build.c        OSQP-style l ≤ Ax ≤ u front-end
src/solver/ipm.c          the interior-point loop
tests/                    closed-form + cvxpy cross-validation, benchmark
examples/                 worked LP / QP / SOCP programs
```

## Roadmap

1. ~~SOCP (second-order cones)~~ — done.
2. ~~OSQP-compatible front-end~~ — done.
3. ~~Sparse linear algebra (LDLᵀ + fill-reducing ordering)~~ — done.
4. **Exponential & PSD cones** — non-symmetric / spectral scalings.
5. **Further performance** — AMD ordering (better fill than RCM on general
   sparsity), Ruiz equilibration, presolve, warm starts, iterative refinement.

## License

MIT (intended). See `LICENSE` if present.
