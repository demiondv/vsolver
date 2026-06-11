#!/usr/bin/env python3
"""Generate reference solutions for vsolver tests using cvxpy.

Each problem is defined directly in vsolver standard form

    minimize   1/2 x'P x + q'x
    subject to A x + s = b,   s in K

so that vsolver and cvxpy solve *identical* data. We solve with cvxpy
(CLARABEL) and emit tests/cvxpy_refs.h with the problem data and the
reference primal solution / optimal value.

Regenerate with:  .venv/bin/python tests/gen_cvxpy_refs.py
cvxpy is only needed to regenerate; the committed header is self-contained.
"""
import numpy as np
import cvxpy as cp

ZERO, NONNEG, SOC = 0, 1, 2   # must match vs_cone_type


def solve(P, q, A, b, cones):
    """Solve the standard-form problem with cvxpy; return (x*, pobj)."""
    n = A.shape[1]
    x = cp.Variable(n)
    s = b - A @ x
    cons, off = [], 0
    for (ctype, d) in cones:
        blk = s[off:off + d]
        if ctype == ZERO:
            cons.append(blk == 0)
        elif ctype == NONNEG:
            cons.append(blk >= 0)
        elif ctype == SOC:
            cons.append(cp.SOC(blk[0], blk[1:]))
        off += d
    obj = q @ x
    if P is not None:
        obj = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + obj
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    assert prob.status == cp.OPTIMAL, f"cvxpy status {prob.status}"
    return np.asarray(x.value).ravel(), float(prob.value)


# ---------------------------------------------------------------------------
# problem library  (all designed to have unique optima)
# ---------------------------------------------------------------------------
def make_problems():
    P = []

    # LP-1: min -x0-2x1 s.t. x0+x1<=4, x0+3x1<=6, x>=0   -> (3,1)
    A = np.array([[1, 1], [1, 3], [-1, 0], [0, -1]], float)
    b = np.array([4, 6, 0, 0], float)
    P.append(dict(name="lp_inequality", P=None, q=np.array([-1, -2.0]),
                  A=A, b=b, cones=[(NONNEG, 4)]))

    # LP-2: min 2x0+3x1+x2 s.t. sum x = 1, x>=0  -> (0,0,1)  (zero + nonneg)
    A = np.array([[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    b = np.array([1, 0, 0, 0], float)
    P.append(dict(name="lp_equality", P=None, q=np.array([2, 3, 1.0]),
                  A=A, b=b, cones=[(ZERO, 1), (NONNEG, 3)]))

    # QP-1: dense SPD P, nonneg bounds
    Pm = np.array([[2.0, 0.5], [0.5, 1.0]])
    A = np.array([[-1, 0], [0, -1]], float)
    b = np.array([0, 0.0])
    P.append(dict(name="qp_dense_bounds", P=Pm, q=np.array([-1.0, -2.0]),
                  A=A, b=b, cones=[(NONNEG, 2)]))

    # QP-2: min 1/2 x'x s.t. sum x = 1  -> (1/3,1/3,1/3)  (equality QP)
    A = np.array([[1, 1, 1]], float)
    b = np.array([1.0])
    P.append(dict(name="qp_equality", P=np.eye(3), q=np.zeros(3),
                  A=A, b=b, cones=[(ZERO, 1)]))

    # SOCP-1: min c'x s.t. ||x|| <= 1  -> x = -c/||c||
    c = np.array([1.0, 2.0])
    A = np.array([[0, 0], [-1, 0], [0, -1]], float)   # s = (1, x)
    b = np.array([1, 0, 0.0])
    P.append(dict(name="socp_ball", P=None, q=c, A=A, b=b,
                  cones=[(SOC, 3)]))

    # SOCP-2: projection of a onto ball of radius r (QP objective + SOC)
    a = np.array([2.0, 2.0]); r = 1.0
    A = np.array([[0, 0], [-1, 0], [0, -1]], float)   # s = (r, x)
    b = np.array([r, 0, 0.0])
    P.append(dict(name="socp_ball_proj", P=np.eye(2), q=-a, A=A, b=b,
                  cones=[(SOC, 3)]))

    # SOCP-3: min c'x s.t. x>=0 and ||x||<=1  (product: nonneg + soc)
    c = np.array([-1.0, -1.0])
    A = np.array([[-1, 0], [0, -1],          # x >= 0
                  [0, 0], [-1, 0], [0, -1]], float)   # s_soc = (1, x)
    b = np.array([0, 0, 1, 0, 0.0])
    P.append(dict(name="socp_nonneg_mix", P=None, q=c, A=A, b=b,
                  cones=[(NONNEG, 2), (SOC, 3)]))

    # QP-3: strictly convex, equality + nonneg (projection onto the simplex)
    A = np.array([[1, 1, 1],                 # sum x = 1   (zero)
                  [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float)   # x >= 0
    b = np.array([1, 0, 0, 0.0])
    P.append(dict(name="qp_simplex", P=np.eye(3), q=np.array([-2.0, -1.0, 0.0]),
                  A=A, b=b, cones=[(ZERO, 1), (NONNEG, 3)]))

    # SOCP-4: two independent second-order cones
    c = np.array([-1.0, -1.0, -1.0, -2.0])   # x = (x0,x1, x2,x3)
    A = np.array([[0, 0, 0, 0], [-1, 0, 0, 0], [0, -1, 0, 0],   # (1, x0, x1)
                  [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],  # (1, x2, x3)
                 float)
    b = np.array([1, 0, 0, 1, 0, 0.0])
    P.append(dict(name="socp_two_blocks", P=None, q=c, A=A, b=b,
                  cones=[(SOC, 3), (SOC, 3)]))

    # QP-4: larger random strictly-convex QP with box constraints
    rng = np.random.default_rng(7)
    nq = 5
    M = rng.standard_normal((nq, nq))
    Pm = M.T @ M + np.eye(nq)                 # SPD
    q = rng.standard_normal(nq)
    rows = np.vstack([np.eye(nq), -np.eye(nq)])     # x<=1, x>=-1
    bb = np.concatenate([np.ones(nq), np.ones(nq)])
    P.append(dict(name="qp_random_box", P=Pm, q=q, A=rows, b=bb,
                  cones=[(NONNEG, 2 * nq)]))

    # QP-5: strictly convex QP with equality AND box (sum=1, 0<=x<=0.5)
    A = np.vstack([
        np.ones((1, 3)),          # sum x = 1            (zero)
        np.eye(3),                # x <= 0.5             (nonneg)
        -np.eye(3),               # x >= 0               (nonneg)
    ])
    b = np.array([1, 0.5, 0.5, 0.5, 0, 0, 0.0])
    P.append(dict(name="qp_box_equality", P=np.eye(3), q=np.array([-1.0, 0.0, 1.0]),
                  A=A, b=b, cones=[(ZERO, 1), (NONNEG, 6)]))

    # SOCP-5: robust LP, ellipsoidal uncertainty (nonneg + soc product)
    #   min -x0-x1  s.t.  x0+x1 + 0.5||x|| <= 2,  x>=0
    A = np.array([
        [-1, 0], [0, -1],                       # x >= 0           (nonneg)
        [1, 1], [-0.5, 0], [0, -0.5],           # s_soc = (2-x0-x1, 0.5x)
    ], float)
    b = np.array([0, 0, 2, 0, 0.0])
    P.append(dict(name="socp_robust_lp", P=None, q=np.array([-1.0, -1.0]),
                  A=A, b=b, cones=[(NONNEG, 2), (SOC, 3)]))

    # SOCP-6: larger second-order cone, min c'x s.t. ||x|| <= R
    c = np.array([1.0, -2.0, 0.5, 1.0]); R = 3.0
    A = np.vstack([np.zeros((1, 4)), -np.eye(4)])   # s = (R, x)
    b = np.array([R, 0, 0, 0, 0.0])
    P.append(dict(name="socp_ball4d", P=None, q=c, A=A, b=b,
                  cones=[(SOC, 5)]))

    return P


# ---------------------------------------------------------------------------
def carr_i(name, vals):
    body = ", ".join(str(int(v)) for v in vals) or "0"
    return f"static const int {name}[] = {{{body}}};"


def carr_d(name, vals):
    body = ", ".join(f"{float(v):.17g}" for v in vals) or "0"
    return f"static const double {name}[] = {{{body}}};"


def emit(problems, path):
    out = ["/* AUTO-GENERATED by tests/gen_cvxpy_refs.py - do not edit. */",
           "/* Reference solutions from cvxpy (CLARABEL). */",
           "#ifndef VSOLVER_CVXPY_REFS_H", "#define VSOLVER_CVXPY_REFS_H", ""]
    out.append("typedef struct {")
    out.append("    const char *name; int n, m, ncones;")
    out.append("    const int *Pi, *Pj; const double *Px; int Pnnz;")
    out.append("    const double *q;")
    out.append("    const int *Ai, *Aj; const double *Ax; int Annz;")
    out.append("    const double *b;")
    out.append("    const int *cone_type, *cone_dim;")
    out.append("    const double *refx; double refobj;")
    out.append("} cvxpy_case;\n")

    table = []
    for idx, pr in enumerate(problems):
        x, pobj = solve(pr["P"], pr["q"], pr["A"], pr["b"], pr["cones"])
        n = pr["A"].shape[1]; m = pr["A"].shape[0]
        pfx = f"c{idx}_"

        # P upper-triangle triplets
        Pi, Pj, Px = [], [], []
        if pr["P"] is not None:
            M = pr["P"]
            for i in range(n):
                for j in range(i, n):
                    if M[i, j] != 0.0:
                        Pi.append(i); Pj.append(j); Px.append(M[i, j])
        # A triplets
        Ai, Aj, Ax = [], [], []
        for i in range(m):
            for j in range(n):
                if pr["A"][i, j] != 0.0:
                    Ai.append(i); Aj.append(j); Ax.append(pr["A"][i, j])

        ctype = [c[0] for c in pr["cones"]]
        cdim = [c[1] for c in pr["cones"]]

        out.append(f"/* {pr['name']}: pobj={pobj:.10g} */")
        out.append(carr_i(pfx + "Pi", Pi)); out.append(carr_i(pfx + "Pj", Pj))
        out.append(carr_d(pfx + "Px", Px))
        out.append(carr_d(pfx + "q", pr["q"]))
        out.append(carr_i(pfx + "Ai", Ai)); out.append(carr_i(pfx + "Aj", Aj))
        out.append(carr_d(pfx + "Ax", Ax))
        out.append(carr_d(pfx + "b", pr["b"]))
        out.append(carr_i(pfx + "ct", ctype)); out.append(carr_i(pfx + "cd", cdim))
        out.append(carr_d(pfx + "refx", x))
        out.append("")
        table.append(
            f'  {{"{pr["name"]}", {n}, {m}, {len(ctype)}, '
            f'{pfx}Pi, {pfx}Pj, {pfx}Px, {len(Px)}, {pfx}q, '
            f'{pfx}Ai, {pfx}Aj, {pfx}Ax, {len(Ax)}, {pfx}b, '
            f'{pfx}ct, {pfx}cd, {pfx}refx, {pobj:.17g}}},')

    out.append("static const cvxpy_case cvxpy_cases[] = {")
    out.extend(table)
    out.append("};")
    out.append("enum { cvxpy_ncases = sizeof(cvxpy_cases)/sizeof(cvxpy_cases[0]) };")
    out.append("")
    out.append("#endif")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {path}: {len(problems)} problems")


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    emit(make_problems(), os.path.join(here, "cvxpy_refs.h"))
