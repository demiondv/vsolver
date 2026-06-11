/* vsolver - symmetric-cone Jordan algebra and Nesterov-Todd scaling.
 *
 * Supported cones: zero, nonnegative orthant, second-order (Lorentz).
 * Each second-order block uses the standard NT scaling
 *
 *     W = eta * Wbar,   Wbar = [ w0   w1'                     ]
 *                              [ w1   I + w1 w1'/(1+w0)       ]
 *
 * with Wbar J Wbar = J (J = diag(1,-I)).  For the nonnegative orthant the
 * scaling degenerates to the diagonal W^2 = diag(s/z), matching the plain QP
 * interior-point method.
 */
#include "internal.h"
#include <math.h>

/* J-inner product for a second-order block: x0 y0 - x1.y1 */
static double jdot(const double *x, const double *y, int d) {
    double s = x[0] * y[0];
    for (int k = 1; k < d; ++k) s -= x[k] * y[k];
    return s;
}

int vs_coneset_init(vs_coneset *cs, const vs_cone *cones, int ncones) {
    cs->cones = cones; cs->ncones = ncones;
    cs->off = vs_malloc_i(ncones);
    cs->eta  = vs_calloc_d(ncones);
    cs->wbar = (double **)calloc(ncones ? ncones : 1, sizeof(double *));
    cs->W2   = (double **)calloc(ncones ? ncones : 1, sizeof(double *));
    int m = 0, deg = 0;
    for (int c = 0; c < ncones; ++c) {
        cs->off[c] = m;
        int d = cones[c].dim;
        m += d;
        switch (cones[c].type) {
            case VS_CONE_NONNEG: deg += d; break;
            case VS_CONE_SOC:    deg += 2;
                cs->wbar[c] = vs_calloc_d(d);
                cs->W2[c]   = vs_calloc_d((size_t)d * d);
                break;
            default: /* zero cone: rank 0 */ break;
        }
    }
    cs->m = m; cs->degree = deg;
    cs->w2diag = vs_calloc_d(m);
    cs->lam    = vs_calloc_d(m);
    return (cs->off && cs->eta && cs->wbar && cs->W2 && cs->w2diag && cs->lam)
               ? 0 : -1;
}

void vs_coneset_free(vs_coneset *cs) {
    if (!cs) return;
    for (int c = 0; c < cs->ncones; ++c) {
        if (cs->wbar) free(cs->wbar[c]);
        if (cs->W2)   free(cs->W2[c]);
    }
    free(cs->off); free(cs->eta); free(cs->wbar); free(cs->W2);
    free(cs->w2diag); free(cs->lam);
}

void vs_cone_identity(const vs_coneset *cs, double *s, double *z) {
    for (int r = 0; r < cs->m; ++r) { s[r] = 0.0; z[r] = 0.0; }
    for (int c = 0; c < cs->ncones; ++c) {
        int o = cs->off[c], d = cs->cones[c].dim;
        switch (cs->cones[c].type) {
            case VS_CONE_NONNEG:
                for (int k = 0; k < d; ++k) { s[o + k] = 1.0; z[o + k] = 1.0; }
                break;
            case VS_CONE_SOC:
                s[o] = 1.0; z[o] = 1.0;     /* e = (1,0,...,0) */
                break;
            default: break;                  /* zero cone stays 0 */
        }
    }
}

void vs_cone_prod(const vs_coneset *cs, const double *a, const double *b,
                  double *out) {
    for (int c = 0; c < cs->ncones; ++c) {
        int o = cs->off[c], d = cs->cones[c].dim;
        if (cs->cones[c].type == VS_CONE_SOC) {
            double a0 = a[o], b0 = b[o];
            double dot = a0 * b0;                 /* (a o b)_0 = a.b */
            for (int k = 1; k < d; ++k) dot += a[o + k] * b[o + k];
            out[o] = dot;
            for (int k = 1; k < d; ++k)           /* (a o b)_1 = a0 b1 + b0 a1 */
                out[o + k] = a0 * b[o + k] + b0 * a[o + k];
        } else { /* nonneg / zero: elementwise */
            for (int k = 0; k < d; ++k) out[o + k] = a[o + k] * b[o + k];
        }
    }
}

void vs_cone_div(const vs_coneset *cs, const double *lam, const double *y,
                 double *out) {
    for (int c = 0; c < cs->ncones; ++c) {
        int o = cs->off[c], d = cs->cones[c].dim;
        if (cs->cones[c].type == VS_CONE_SOC) {
            /* out = lam^{-1} o y, with lam^{-1} = (1/det)(lam0, -lam1) */
            double l0 = lam[o];
            double det = jdot(lam + o, lam + o, d);          /* l0^2-||l1||^2 */
            double linv0 = l0 / det;
            double dot = linv0 * y[o];                       /* (linv o y)_0 */
            for (int k = 1; k < d; ++k) dot += (-lam[o + k] / det) * y[o + k];
            out[o] = dot;
            for (int k = 1; k < d; ++k)
                out[o + k] = linv0 * y[o + k] + y[o] * (-lam[o + k] / det);
        } else {
            for (int k = 0; k < d; ++k) out[o + k] = y[o + k] / lam[o + k];
        }
    }
}

int vs_cone_nt(vs_coneset *cs, const double *s, const double *z) {
    for (int c = 0; c < cs->ncones; ++c) {
        int o = cs->off[c], d = cs->cones[c].dim;
        switch (cs->cones[c].type) {
            case VS_CONE_NONNEG:
                for (int k = 0; k < d; ++k) {
                    cs->w2diag[o + k] = s[o + k] / z[o + k];
                    cs->lam[o + k] = sqrt(s[o + k] * z[o + k]);
                }
                break;
            case VS_CONE_SOC: {
                double sJs = jdot(s + o, s + o, d);
                double zJz = jdot(z + o, z + o, d);
                if (sJs <= 0.0 || zJz <= 0.0) return -1;
                double sn = sqrt(sJs), zn = sqrt(zJz);
                double *wb = cs->wbar[c];
                /* normalized sbar, zbar (J-norm 1) */
                double sb0 = s[o] / sn, zb0 = z[o] / zn;
                double sdotz = sb0 * zb0;
                for (int k = 1; k < d; ++k)
                    sdotz += (s[o + k] / sn) * (z[o + k] / zn);
                double gamma = sqrt((1.0 + sdotz) / 2.0);
                /* wbar = (sbar + J zbar) / (2 gamma) */
                wb[0] = (sb0 + zb0) / (2.0 * gamma);
                for (int k = 1; k < d; ++k)
                    wb[k] = (s[o + k] / sn - z[o + k] / zn) / (2.0 * gamma);
                double eta = sqrt(sn / zn);
                cs->eta[c] = eta;

                /* dense W2 = eta^2 * Wbar^2; build Wbar in W2, then square */
                double *W2 = cs->W2[c];
                double w0 = wb[0], denom = 1.0 + w0;
                for (int i = 0; i < d; ++i)
                    for (int j = 0; j < d; ++j) {
                        double v;
                        if (i == 0 && j == 0)      v = w0;
                        else if (i == 0)           v = wb[j];
                        else if (j == 0)           v = wb[i];
                        else v = ((i == j) ? 1.0 : 0.0) + wb[i] * wb[j] / denom;
                        W2[(size_t)i * d + j] = v;   /* Wbar in row-major */
                    }
                /* square in place: need a copy */
                double *tmp = vs_malloc_d((size_t)d * d);
                if (!tmp) return -1;
                memcpy(tmp, W2, (size_t)d * d * sizeof(double));
                double e2 = eta * eta;
                for (int i = 0; i < d; ++i)
                    for (int j = 0; j < d; ++j) {
                        double acc = 0.0;
                        for (int k = 0; k < d; ++k)
                            acc += tmp[(size_t)i * d + k] * tmp[(size_t)k * d + j];
                        W2[(size_t)i * d + j] = e2 * acc;
                    }
                free(tmp);

                /* lam = W z = eta * Wbar z */
                {
                    double *lo = cs->lam + o;
                    double w1z = 0.0;
                    for (int k = 1; k < d; ++k) w1z += wb[k] * z[o + k];
                    lo[0] = eta * (wb[0] * z[o] + w1z);
                    for (int k = 1; k < d; ++k)
                        lo[k] = eta * (z[o] * wb[k] + z[o + k]
                                       + wb[k] * w1z / denom);
                }
            } break;
            default: break; /* zero cone */
        }
    }
    return 0;
}

/* apply Wbar (sign=+1) or Wbar with w1->-w1 (sign=-1, gives Wbar^{-1}) */
static void soc_apply_wbar(const double *wb, int d, double sign,
                           const double *x, double *y) {
    double w0 = wb[0], denom = 1.0 + w0;
    double d1x = 0.0;
    for (int k = 1; k < d; ++k) d1x += sign * wb[k] * x[k];
    y[0] = w0 * x[0] + d1x;
    for (int k = 1; k < d; ++k)
        y[k] = sign * wb[k] * x[0] + x[k] + sign * wb[k] * (d1x / denom);
}

void vs_cone_W(const vs_coneset *cs, const double *x, double *out) {
    for (int c = 0; c < cs->ncones; ++c) {
        int o = cs->off[c], d = cs->cones[c].dim;
        if (cs->cones[c].type == VS_CONE_SOC) {
            soc_apply_wbar(cs->wbar[c], d, +1.0, x + o, out + o);
            for (int k = 0; k < d; ++k) out[o + k] *= cs->eta[c];
        } else if (cs->cones[c].type == VS_CONE_NONNEG) {
            for (int k = 0; k < d; ++k) out[o + k] = sqrt(cs->w2diag[o + k]) * x[o + k];
        } else {
            for (int k = 0; k < d; ++k) out[o + k] = 0.0;
        }
    }
}

void vs_cone_Winv(const vs_coneset *cs, const double *x, double *out) {
    for (int c = 0; c < cs->ncones; ++c) {
        int o = cs->off[c], d = cs->cones[c].dim;
        if (cs->cones[c].type == VS_CONE_SOC) {
            soc_apply_wbar(cs->wbar[c], d, -1.0, x + o, out + o);
            for (int k = 0; k < d; ++k) out[o + k] /= cs->eta[c];
        } else if (cs->cones[c].type == VS_CONE_NONNEG) {
            for (int k = 0; k < d; ++k) out[o + k] = x[o + k] / sqrt(cs->w2diag[o + k]);
        } else {
            for (int k = 0; k < d; ++k) out[o + k] = 0.0;
        }
    }
}

/* smallest positive root of a*t^2 + 2b*t + c = 0, else INFINITY */
static double smallest_pos_quad(double a, double b, double c) {
    const double INF = (double)INFINITY;
    if (a == 0.0) {
        if (b == 0.0) return INF;
        double t = -c / (2.0 * b);
        return (t > 0.0) ? t : INF;
    }
    double disc = b * b - a * c;
    if (disc < 0.0) return INF;
    double sq = sqrt(disc);
    double t1 = (-b - sq) / a, t2 = (-b + sq) / a;
    double lo = VS_MIN(t1, t2), hi = VS_MAX(t1, t2);
    if (lo > 0.0) return lo;
    if (hi > 0.0) return hi;
    return INF;
}

double vs_cone_max_step(const vs_coneset *cs, const double *v,
                        const double *dv) {
    double amax = (double)INFINITY;
    for (int c = 0; c < cs->ncones; ++c) {
        int o = cs->off[c], d = cs->cones[c].dim;
        if (cs->cones[c].type == VS_CONE_NONNEG) {
            for (int k = 0; k < d; ++k)
                if (dv[o + k] < 0.0) {
                    double t = -v[o + k] / dv[o + k];
                    if (t < amax) amax = t;
                }
        } else if (cs->cones[c].type == VS_CONE_SOC) {
            /* boundary: jdot(v+t dv, v+t dv) = 0 and (v+t dv)_0 >= 0 */
            double a = jdot(dv + o, dv + o, d);
            double b = jdot(v + o, dv + o, d);
            double cc = jdot(v + o, v + o, d);
            double t = smallest_pos_quad(a, b, cc);
            if (t < amax) amax = t;
            if (dv[o] < 0.0) {
                double th = -v[o] / dv[o];
                if (th < amax) amax = th;
            }
        }
    }
    return amax;
}
