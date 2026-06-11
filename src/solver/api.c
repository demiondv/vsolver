/* vsolver - settings, status, solution lifecycle */
#include "internal.h"

void vs_default_settings(vs_settings *set) {
    set->max_iter   = 100;
    set->eps_abs    = 1e-9;
    set->eps_rel    = 1e-9;
    set->reg_static = 1e-8;
    set->verbose    = 0;
}

const char *vs_status_str(vs_status s) {
    switch (s) {
        case VS_SOLVED:        return "solved";
        case VS_PRIMAL_INFEAS: return "primal infeasible";
        case VS_DUAL_INFEAS:   return "dual infeasible";
        case VS_MAX_ITER:      return "max iterations";
        case VS_NUMERICAL:     return "numerical error";
        case VS_ERROR:         return "error";
        default:               return "unknown";
    }
}

void vs_solution_free(vs_solution *sol) {
    if (!sol) return;
    free(sol->x); sol->x = NULL;
    free(sol->s); sol->s = NULL;
    free(sol->z); sol->z = NULL;
}
