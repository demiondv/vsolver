# Examples

Standalone programs showing how to model and solve problems with vsolver. Each
file states the problem, gives the equivalent [cvxpy](https://www.cvxpy.org/)
formulation in a comment, then builds and solves it with the vsolver C API.

| file               | class | API shown                                  |
|--------------------|-------|--------------------------------------------|
| `lp.c`             | LP    | direct cone construction (nonneg)          |
| `portfolio.c`      | QP    | product cone (zero + nonneg), `vs_csc_symv`|
| `least_squares.c`  | QP    | OSQP-style `vs_build_osqp` (box bounds)    |
| `socp.c`           | SOCP  | second-order cone construction             |

## Build & run

The examples are built as part of the normal CMake build:

```sh
cmake -S . -B build && cmake --build build
./build/example_lp
./build/example_portfolio
./build/example_least_squares
./build/example_socp
```

Expected output (abridged):

```
LP (diet problem)            -> x = (0, 0, 10),  cost 10
Markowitz portfolio          -> weights sum to 1, tilted to low-variance asset
Bounded least squares        -> x clamped into [-1, 1]
SOCP (linear over a ball)    -> matches the analytic optimum c'x0 - r||c||
```
