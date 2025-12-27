
Integration steps:

1) Copy files:
   - include/bfgs.h  -> your project's include/
   - src/bfgs.cpp    -> your project's src/

2) Register in factory (src/factory.cpp):
   Add:   #include "bfgs.h"
   And in makeMethod(...):
          if (eq(name, "bfgs")) return std::make_unique<BFGS>();

3) Optional config ([bfgs] section):
   alpha0 = 1.0
   c1 = 1e-4
   c2 = 0.9
   backtracks = 20
   tol = 1e-8
   max_iters = 2000
   grad_eps = 1e-8
   reset_on_nan = 1

4) Usage:
   - Standalone:   ./optimsolution bfgs rosenbrock 10
   - As local refine (e.g., inside DE): set local_method = bfgs
