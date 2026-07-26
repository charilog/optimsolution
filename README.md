<p align="center">
  <img src="./docs/optimsolution.png" alt="Optimsolution logo" width="720">
</p>

Optimsolution (version 52+) is a C++ optimization framework that combines a full-featured Qt-based GUI with a CLI for experimenting with population-based metaheuristics and numerical optimizers across benchmark and real-world problems. The GUI drives the complete workflow selecting methods and problems, configuring runs, launching batch experiments, visualizing convergence, and analyzing results through ranking tables and statistical tests while the CLI provides the same capabilities for scripted and headless execution. Both interfaces share the same optimization core and configuration file, and both support explicit computational budgets, multiple initialization strategies, local search integration, and sensitivity analysis of method and problem parameters.

> English manual: [optimsolution_manual_EN.pdf](./docs/optimsolutionManual_EN.pdf)
>
> Greek manual: [optimsolution_manual_GR.pdf](./docs/optimsolutionManual_GR.pdf)

---

## 1) Changelog

### v51 → v52

### OpenMP parallelization across every run mode
Independent runs can now execute in parallel via OpenMP instead of strictly one-at-a-time. This applies uniformly to **all four run modes** — Single run, Batch run, and both Sensitivity analysis modes (method parameters and problem parameters) — since all of them are ultimately built on the same "N independent runs" execution path.

Two new `[global]` settings control this:

```ini
[global]
parallel_runs = 0   ; 0 = serial (default), 1 = parallel independent runs
omp_threads   = 0   ; 0 = use all available cores, N = cap at N threads
```

Parallel execution is designed to be **numerically identical** to serial execution: every run keeps its own `Problem` instance and its own RNG seeded with `seed_base + run_index`, exactly as in the serial path, so turning `parallel_runs` on or off never changes the reported statistics — only the wall-clock time. Problems with non-thread-safe global state (the GKLS generator family) are automatically kept serial regardless of this setting, with a note printed to the console.

### Correctness fixes across several problems
A systematic review of the problem library turned up and corrected a number of issues, including (non-exhaustive):
- Analytic gradient errors (a mis-scaled term in `levy`, a missing contribution in `stepellipsoidal`, and a `weierstrass` gradient replaced with a closed-form derivative after its finite-difference version proved numerically unusable at the default parameters).
- A malformed objective function in `test30n` (an incorrect term combination that silently dropped part of the intended landscape).
- An incorrect cutoff-function formula and mismatched reference parameters in `tersoffb`/`tersoffc`, plus a corrected default dimension (now D=30, matching the literature).
- Several stale or duplicated entries in the fixed-dimension problem registry (affecting `hydrothermal`, `messenger`, `polyphase`, `shubert`, among others), which had been causing the GUI to report or enforce the wrong dimension for those problems.
- A GUI/state bug where the Batch run **Results table** could go blank after a batch finished when exactly one problem was selected (the underlying result data was always safe on disk — this was purely a display/reconstruction issue).

### Code Wizard improvements
The **Code Wizard** panel supports creating and deleting methods and problems directly from the GUI. When a new method or problem is generated or deleted, the application writes a pending-rebuild flag (`.rebuild_pending`) and prompts for a restart. On the next startup, if the flag is detected, the GUI offers to rebuild automatically before loading the factory — eliminating the LNK1104 locked-executable error that would otherwise occur on Windows.

### v52 → v52+

#### New run mode: Multi-objective optimization
A fifth run mode joins Single/Batch/Sensitivity: **Multi-objective optimization**, producing a Pareto front instead of a single best value. Its settings panel lists **Method** first, then **Problem**, then **Dimension** — population and generations are intentionally **not** exposed here; like every other run mode, they are read from `optimsolution.cfg` (each method's own section, e.g. `[nsga2]`), so a batch of multi-objective runs stays configured from the same single file as everything else. Output CSVs from this mode are written to `optimsolution_gui_run/`, matching the convention already used by Single/Batch/Sensitivity, instead of the application's root folder.

New multi-objective methods (3):

| Short name | Full name |
|---|---|
| `nsga2` | NSGA-II (Non-dominated Sorting Genetic Algorithm II) |
| `moead` | MOEA/D-DE (decomposition-based, Tchebycheff scalarization) |
| `mopso` | Multi-Objective Particle Swarm Optimization (external archive + crowding-based leader selection) |

New multi-objective benchmark problems — the ZDT family (Zitzler-Deb-Thiele), each with a known analytic Pareto front for validating that a method actually converges to the right front rather than merely producing *some* trade-off curve:

| Problem | Pareto front shape | What it stresses |
|---|---|---|
| `zdt1` | Convex | Baseline correctness |
| `zdt2` | Concave | Non-convexity (defeats naive weighted-sum scalarization) |
| `zdt3` | Disconnected (~5 arcs) | Diversity preservation across disjoint regions |
| `zdt4` | Convex, surrounded by ~21^(D-1) local fronts | Global convergence vs. deceptive local fronts |

#### New single-objective methods
- **`rdex`** — RDEx-SOP, winner of the IEEE CEC 2025 Single-Objective Optimization competition. Hybridizes a standard current-to-pbest/1 branch with an exploitation-biased branch, adapted via success-history memories and linear population reduction.
- **`rde`** — RDE (Reconstructed Differential Evolution), RDEx-SOP's predecessor: adds a JADE/SHADE-style external archive and Extended Rank-based Selective Pressure (RSP) on top of the same success-history lineage, with population scaled to dimension (`Nmax = 18·D`, budget-aware capped so high-D runs are not starved of generations under this project's fixed `max_evals` convention).

Both were corrected after benchmarking surfaced two real bugs: `rdex`'s Cauchy local-perturbation step originally gated every non-crossover dimension independently, making its disruptiveness scale with D (fine at low D, corrosive at high D) — fixed to perturb a single randomly chosen dimension per individual, matching standard iLSHADE-RSP-style local perturbation. `rde` additionally needed the budget-aware population cap described above for the same underlying reason (population and generation-count trading off very differently once the evaluation budget stops scaling with D).

#### Full CEC2017 benchmark suite (29 of 29 usable functions)
Every function in the CEC2017 bound-constrained single-objective suite is now available — `cec2017f1`, `cec2017f3` through `cec2017f30` (F2 is **not** missing by omission: it was officially deleted from the suite by the benchmark's own authors due to a known instability, and the official reference code refuses to evaluate it). Coverage: 2 unimodal, 7 simple multimodal, 10 hybrid, and 10 composition functions, each supporting **D = 10, 30, 50** (the dimensions the official data files actually provide — the GUI's dimension spinbox now enforces this, see below).

Every function's shift vector, rotation matrix, and (where applicable) shuffle permutation are the **exact official data**, extracted from the official release and verified by recompiling the official C reference implementation locally (after fixing a `%Lf`/`%lf` scanf format bug in that reference code that otherwise corrupts data loading under Linux/GCC) and comparing outputs point-for-point. Two verified reference-code quirks are deliberately **reproduced rather than "fixed"**, since every published CEC2017 comparison table was computed against this exact behavior: the standalone F6 and part of the F14/F20 hybrids read a stale, unrotated/misaligned buffer instead of their own correctly-transformed input, and F8's "non-continuous" rounding step is silently overwritten before it can take effect, making F8 numerically identical to a plain rotated Rastrigin. Both are called out explicitly in the corresponding source files.

#### GUI dimension input now enforces what a problem actually supports
Some problems accept more than one dimension but not an arbitrary one — CEC2017 only supports D ∈ {10, 30, 50}, CEC2022 only D ∈ {10, 20}. The dimension spinbox previously had no notion of this and would happily accept (and pass to the CLI, where it would fail) any value in its general range. It now:
- Restricts the spinbox's range and step to the problem's actual allowed set once such a problem is selected (so the up/down arrows step directly between valid values, e.g. 10 → 30 → 50, rather than one-by-one through unsupported values in between).
- Snaps a manually typed value to the nearest allowed one once editing finishes, without interrupting mid-keystroke typing.
- Fully restores the standard unrestricted range when switching back to a scalable problem, so a prior restriction never lingers.

This is a separate mechanism from the existing single-fixed-dimension handling (e.g. `antennaarray`, `tnep`), which is unaffected.

#### Settings dialog simplification
**Reload settings** no longer asks for confirmation before reloading — it reloads immediately and shows only the existing "reloaded successfully" message, matching **Save settings**' single-message behavior.

---

## 2) Available methods and problems (CLI)

CLI syntax:

```bash
optimsolution <method> <problem>
```
or

```bash
optimsolution <method> <problem> <dimension>
```

### Methods (74)

#### Differential evolution and variants

| Short name | Full name |
|---|---|
| `arq` | ARQ: Adaptive RTR with Quarantine |
| `arq2` | ARQ2: ARQ/IDE roulette with Quarantine + ARQ-only micro-restart + jSO-style K |
| `arq3` | ARQ3: Extended ARQ variant |
| `awjso` | Adaptive-Weight jSO |
| `bjso` | Band-guided jSO |
| `bwjso` | Best-Worst corrected jSO |
| `de` | Differential Evolution |
| `ea4eig` | Evolutionary Algorithms with Eigen crossover |
| `jde` | Self-adaptive Differential Evolution |
| `jso` | Hybrid Differential Evolution JSO |
| `mjso` | Modified jSO |
| `mlshaderl` | Multi-operator L-SHADE with Reinforcement Learning |
| `nlshadelbc` | NL-SHADE-LBC |
| `pde` | Parallel Differential Evolution |
| `rde` | RDE: Reconstructed Differential Evolution |
| `rdex` | RDEx-SOP: Exploitation-Biased Reconstructed Differential Evolution (IEEE CEC 2025 SOP winner) |
| `sade` | Self-adaptive Differential Evolution (SaDE) |
| `sfcde` | Success-Failure Competitive Differential Evolution |
| `sparq` | SPARQ Optimizer |
| `tridentde` | TRIDENT Differential Evolution |
| `ude` | Unified Differential Evolution |
| `ude3` | Enhanced Unified Differential Evolution Algorithm 3 |
| `ujso` | Updated jSO |

#### Population-based, swarm, evolutionary, and hybrid optimizers

| Short name | Full name |
|---|---|
| `abc` | Artificial Bee Colony |
| `aco` | Ant Colony Optimization |
| `acor` | Ant Colony Optimization for Continuous Domains |
| `alo` | Ant Lion Optimizer |
| `ba` | Bat Algorithm |
| `bho` | BioHealing Optimization |
| `clpso` | Comprehensive Learning Particle Swarm Optimization |
| `cmaes` | Covariance Matrix Adaptation Evolution Strategy |
| `cs` | Cuckoo Search |
| `egco` | Eel and Grouper Optimizer |
| `eo` | Equilibrium Optimizer |
| `fa` | Firefly Algorithm |
| `ga` | Genetic Algorithm |
| `gahs` | Genetic Algorithm with Harmony Search |
| `gao` | Giant Armadillo Optimizer |
| `gsa` | Gravitational Search Algorithm |
| `gwo` | Grey Wolf Optimizer |
| `hades` | HADES Optimizer |
| `hba` | Honey Badger Algorithm |
| `hho` | Harris Hawks Optimization |
| `hs` | Harmony Search |
| `jaya` | JAYA |
| `kh` | Krill Herd |
| `lmcmaes` | Limited-Memory CMA-ES |
| `lracmaes` | Low-Rank Adaptation CMA-ES |
| `mewoa` | Modified Enhanced Whale Optimization Algorithm |
| `mfo` | Moth-Flame Optimization |
| `mpa` | Marine Predators Algorithm |
| `mscso` | Modified Sand Cat Swarm Optimization (multi-strategy fusion) |
| `mvo` | Multi-Verse Optimizer |
| `pga` | Parallel Genetic Algorithm |
| `ppso` | Parallel Particle Swarm Optimization |
| `psao` | Parallel Smell Agent Optimization |
| `psioa` | Parallel Sporulation-Inspired Optimization Algorithm |
| `pso` | Particle Swarm Optimization |
| `sa` | Simulated Annealing |
| `sao` | Smell Agent Optimization |
| `sca` | Sine Cosine Algorithm |
| `sioa` | Sporulation-Inspired Optimization Algorithm |
| `sma` | Slime Mould Algorithm |
| `so` | Snake Optimizer |
| `tlbo` | Teaching-Learning-Based Optimization |
| `wca` | Water Cycle Algorithm |
| `woa` | Whale Optimization Algorithm |

#### Local search methods

| Short name | Full name |
|---|---|
| `bfgs` | Broyden-Fletcher-Goldfarb-Shanno |
| `gd` | Gradient Descent |
| `lbfgs` | Limited-memory Broyden-Fletcher-Goldfarb-Shanno |
| `nm` | Nelder-Mead Simplex |

#### Multi-objective methods (used only in the Multi-objective optimization run mode)
| Short name | Full name |
|---|---|
| `nsga2` | NSGA-II (Non-dominated Sorting Genetic Algorithm II) |
| `moead` | MOEA/D-DE (decomposition-based, Tchebycheff scalarization) |
| `mopso` | Multi-Objective Particle Swarm Optimization |

### Problems (157)

#### Classical and synthetic benchmarks

`ackley`, `alpine1`, `attractivesector`, `beale`, `bohachevsky1`, `bohachevsky2`, `bohachevsky3`,
`booth`, `branin`, `bucherastrigin`, `bukinn6`, `camel`, `cigar`, `colville`, `cosinemixture`,
`crossintray`, `differentpowers`, `diracproblem`, `dixonprice`, `dropwave`, `easom`, `eggholder`,
`ellipsoidal`, `equalmaxima`, `expotential`, `gallagher101`, `gallagher21`, `goldstein`, `griewank`,
`griewankrosenbrock`, `hansen`, `hartmann3`, `hartmann6`, `holdertable`, `katsuura`, `langermann`,
`levy`, `lunacekbirastrigin`, `matyas`, `mccormick`, `michalewicz`, `perm`, `potential`, `powell`,
`rastrigin`, `rastrigin2`, `rosenbrock`, `rotatedrosenbrock`, `salomon`, `schaffer`, `schwefel`,
`shekel5`, `shekel7`, `shekel10`, `shubert`, `sinusoidal`, `sphere`, `stepellipsoidal`, `test2n`,
`test30n`, `trid`, `weierstrass`, `whitley`, `zakharov`

#### Niching / multimodal specialized benchmarks

`vincent`, `fiveunevenpeaktrap`
*(see also `equalmaxima`, `gallagher101`, `gallagher21`, `shubert` above — all originate from the same CEC-style niching/multimodal literature)*

#### Real-world and application-driven problems

`antennaarray`, `antennaula`, `bifunctionalcatalyst`, `cassini`, `cassini1`, `datacentercooling`,
`ded1`, `ded2`, `eld1`, `eld2`, `eld3`, `eld4`, `eld5`, `fmsynth`, `gascycle`, `gtoc1`,
`heatexchanger`, `himmelblau`, `hydrothermal`, `ik6dof`, `messenger`, `ofdmpower`, `polyphase`,
`portfoliomv`, `rosetta`, `sagas`, `smartportenergy`, `stirredtankreactor`, `tandem`, `tersoffb`,
`tersoffc`, `tnep`, `transmissionpricing`, `vibratingplatform`, `weatherirrigation`,
`wirelesscoverage`

#### Mechanical engineering design benchmarks

`weldedbeam`, `speedreducer`, `pressurevessel`, `springdesign`, `cantileverbeam`, `threebartruss`,
`geartrain`

#### GKLS test classes

`gkls250`, `gkls350`, `gkls2100`

#### CEC 2022 representative functions

`cec2022_zakharov`, `cec2022_rosenbrock`, `cec2022_schafferf7`,
`cec2022_noncontinuous_rastrigin`, `cec2022_levy`, `cec2022_hybrid2`, `cec2022_hybrid6`,
`cec2022_hybrid10`, `cec2022_composition1`, `cec2022_composition2`, `cec2022_composition6`,
`cec2022_composition7`

#### CEC 2017 functions (29 of 29 usable functions -- F2 was officially deleted by the benchmark's own authors)

`cec2017f1`, `cec2017f3`, `cec2017f4`, `cec2017f5`, `cec2017f6`, `cec2017f7`, `cec2017f8`,
`cec2017f9`, `cec2017f10`, `cec2017f11`, `cec2017f12`, `cec2017f13`, `cec2017f14`, `cec2017f15`,
`cec2017f16`, `cec2017f17`, `cec2017f18`, `cec2017f19`, `cec2017f20`, `cec2017f21`, `cec2017f22`,
`cec2017f23`, `cec2017f24`, `cec2017f25`, `cec2017f26`, `cec2017f27`, `cec2017f28`, `cec2017f29`,
`cec2017f30`

All shift vectors, rotation matrices, and shuffle permutations are the exact official data,
verified against a recompiled copy of the official reference implementation. Every function
supports **D = 10, 30, 50** (see the source comments in `src/problems/cec2017_f6.h`,
`cec2017_f8.h`, `cec2017_f14.h`, and `cec2017_f20.h` for the reference-code quirks that are
deliberately reproduced rather than "fixed").

#### Multi-objective benchmarks (used only in the Multi-objective optimization run mode)

`zdt1`, `zdt2`, `zdt3`, `zdt4`
*(see also `portfoliomv` above, under Real-world and application-driven problems -- it is
multi-objective too)*

---

## 3) Windows (Console + GUI)

> **Windows installer available:** Download and run **Optimsolution.exe** from:
>
> **[Optimsolution.exe (Windows Installer)](https://www.dit.uoi.gr/files/optimsolution.zip)**

### Prerequisites
- Visual Studio 2022 (Community or Build Tools)
  - Workload: **Desktop development with C++**
  - Components: **MSVC v143**, **Windows 10/11 SDK**
- CMake available in **PATH**
- **Qt 6.x (MSVC 2022 x64)** for the GUI target, e.g. `C:\Qt\6.10.1\msvc2022_64`

Recommended shell: **x64 Native Tools Command Prompt for VS 2022** or **Developer PowerShell for VS 2022**.

### Build and run (GUI + CLI) — Release

Delete any existing build directory.

```powershell
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
```

Configure with the GUI target enabled.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  "-DCMAKE_PROJECT_INCLUDE:FILEPATH=$PWD/cmake/optimsolution_gui.cmake" `
  "-DCMAKE_PREFIX_PATH=C:\Qt\6.10.1\msvc2022_64"
```

Build both targets.

```powershell
cmake --build build --config Release --target optimsolution_gui
cmake --build build --config Release --target optimsolution
```

Deploy Qt runtime next to the GUI executable.

```powershell
& "C:\Qt\6.10.1\msvc2022_64\bin\windeployqt.exe" `
  --no-translations --compiler-runtime `
  ".\build\Release\optimsolution_gui.exe"
```

Run.

```powershell
.\build\Release\optimsolution_gui.exe
.\build\Release\optimsolution.exe arq tersoffc
```

---

## 4) Linux (Console + GUI)

### Prerequisites
- GCC or Clang, CMake, Ninja (recommended)
- Qt 6.x development packages

### Build and run — Release

```bash
sudo apt update && sudo apt upgrade
sudo apt install qt6-base-dev qt6-base-dev-tools qt6-tools-dev qt6-tools-dev-tools
rm -rf build
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PROJECT_INCLUDE=cmake/optimsolution_gui.cmake
cmake --build build --target optimsolution_gui
cmake --build build --target optimsolution
./build/optimsolution_gui
./build/optimsolution arq tersoffc
```

---

## 5) Settings (`optimsolution.cfg`)

Both the CLI and GUI read experiment defaults from `optimsolution.cfg`. The GUI can apply additional runtime-only overrides (e.g. forcing CSV convergence output for plots) without modifying the file.

### Key sections

```ini
[global]
population       = 200
max_iters        = 500000
max_evals        = 150000
seed_base        = 4242
runs             = 30
success_tol      = 1e-8
local_rate       = 0.00
local_method     = lbfgs
end_local_refine = 0
end_local_method = lbfgs   ; bfgs / nm / gd
summary_enable   = 1
parallel_runs    = 0       ; 0 = serial, 1 = OpenMP-parallel independent runs (same results, either way)
omp_threads      = 0       ; 0 = auto (all cores), N = cap at N threads

[stop]
rule = maxevals            ; bss|wss|tss|boss|srs|irs|doublebox|maxevals|all|none
eps  = 1e-10

[init]
type = uniform             ; uniform|normal|cauchy|laplace|lognormal|exponential|beta|levy|lhs|halton|oppositional
```

A method section overrides any global key for that optimizer only.

```ini
[arq]
population   = 100
pbest_frac   = 0.12
F            = 0.1
CR           = 0.9
archive_rate = 1.5
batch_frac   = 0.5
local_rate   = 0.00
local_method = lbfgs
```

---

## 6) GUI overview

The GUI and CLI share the same optimization core. The GUI adds four run modes, convergence plotting, batch analysis, sensitivity studies, and a code wizard for extending the framework without leaving the application.

![optimsolution GUI](./docs/run1.png)
![optimsolution GUI](./docs/run2.png)
![optimsolution GUI](./docs/run3.png)

### Run modes

| Mode | Description |
|---|---|
| **Single run** | One run of the selected method on the selected problem. Produces a convergence plot and run summary. |
| **Batch run** | All selected methods × all selected problems for N runs. Results appear in eight analysis views: Results Table, Best Table, Mean Table, Best Ranking, Mean Ranking, Final Ranking, Pairwise Wilcoxon, Friedman Ranking. |
| **Sensitivity analysis of method parameters** | Sweeps method parameters over a defined range (problem fixed). Shows how each parameter value affects the result. |
| **Sensitivity analysis of problem parameters** | Sweeps problem parameters such as dimension or bounds (method fixed). Characterizes problem difficulty as a function of its own configuration. |
| **Multi-objective optimization** | Runs a Pareto-based method (`nsga2`, `moead`, `mopso`) against a multi-objective problem (`zdt1`-`zdt4`, `portfoliomv`) and plots the resulting Pareto front. Population and generations are read from `optimsolution.cfg`, not set in this panel. |

All four single-objective modes above can run their independent repetitions in parallel via OpenMP (see [§5](#5-settings-optimsolutioncfg)), with results that are numerically identical to a fully serial run.

### Why the GUI is significantly faster than running the CLI directly

Although both the GUI and CLI share the same optimization core, the GUI delivers substantially higher throughput for multi-run and batch experiments. The performance gap comes from several architectural decisions in the GUI:

**Zero inter-run overhead.** The GUI manages a job queue (`batchQueue_`) and launches each CLI process immediately after the previous one completes — with no manual setup, no re-reading of configuration files between jobs, and no human latency between runs. In a batch of *M* methods × *P* problems × *N* runs, all *M × P* jobs are dispatched sequentially and automatically without any user interaction.

**Runtime configuration snapshots.** Instead of writing and re-reading `optimsolution.cfg` on disk for every run, the GUI builds an in-memory configuration snapshot (`optimsolution_gui_merged.cfg`) and passes it to the CLI via `--config`. GUI edits (population size, budget, method parameters) act as runtime overrides and are applied instantly without touching the on-disk file. This eliminates repeated file-system I/O for configuration between jobs.

**Off-thread CSV post-processing.** After each run completes, CSV parsing, table population, and statistical aggregation (rankings, Wilcoxon, Friedman) are performed off the GUI thread via `QtConcurrent::run`. The GUI thread never blocks waiting for post-processing: it immediately starts the next job in the queue while the analysis runs in the background.

**Dedicated log-writer thread.** Batch run logging uses a dedicated `batchLogWriter_` thread so that writing progress to the UI log panel does not compete with the optimizer process for CPU or I/O time.

**High-frequency live monitoring.** UI progress updates are driven by timers: 250 ms for batch log flushing, 200 ms for single-run log flushing, and 150 ms for sensitivity CSV polling. This tight polling interval means convergence data and progress bars are updated in near real-time without any manual file inspection by the user.

**Automatic configuration consistency.** The GUI guarantees that every job in a batch uses exactly the same configuration snapshot. When running the CLI manually, configuration drift between runs (accidental edits to `.cfg`, different working directories, forgotten flags) is a common source of irreproducible results. The GUI eliminates this class of error entirely.

The practical effect is that a batch experiment that would take several hours of manual CLI orchestration — launching processes, checking outputs, updating configuration, re-running failed jobs — can be completed in the same wall-clock time as the computations themselves, with no idle time between jobs. With OpenMP parallel runs enabled on top of this, the computations themselves can now also make use of all available CPU cores.

### Code Wizard
The Code Wizard panel (bottom-right) generates skeleton `.h` and `.cpp` files for new methods or problems, patches `factory.cpp` and `CMakeLists.txt` automatically, and triggers a rebuild on the next startup.

For full details see the English manual: [optimsolution_manual_EN.pdf](./docs/optimsolutionManual_EN.pdf)

---

## 7) Example run console output

The following screenshot shows the last iterations of **Run 30** and the final summary for:

- **Method:** `arq` · **Problem:** `tersoffc (Dim=24)` · **Runs:** 30 · **Budget:** 150 000 evals/run
- CLI syntax:
```bash
optimsolution arq tersoffc
```

![Console output: ARQ on tersoffc (Dim=24)](./docs/arq_tersoffc.png)
