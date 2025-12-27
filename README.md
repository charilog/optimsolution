# Build & Run (Debug) + Configuration

---

## 1) Windows (Debug)

### Prerequisites
- Visual Studio 2022 (Community or Build Tools)
  - Workload: Desktop development with C++
  - Components: MSVC v143, Windows 10/11 SDK, CMake Tools (optional but recommended)
- CMake (if not installed via Visual Studio): ensure it is available in your system PATH
- (Optional) Ninja for faster builds

### Configure / Build / Run (Debug)
```bat
1] cd /path/to/optimsolution
2] cmake -S . -B build -G "Visual Studio 17 2022" -A x64
3] cmake --build build --config Debug
4] cd build
5] .\Debug\optimsolution jso rastrigin 30
```

---

## 2) Linux (Debug)

### Install
```bash
sudo apt update
sudo apt install -y build-essential cmake gdb
sudo apt install -y ninja-build
```

### Configure / Build / Run (Debug)
```bash
1] cd /path/to/optimsolution
2] cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
3] cmake --build build -j
4] cd build
5] ./optimsolution jso rastrigin 30
```

---

## 3) Settings used (optimsolution.cfg)

The executable reads experiment defaults from `optimsolution.cfg`.  
The **[global]** section defines defaults, while a **method section** (e.g., **[arq]**) can **override** keys such as `population`, `local_rate`, etc.

### Key defaults
```ini
[global]
population = 200
max_iters  = 500000
max_evals  = 150000
seed_base  = 4242
runs       = 30
success_tol = 1e-8
local_rate   = 0.00
local_method = lbfgs
end_local_refine = 0
end_local_method = lbfgs   ; or bfgs / nm / gd
summary_enable           = 1

[stop]
rule    = maxevals         ; bss|wss|tss|boss|srs|irs|doublebox|maxevals|all|none
eps     = 1e-10

[init]
type = uniform             ; uniform | normal | cauchy | laplace | lognormal | exponential | beta | levy | lhs | halton | oppositional
```

### ARQ method overrides (used in the attached run)
```ini
[arq]
population 	 = 100
pbest_frac   = 0.12
F            = 0.1
CR           = 0.9
archive_rate = 1.5
batch_frac   = 0.5
local_rate   = 0.00
local_method = lbfgs
```

### How these settings map to the run summary
- **Population**
  - Default: `global.population = 200`
  - **ARQ override:** `arq.population = 100` → shown as `Population: 100`
- **Runs**
  - `global.runs = 30` → shown as `Runs: 30`
- **Stopping rule / evaluation budget**
  - `stop.rule = maxevals`
  - `global.max_evals = 150000` → shown as `Max evals/run: 150000` and the run stops at `evals 150000`
- **Maximum iterations**
  - `global.max_iters = 500000` → shown as `Max iters/run: 500000`
- **Initialization**
  - `init.type = uniform` → shown as `Init distribution: uniform`
- **Local search**
  - `local_rate = 0.00` (global and ARQ) → shown as `Local search: none`

---

## 4) Example run output (end of execution)

The following screenshot shows the **last iterations of Run 30** and the **final run summary** for:
- **Method:** `arq`
- **Problem:** `tersoffc (Dim=24)`
- **Runs:** 30
- **Budget:** 150000 evals/run

![Console output: ARQ on tersoffc (Dim=24)](arq_tersoffc.png)

### What the last iteration lines mean
Lines like:
- `iter 2495 | evals 150000 | best_f = -32.013207...`

indicate:
- **iter**: iteration counter,
- **evals**: total function evaluations used so far,
- **best_f**: best objective value found so far in that run.

The flat `best_f` in the last lines indicates no improvement near the end, and the run stops because it reaches the configured evaluation budget (`maxevals` / `max_evals`).

### How to read the RUN SUMMARY (quick)
- **Best f (min)**: best result over all runs.
- **Best f (mean ± sd)**: average performance (and variability) over runs.
- **Median, Q1–Q3**: robust distribution summary.
- **Success rate**: how many runs satisfy the success criterion (here `1/30`).
- **Timing & memory**: runtime statistics and memory footprint.
