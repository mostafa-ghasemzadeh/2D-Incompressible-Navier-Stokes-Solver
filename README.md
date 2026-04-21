# 2D Incompressible Navier-Stokes Solver

A finite-difference solver for 2D incompressible lid-driven cavity flow using the pressure projection method. Includes multiple time integration schemes and validation against Ghia et al. (1982) benchmark data.

## Features

- **Multiple time integration schemes**: Euler, Adams-Bashforth 2, Runge-Kutta 3
- **Flexible pressure solvers**: Direct sparse solver or multigrid V-cycle
- **Adaptive time stepping**: CFL-based $\Delta t$ adjustment
- **Comprehensive output**: Field data (CSV), contour plots, streamlines, velocity profiles
- **Benchmark validation**: Comparison with Ghia et al. (1982) for Re=100
- **Steady-state detection**: Automatic convergence monitoring

## Installation

pip install numpy scipy matplotlib

## Quick Start

python
from config import Config
from main import run

# Create configuration
config = Config(
Nx=128, Ny=128,
Re=100,
dt=0.001,
t_end=50.0,
time_scheme='rk3',
output_interval=500
)

# Run simulation
run(config)

## Project Structure


.
├── config.py              # Configuration parameters
├── grid.py                # Computational mesh
├── boundary_conditions.py # Boundary condition enforcement
├── navier_stokes.py       # Momentum equation terms
├── projection.py          # Pressure projection (direct solver)
├── multigrid.py          # Multigrid Poisson solver
├── time_integrator.py    # Time advancement schemes
├── output.py             # Visualization and data export
├── validation.py         # Benchmark comparison
└── main.py               # Main simulation loop

## Module Overview

### `config.py`
Central configuration using Python dataclass. Groups parameters into:
- Domain/grid settings
- Flow physics (Reynolds number, lid velocity)
- Time integration
- Convergence criteria
- Pressure solver options
- Output and validation settings

### `grid.py`
Uniform Cartesian mesh with:
- Grid dimensions: `Nx`, `Ny`
- Domain size: `Lx`, `Ly`
- Grid spacing: `dx`, `dy`
- Coordinate arrays: `x`, `y`, `X`, `Y`

### `boundary_conditions.py`
Enforces boundary conditions:
- **Velocity**: No-slip walls + moving top lid
- **Pressure**: Neumann (zero normal gradient)

### `navier_stokes.py`
Computes momentum equation terms:
- `compute_rhs()`: Convection + diffusion RHS
- `compute_vorticity()`: Scalar vorticity field

### `projection.py`
Pressure projection method (direct solver):
- `setup_poisson_matrix()`: 5-point Laplacian stencil
- `solve_pressure()`: Solves $\nabla^2 p = \frac{1}{\Delta t}\nabla \cdot u^*$
- `project_velocity()`: Corrects velocity to enforce incompressibility

### `multigrid.py`
Alternative iterative Poisson solver:
- V-cycle multigrid with Gauss-Seidel smoothing
- Better scaling for large grids (>256²)

### `time_integrator.py`
Three time advancement schemes:
- **Euler**: 1st-order explicit
- **AB2**: 2nd-order Adams-Bashforth
- **RK3**: 3rd-order Runge-Kutta

Each step: compute RHS → solve pressure → project velocity → apply BCs

### `output.py`
Visualization and data export:
- `save_fields()`: CSV output (u, v, p)
- `plot_contours()`: Velocity magnitude + vorticity
- `plot_streamlines()`: Streamfunction visualization
- `plot_profiles()`: Centerline velocity profiles

### `validation.py`
Benchmark comparison with Ghia et al. (1982):
- Embedded reference data for Re=100
- Extracts centerline profiles
- Computes L2 error
- Generates comparison plots

### `main.py`
Main simulation orchestration:
- Adaptive CFL time stepping
- Lid velocity ramping
- Convergence monitoring
- NaN/Inf checks
- Periodic output and validation

## Numerical Methods

### Spatial Discretization
2nd-order centered finite differences on uniform Cartesian grid

### Temporal Schemes
- **Euler**: 1st-order, stable for small $\Delta t$
- **AB2**: 2nd-order, requires history storage
- **RK3**: 3rd-order, three substeps per time step

### Incompressibility Enforcement
Fractional-step projection method:
1. **Predictor**: Advance momentum without pressure
2. **Poisson solve**: Compute pressure to enforce $\nabla \cdot u = 0$
3. **Corrector**: Update velocity with pressure gradient

### Boundary Conditions
- **Walls**: No-slip ($u=v=0$)
- **Top lid**: $u=U_{\text{lid}}$, $v=0$
- **Pressure**: Neumann (zero normal gradient)

## Configuration Parameters

### Domain
python
Nx, Ny = 128, 128  # Grid points
Lx, Ly = 1.0, 1.0  # Domain size

### Physics
python
Re = 100           # Reynolds number
U_lid = 1.0        # Lid velocity

### Time Integration
python
dt = 0.001                    # Time step (or computed from CFL)
t_end = 50.0                  # Simulation end time
time_scheme = 'rk3'           # 'euler', 'ab2', or 'rk3'
CFL_max = 0.5                 # Adaptive time stepping (optional)

### Convergence
python
steady_tol = 1e-6                    # L∞ tolerance for steady state
check_convergence_interval = 100     # Steps between checks

### Pressure Solver
python
pressure_solver = 'direct'    # 'direct' or 'multigrid'
multigrid_levels = 3          # For multigrid solver
multigrid_vcycles = 2

### Output
python
output_dir = 'output'
output_interval = 500
save_fields = True
plot_contours = True
plot_streamlines = True
plot_profiles = True

### Validation
python
validate_ghia = True
Re_ghia = 100

## Output Files


output/
├── fields/
│   ├── u_000500.csv
│   ├── v_000500.csv
│   └── p_000500.csv
└── plots/
├── contours_000500.png
├── streamlines_000500.png
├── profiles_000500.png
└── validation_ghia.png

## Validation

For Re=100, the solver compares centerline velocity profiles with Ghia et al. (1982):
- $u$ along vertical centerline ($x=0.5$)
- $v$ along horizontal centerline ($y=0.5$)
- Computes L2 error and generates comparison plots

## Typical Reynolds Numbers

| Re | Flow Regime | Characteristics |
|----|-------------|-----------------|
| 100 | Steady | Single primary vortex |
| 400 | Steady | Secondary corner vortices |
| 1000 | Steady/weakly unsteady | Complex vortex structure |
| >5000 | Unsteady | Turbulent transition |

## Performance Notes

- **Direct solver**: Fast for small grids (<256²), memory-intensive for large grids
- **Multigrid**: Better scaling for large grids (>256²)
- **RK3 scheme**: Best accuracy but 3× cost per step vs Euler
- **Adaptive CFL**: Maintains stability, adjusts $\Delta t$ based on max velocity

## Example: High Reynolds Number Simulation

python
config = Config(
Nx=256, Ny=256,
Re=1000,
CFL_max=0.5,              # Adaptive time stepping
t_end=100.0,
time_scheme='rk3',
pressure_solver='multigrid',
output_interval=1000,
steady_tol=1e-6,
check_convergence_interval=100
)

run(config)

## References

- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

## License

MIT License
