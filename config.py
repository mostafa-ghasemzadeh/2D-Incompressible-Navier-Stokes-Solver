from dataclasses import dataclass

@dataclass
class Config:
    # Domain
    Lx: float = 1.0
    Ly: float = 1.0
    
    # Grid
    Nx: int = 64
    Ny: int = 64
    
    # Flow
    Re: float = 100.0
    U_lid: float = 1.0
    
    # Time
    dt: float = 0.001
    t_end: float = 50.0
    time_scheme: str = 'euler'  # 'euler', 'ab2', 'rk3'
    CFL_max: float = 0.5
    
    # Convergence
    steady_tol: float = 1e-6  # tolerance for steady-state
    check_convergence_interval: int = 100  # check every N steps
    
    # Solver
    pressure_solver: str = 'direct'  # 'direct' or 'multigrid'
    
    # Output
    output_dir: str = 'output'
    output_interval: int = 500
    save_fields: bool = True
    plot_contours: bool = True
    plot_streamlines: bool = True
    plot_profiles: bool = True
    validate_ghia: bool = True
    
    # Video
    save_video: bool = True
    video_fps: int = 10
