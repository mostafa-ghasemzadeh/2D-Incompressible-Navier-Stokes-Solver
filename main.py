import numpy as np
import sys
from config import Config
from solver.grid import Grid
from solver.boundary_conditions import BoundaryConditions
from solver.navier_stokes import NavierStokes
from solver.projection import ProjectionMethod
from solver.time_integrator import TimeIntegrator
from output import Output
from utils.validation import Validation

def lid_velocity(t, t_ramp=2.0):
    """Soft-start for lid velocity"""
    return min(1.0, t / t_ramp)

def check_convergence(u, v, u_old, v_old, tol):
    """Check if solution has converged to steady state"""
    du = np.max(np.abs(u - u_old))
    dv = np.max(np.abs(v - v_old))
    return max(du, dv) < tol

def run(config):
    print("="*60)
    print("Lid-Driven Cavity Flow Simulation")
    print("="*60)
    print(f"Re = {config.Re}")
    print(f"Grid = {config.Nx} x {config.Ny}")
    print(f"Time scheme = {config.time_scheme}")
    print(f"Convergence tolerance = {config.steady_tol}")
    print("="*60)
    
    # Initialize grid
    grid = Grid(config)
    
    # Initialize fields
    u = np.zeros((config.Nx, config.Ny))
    v = np.zeros((config.Nx, config.Ny))
    p = np.zeros((config.Nx, config.Ny))
    
    # For convergence check
    u_old = u.copy()
    v_old = v.copy()
    
    # Initialize components
    bc = BoundaryConditions(config)
    ns = NavierStokes(config, grid)
    projection = ProjectionMethod(config, grid)
    integrator = TimeIntegrator(config, grid, ns, projection, bc)
    output = Output(config)
    
    # Time loop
    t = 0.0
    step = 0
    dt = config.dt
    converged = False
    
    print("\nStarting time integration...")
    
    while t < config.t_end and not converged:
        # CFL check
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        if u_max > 0 or v_max > 0:
            dt_cfl = config.CFL_max * min(grid.dx / (u_max + 1e-12), 
                                          grid.dy / (v_max + 1e-12))
            dt = min(dt, dt_cfl)
        
        # Soft-start lid velocity
        lid_vel = lid_velocity(t)
        bc.apply_velocity(u, v, lid_velocity=lid_vel)
        
        # Time step
        u, v, p = integrator.step(u, v, p, dt)
        
        t += dt
        step += 1
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)) or np.any(~np.isfinite(p)):
            print(f"\n[ERROR] NaN/Inf detected at step {step}, t={t:.4f}")
            print("Simulation stopped.")
            sys.exit(1)
        
        # Convergence check
        if step % config.check_convergence_interval == 0:
            if check_convergence(u, v, u_old, v_old, config.steady_tol):
                print(f"\n✓ Converged at step {step}, t={t:.4f}")
                converged = True
            u_old = u.copy()
            v_old = v.copy()
        
        # Output
        if step % config.output_interval == 0 or converged:
            print(f"Step {step:6d} | t = {t:8.4f} | dt = {dt:.6f} | max(|u|) = {np.max(np.abs(u)):.6f}")
            
            vorticity = ns.compute_vorticity(u, v)
            
            if config.save_fields:
                output.save_fields(grid.x, grid.y, u, v, p, step)
            
            if config.plot_contours:
                output.plot_contours(grid.x, grid.y, u, v, p, vorticity, step)
            
            if config.plot_streamlines:
                output.plot_streamlines(grid.x, grid.y, u, v, step)
            
            if config.plot_profiles:
                output.plot_profiles(grid.x, grid.y, u, v, step)
    

    print("\n" + "="*60)
    if converged:
        print(f"Simulation converged at t={t:.4f}")
    else:
        print(f"Simulation completed at t={t:.4f}")
    print("="*60)
    
    # Validation
    if config.validate_ghia:
        print("\nValidating against Ghia et al. (1982)...")
        validator = Validation(config)
        validator.compare_with_ghia(grid.x, grid.y, u, v, config.Re)


if __name__ == "__main__":
    config = Config()

    run(config)
