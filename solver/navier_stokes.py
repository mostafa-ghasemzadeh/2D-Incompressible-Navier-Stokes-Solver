# solver/navier_stokes.py
import numpy as np


class NavierStokes:
    def __init__(self, config, grid):
        self.config = config
        self.grid = grid
        self.Re = config.Re
        self.nu = 1.0 / self.Re
    
    def compute_rhs(self, u, v):
        """Compute RHS of momentum equations"""
        dx, dy = self.grid.dx, self.grid.dy
        Ny, Nx = u.shape
        
        rhs_u = np.zeros_like(u)
        rhs_v = np.zeros_like(v)
        
        # Interior points
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                # Convection (upwind)
                u_conv = u[j,i] * (u[j,i] - u[j,i-1])/dx + v[j,i] * (u[j,i] - u[j-1,i])/dy
                v_conv = u[j,i] * (v[j,i] - v[j,i-1])/dx + v[j,i] * (v[j,i] - v[j-1,i])/dy
                
                # Diffusion (central)
                u_diff = self.nu * ((u[j,i+1] - 2*u[j,i] + u[j,i-1])/dx**2 + 
                                    (u[j+1,i] - 2*u[j,i] + u[j-1,i])/dy**2)
                v_diff = self.nu * ((v[j,i+1] - 2*v[j,i] + v[j,i-1])/dx**2 + 
                                    (v[j+1,i] - 2*v[j,i] + v[j-1,i])/dy**2)
                
                rhs_u[j,i] = -u_conv + u_diff
                rhs_v[j,i] = -v_conv + v_diff
        
        return rhs_u, rhs_v
    
    def compute_vorticity(self, u, v):
        """Compute vorticity field"""
        dx, dy = self.grid.dx, self.grid.dy
        Ny, Nx = u.shape
        
        omega = np.zeros_like(u)
        
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                omega[j,i] = (v[j,i+1] - v[j,i-1])/(2*dx) - (u[j+1,i] - u[j-1,i])/(2*dy)
        
        return omega
