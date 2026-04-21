# solver/projection.py
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class ProjectionMethod:
    def __init__(self, config, grid):
        self.config = config
        self.grid = grid
        self.setup_poisson_matrix()
    
    def setup_poisson_matrix(self):
        """Setup sparse matrix for Poisson equation"""
        Nx, Ny = self.grid.Nx, self.grid.Ny
        dx, dy = self.grid.dx, self.grid.dy
        
        N = Nx * Ny
        
        # Coefficients
        cx = 1.0 / dx**2
        cy = 1.0 / dy**2
        cc = -2.0 * (cx + cy)
        
        # Build sparse matrix
        diagonals = [
            np.ones(N) * cc,      # main diagonal
            np.ones(N-1) * cx,    # upper diagonal
            np.ones(N-1) * cx,    # lower diagonal
            np.ones(N-Nx) * cy,   # upper Nx diagonal
            np.ones(N-Nx) * cy    # lower Nx diagonal
        ]
        
        self.A = diags(diagonals, [0, 1, -1, Nx, -Nx], format='csr')
    
    def solve_pressure(self, u_star, v_star, dt):
        """Solve Poisson equation for pressure"""
        dx, dy = self.grid.dx, self.grid.dy
        Ny, Nx = u_star.shape
        
        # Compute divergence
        div = np.zeros_like(u_star)
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                div[j,i] = ((u_star[j,i+1] - u_star[j,i-1])/(2*dx) + 
                           (v_star[j+1,i] - v_star[j-1,i])/(2*dy)) / dt
        
        # Solve
        rhs = div.flatten()
        p_flat = spsolve(self.A, rhs)
        p = p_flat.reshape((Ny, Nx))
        
        return p
    
    def project_velocity(self, u_star, v_star, p, dt):
        """Project velocity to divergence-free field"""
        dx, dy = self.grid.dx, self.grid.dy
        Ny, Nx = u_star.shape
        
        u = u_star.copy()
        v = v_star.copy()
        
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                u[j,i] -= dt * (p[j,i+1] - p[j,i-1]) / (2*dx)
                v[j,i] -= dt * (p[j+1,i] - p[j-1,i]) / (2*dy)
        
        return u, v
