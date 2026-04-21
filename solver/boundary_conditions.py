# solver/boundary_conditions.py
import numpy as np


class BoundaryConditions:
    def __init__(self, config):
        self.config = config
        self.U_lid = config.U_lid
    
    def apply_velocity(self, u, v, lid_velocity=1.0):
        """Apply boundary conditions for velocity field"""
        # Bottom wall (y=0): no-slip
        u[0, :] = 0.0
        v[0, :] = 0.0
        
        # Top wall (y=H): moving lid
        u[-1, :] = self.U_lid * lid_velocity
        v[-1, :] = 0.0
        
        # Left wall (x=0): no-slip
        u[:, 0] = 0.0
        v[:, 0] = 0.0
        
        # Right wall (x=L): no-slip
        u[:, -1] = 0.0
        v[:, -1] = 0.0
    
    def apply_pressure(self, p):
        """Apply Neumann BC for pressure (zero gradient at walls)"""
        # Bottom
        p[0, :] = p[1, :]
        # Top
        p[-1, :] = p[-2, :]
        # Left
        p[:, 0] = p[:, 1]
        # Right
        p[:, -1] = p[:, -2]
