# solver/grid.py
import numpy as np


class Grid:
    def __init__(self, config):
        self.Nx = config.Nx
        self.Ny = config.Ny
        self.Lx = config.Lx
        self.Ly = config.Ly
        
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        
        self.X, self.Y = np.meshgrid(self.x, self.y)
