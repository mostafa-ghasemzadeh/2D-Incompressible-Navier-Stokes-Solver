import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
class Output:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.output_dir  # تغییر از config.get به config.output_dir
        
        # Create output directories
        Path(f'{self.output_dir}/fields').mkdir(parents=True, exist_ok=True)
        Path(f'{self.output_dir}/plots').mkdir(parents=True, exist_ok=True)
    
    def save_fields(self, x, y, u, v, p, step):
        """Save velocity and pressure fields to CSV"""
        X, Y = np.meshgrid(x, y)
        data = np.column_stack([
            X.ravel(),
            Y.ravel(),
            u.ravel(),
            v.ravel(),
            p.ravel()
        ])
        
        header = 'x,y,u,v,p'
        filename = f'{self.output_dir}/fields/step_{step:06d}.csv'
        np.savetxt(filename, data, delimiter=',', header=header, comments='')
    
    def plot_contours(self, x, y, u, v, p, vorticity, step):
        """Plot contours of velocity, pressure, and vorticity"""
        X, Y = np.meshgrid(x, y)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # U velocity
        im1 = axes[0, 0].contourf(X, Y, u, levels=20, cmap='RdBu_r', origin='lower')
        axes[0, 0].set_title(f'U Velocity (Step {step})')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # V velocity
        im2 = axes[0, 1].contourf(X, Y, v, levels=20, cmap='RdBu_r', origin='lower')
        axes[0, 1].set_title(f'V Velocity (Step {step})')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Pressure
        im3 = axes[1, 0].contourf(X, Y, p, levels=20, cmap='viridis', origin='lower')
        axes[1, 0].set_title(f'Pressure (Step {step})')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Vorticity
        im4 = axes[1, 1].contourf(X, Y, vorticity, levels=20, cmap='RdBu_r', origin='lower')
        axes[1, 1].set_title(f'Vorticity (Step {step})')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/contours_{step:06d}.png', dpi=150)
        plt.close()
    
    def plot_streamlines(self, x, y, u, v, step):
        """Plot streamlines of velocity field"""
        X, Y = np.meshgrid(x, y)
        
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)):
            print(f"Warning: Non-finite values in velocity at step {step}")
            return
        
        speed = np.sqrt(u**2 + v**2)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        strm = ax.streamplot(X, Y, u, v, color=speed, cmap='viridis', 
                            density=1.5, linewidth=1)
        ax.set_title(f'Streamlines (Step {step})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(strm.lines, ax=ax, label='Speed')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/streamlines_{step:06d}.png', dpi=150)
        plt.close()
    
    def plot_profiles(self, x, y, u, v, step):
        """Plot velocity profiles along centerlines"""
        mid_x = len(x) // 2
        mid_y = len(y) // 2
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(u[:, mid_x], y, 'b-', linewidth=2)
        ax1.set_xlabel('U velocity')
        ax1.set_ylabel('y')
        ax1.set_title(f'U profile at x={x[mid_x]:.2f} (Step {step})')
        ax1.grid(True)
        
        ax2.plot(x, v[mid_y, :], 'r-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('V velocity')
        ax2.set_title(f'V profile at y={y[mid_y]:.2f} (Step {step})')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/profiles_{step:06d}.png', dpi=150)
        plt.close()
