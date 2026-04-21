# utils/validation.py
import numpy as np
import matplotlib.pyplot as plt


class Validation:
    def __init__(self, config):
        self.config = config
        
        # Ghia et al. (1982) benchmark data for Re=100
        self.ghia_re100_u = {
            'y': [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 
                  0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 
                  0.9688, 0.9766, 1.0000],
            'u': [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, 
                  -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 
                  0.68717, 0.73722, 0.78871, 0.84123, 1.00000]
        }
        
        self.ghia_re100_v = {
            'x': [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 
                  0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 
                  0.9609, 0.9688, 1.0000],
            'v': [0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 
                  0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, 
                  -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]
        }
    
    def compare_with_ghia(self, x, y, u, v, Re):
        """Compare results with Ghia benchmark"""
        if Re != 100:
            print(f"⚠ Ghia data available only for Re=100, current Re={Re}")
            return
        
        Ny, Nx = u.shape
        mid_x = Nx // 2
        mid_y = Ny // 2
        
        # Extract profiles
        u_centerline = u[:, mid_x]
        v_centerline = v[mid_y, :]
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # U velocity
        ax1.plot(u_centerline, y, 'b-', linewidth=2, label='Current')
        ax1.plot(self.ghia_re100_u['u'], self.ghia_re100_u['y'], 
                'ro', markersize=6, label='Ghia et al. (1982)')
        ax1.set_xlabel('U velocity')
        ax1.set_ylabel('y')
        ax1.set_title(f'U profile at x=0.5 (Re={Re})')
        ax1.legend()
        ax1.grid(True)
        
        # V velocity
        ax2.plot(x, v_centerline, 'r-', linewidth=2, label='Current')
        ax2.plot(self.ghia_re100_v['x'], self.ghia_re100_v['v'], 
                'bo', markersize=6, label='Ghia et al. (1982)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('V velocity')
        ax2.set_title(f'V profile at y=0.5 (Re={Re})')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/validation_ghia_re{Re}.png", dpi=150)
        plt.close()
        
        print(f"✓ Validation plot saved")
