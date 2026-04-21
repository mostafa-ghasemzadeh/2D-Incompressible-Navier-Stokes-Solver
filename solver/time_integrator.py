# solver/time_integrator.py
import numpy as np


class TimeIntegrator:
    def __init__(self, config, grid, ns, projection, bc):
        self.config = config
        self.grid = grid
        self.ns = ns
        self.projection = projection
        self.bc = bc
        self.scheme = config.time_scheme
        
        # For multi-step schemes
        self.rhs_u_old = None
        self.rhs_v_old = None
    
    def step(self, u, v, p, dt):
        """Advance one time step"""
        if self.scheme == 'euler':
            return self.euler_step(u, v, p, dt)
        elif self.scheme == 'ab2':
            return self.ab2_step(u, v, p, dt)
        elif self.scheme == 'rk3':
            return self.rk3_step(u, v, p, dt)
        else:
            raise ValueError(f"Unknown time scheme: {self.scheme}")
    
    def euler_step(self, u, v, p, dt):
        """Forward Euler with projection"""
        # Compute RHS
        rhs_u, rhs_v = self.ns.compute_rhs(u, v)
        
        # Predictor step
        u_star = u + dt * rhs_u
        v_star = v + dt * rhs_v
        
        self.bc.apply_velocity(u_star, v_star)
        
        # Solve pressure
        p = self.projection.solve_pressure(u_star, v_star, dt)
        self.bc.apply_pressure(p)
        
        # Corrector step
        u_new, v_new = self.projection.project_velocity(u_star, v_star, p, dt)
        self.bc.apply_velocity(u_new, v_new)
        
        return u_new, v_new, p
    
    def ab2_step(self, u, v, p, dt):
        """Adams-Bashforth 2nd order"""
        rhs_u, rhs_v = self.ns.compute_rhs(u, v)
        
        if self.rhs_u_old is None:
            # First step: use Euler
            u_star = u + dt * rhs_u
            v_star = v + dt * rhs_v
        else:
            # AB2
            u_star = u + dt * (1.5 * rhs_u - 0.5 * self.rhs_u_old)
            v_star = v + dt * (1.5 * rhs_v - 0.5 * self.rhs_v_old)
        
        self.bc.apply_velocity(u_star, v_star)
        
        p = self.projection.solve_pressure(u_star, v_star, dt)
        self.bc.apply_pressure(p)
        
        u_new, v_new = self.projection.project_velocity(u_star, v_star, p, dt)
        self.bc.apply_velocity(u_new, v_new)
        
        # Store for next step
        self.rhs_u_old = rhs_u
        self.rhs_v_old = rhs_v
        
        return u_new, v_new, p
    
    def rk3_step(self, u, v, p, dt):
        """3rd order Runge-Kutta"""
        # Stage 1
        rhs_u1, rhs_v1 = self.ns.compute_rhs(u, v)
        u1 = u + dt * rhs_u1
        v1 = v + dt * rhs_v1
        self.bc.apply_velocity(u1, v1)
        
        # Stage 2
        rhs_u2, rhs_v2 = self.ns.compute_rhs(u1, v1)
        u2 = 0.75*u + 0.25*u1 + 0.25*dt*rhs_u2
        v2 = 0.75*v + 0.25*v1 + 0.25*dt*rhs_v2
        self.bc.apply_velocity(u2, v2)
        
        # Stage 3
        rhs_u3, rhs_v3 = self.ns.compute_rhs(u2, v2)
        u_star = (1.0/3.0)*u + (2.0/3.0)*u2 + (2.0/3.0)*dt*rhs_u3
        v_star = (1.0/3.0)*v + (2.0/3.0)*v2 + (2.0/3.0)*dt*rhs_v3
        self.bc.apply_velocity(u_star, v_star)
        
        # Projection
        p = self.projection.solve_pressure(u_star, v_star, dt)
        self.bc.apply_pressure(p)
        
        u_new, v_new = self.projection.project_velocity(u_star, v_star, p, dt)
        self.bc.apply_velocity(u_new, v_new)
        
        return u_new, v_new, p
